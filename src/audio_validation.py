"""Audit speech and noise files and mark each one valid, review or invalid.

Checks add flags (`audit_flags`); speech files in review are then decided by the rules in
`speech_audit.auto_review`. Manual decisions in manual_review_decisions.csv override both
(columns: manifest, id, decision, note).

Usage:
    python -m src.audio_validation --target speech|noise|all|noise-speech [--summary-only]
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf

from src.utils import get_logger, load_config, manifest_path, project_path, save_json

log = get_logger("audio_validation")

DECISIONS_FILE = "manual_review_decisions.csv"
EPS = 1e-12
RATE_REASONS = {"too_short", "too_long", "speaking_rate_too_slow", "speaking_rate_too_fast"}


def signal_stats(audio: np.ndarray, clip_peak: float) -> dict:
    """Amplitude statistics used for the audit (audio: mono float32)."""
    if audio.size == 0:
        return {"num_samples": 0, "rms_dbfs": float("-inf"), "peak": 0.0, "clipped_fraction": 0.0}
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return {
        "num_samples": int(audio.size),
        "rms_dbfs": round(20 * np.log10(rms + EPS), 2),
        "peak": round(float(np.max(np.abs(audio))), 5),
        "clipped_fraction": round(float(np.mean(np.abs(audio) >= clip_peak)), 6),
    }


def inspect_file(path, clip_peak: float) -> dict:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # unreadable / corrupt
        return {"readable": False, "read_error": f"{type(exc).__name__}: {exc}"[:200]}
    stats = signal_stats(audio.mean(axis=1), clip_peak)
    stats.update({"readable": True, "read_error": "", "actual_sample_rate": sr,
                  "actual_channels": audio.shape[1],
                  "actual_duration_sec": round(audio.shape[0] / sr, 4)})
    return stats


def _inspect_all(paths: list, clip_peak: float, workers: int) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: inspect_file(p, clip_peak), paths, chunksize=64))
    return pd.DataFrame(results)


def _apply_manual_decisions(df: pd.DataFrame, manifest: str, id_col: str, cfg: dict) -> pd.DataFrame:
    path = manifest_path(cfg, DECISIONS_FILE)
    if not path.exists():
        pd.DataFrame(columns=["manifest", "id", "decision", "note"]).to_csv(path, index=False)
        return df
    dec = pd.read_csv(path, keep_default_na=False)
    dec = dec[dec["manifest"] == manifest]
    bad = set(dec["decision"]) - {"valid", "invalid"}
    if bad:
        raise ValueError(f"Invalid decisions in {path}: {bad}")
    for _, row in dec.iterrows():
        mask = df[id_col] == row["id"]
        if not mask.any():
            log.warning("Manual decision for unknown %s id %s", manifest, row["id"])
            continue
        df.loc[mask, "quality_status"] = row["decision"]
        df.loc[mask, "manual_decision"] = f"{row['decision']}: {row['note']}"
        df.loc[mask, "exclusion_reason"] = "manual_review" if row["decision"] == "invalid" else ""
    log.info("Applied %d manual %s decisions", len(dec), manifest)
    return df


def _auto_review_speech(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Decide "review" files flagged only for clipping or duration/speaking rate, using the rules in
    the config instead of listening to each file. The measured values go to `auto_review`."""
    from src.noise_mixing import active_speech_mask

    r = cfg["speech_audit"].get("auto_review") or {}
    if not r.get("enabled"):
        return df
    mcfg = cfg["mixing"]
    root = project_path(cfg["paths"]["speech_root"])
    df["auto_review"] = ""
    for idx in df.index[df["quality_status"] == "review"]:
        reasons = set(df.at[idx, "audit_flags"].split(";"))
        if reasons - RATE_REASONS - {"possible_clipping"}:
            continue
        decision, notes, causes = "valid", [], []
        if "possible_clipping" in reasons:
            frac = float(df.at[idx, "clipped_fraction"])
            if frac >= r["max_clipped_fraction"]:
                decision = "invalid"
                causes.append("clipping_above_limit")
            notes.append(f"clipped {frac:.2%} (limit {r['max_clipped_fraction']:.0%})")
        if reasons & RATE_REASONS:
            audio, sr = sf.read(str(root / df.at[idx, "audio_path"]), dtype="float32", always_2d=True)
            mask = active_speech_mask(audio.mean(axis=1), sr, mcfg["active_frame_ms"], mcfg["active_hop_ms"],
                                      r["active_threshold_db"])
            active = np.flatnonzero(mask)
            span = (active[-1] - active[0]) / sr if active.size else 0.0
            rate = len(str(df.at[idx, "transcript"])) / max(span, EPS)
            df.at[idx, "active_span_sec"] = round(span, 3)
            df.at[idx, "active_chars_per_sec"] = round(rate, 3)
            if rate < r["min_active_chars_per_sec"]:
                decision = "invalid"
                causes.append("active_speaking_rate_too_slow")
            elif rate > r["max_active_chars_per_sec"]:
                decision = "invalid"
                causes.append("active_speaking_rate_too_fast")
            notes.append(f"active span {span:.1f}s, {rate:.1f} chars/s "
                         f"(range {r['min_active_chars_per_sec']}-{r['max_active_chars_per_sec']})")
        df.at[idx, "quality_status"] = decision
        df.at[idx, "exclusion_reason"] = ";".join(causes)
        df.at[idx, "auto_review"] = f"{decision}: " + "; ".join(notes)
    resolved = df["auto_review"] != ""
    log.info("Auto-review resolved %d files: %s", int(resolved.sum()),
             df.loc[resolved, "quality_status"].value_counts().to_dict())
    return df


def _start_audit(df: pd.DataFrame) -> None:
    """Every file starts valid without flags; the checks can only lower its status."""
    df["audit_flags"] = ""
    df["exclusion_reason"] = ""
    df["quality_status"] = "valid"
    df["manual_decision"] = ""


def _assign(df: pd.DataFrame, mask, status: str, flag: str) -> None:
    """Flag rows; 'invalid' overrides 'review', which overrides 'valid'. Flags accumulate."""
    rank = {"valid": 0, "review": 1, "invalid": 2}
    for idx in df.index[mask]:
        if rank[status] > rank[df.at[idx, "quality_status"]]:
            df.at[idx, "quality_status"] = status
        prev = df.at[idx, "audit_flags"]
        df.at[idx, "audit_flags"] = f"{prev};{flag}" if prev else flag


def _finalize_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `exclusion_reason` for files that are not valid (from their flags if no rule gave one)."""
    not_valid = df["quality_status"] != "valid"
    fallback = not_valid & (df["exclusion_reason"] == "")
    df.loc[fallback, "exclusion_reason"] = df.loc[fallback, "audit_flags"]
    df.loc[~not_valid, "exclusion_reason"] = ""
    return df


def _count_tokens(series: pd.Series) -> dict:
    return series.str.split(";").explode().value_counts().drop("", errors="ignore").to_dict()


def audit_speech(cfg: dict, summary_only: bool = False) -> pd.DataFrame:
    a = cfg["speech_audit"]
    root = project_path(cfg["paths"]["speech_root"])
    df = pd.read_csv(manifest_path(cfg, "speech_manifest_raw.csv"), keep_default_na=False)
    _start_audit(df)

    present = df.index[df["audio_path"].map(lambda p: (root / p).exists())]
    log.info("Inspecting %d / %d speech files", len(present), len(df))
    stats = _inspect_all([root / df.at[i, "audio_path"] for i in present], a["clipping_peak"], a["num_workers"])
    stats.index = present
    df = df.join(stats)

    durations = df["actual_duration_sec"].dropna()
    rms = df["rms_dbfs"].replace(-np.inf, np.nan).dropna()
    summary = {
        "n_rows": len(df), "n_inspected": int(len(present)),
        "duration_sec": durations.describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).round(3).to_dict(),
        "rms_dbfs": rms.describe(percentiles=[.001, .01, .05, .5]).round(2).to_dict(),
    }
    log.info("Duration summary (sec): %s", summary["duration_sec"])
    log.info("RMS summary (dBFS): %s", summary["rms_dbfs"])
    if summary_only:
        return df

    target_sr = cfg["audio"]["target_sample_rate"]
    readable = df["readable"].fillna(False).astype(bool)
    _assign(df, df["readable"].isna(), "invalid", "audio_not_found")
    _assign(df, df["readable"].notna() & ~readable, "invalid", "unreadable")
    _assign(df, readable & (df["num_samples"] == 0), "invalid", "empty_waveform")
    _assign(df, df["transcript"].astype(str).str.strip() == "", "invalid", "missing_transcript")
    _assign(df, readable & (df["peak"] == 0), "invalid", "all_zero_waveform")
    _assign(df, readable & (df["rms_dbfs"] < a["silence_rms_dbfs"]), "review", "near_silent")
    _assign(df, readable & (df["clipped_fraction"] >= a["clipping_fraction_review"]), "review",
            "possible_clipping")
    _assign(df, readable & (df["actual_sample_rate"] != target_sr), "review", "sample_rate_mismatch")
    meta_dur = pd.to_numeric(df["duration_sec"], errors="coerce")
    _assign(df, readable & ((meta_dur - df["actual_duration_sec"]).abs() > a["metadata_duration_tolerance_sec"]),
            "review", "duration_metadata_mismatch")
    if a.get("min_duration_sec") is not None:
        _assign(df, readable & (df["actual_duration_sec"] < a["min_duration_sec"]), "review", "too_short")
    if a.get("max_duration_sec") is not None:
        _assign(df, readable & (df["actual_duration_sec"] > a["max_duration_sec"]), "review", "too_long")
    else:
        log.warning("Duration thresholds are not set; inspect the summary and set them in config.yaml")
    df["chars_per_sec"] = (df["transcript"].astype(str).str.len() / df["actual_duration_sec"]).round(3)
    if a.get("min_chars_per_sec") is not None:
        _assign(df, readable & (df["chars_per_sec"] < a["min_chars_per_sec"]), "review", "speaking_rate_too_slow")
    if a.get("max_chars_per_sec") is not None:
        _assign(df, readable & (df["chars_per_sec"] > a["max_chars_per_sec"]), "review", "speaking_rate_too_fast")

    df = _auto_review_speech(df, cfg)
    df = _apply_manual_decisions(df, "speech", "utt_id", cfg)  # manual decisions override the rules
    df = _finalize_reasons(df)

    df["eligible_primary"] = (df["quality_status"] == "valid") & ~(
        df["is_synthetic"].astype(str).str.lower().eq("true") & bool(a["exclude_synthetic"]))

    out = manifest_path(cfg, "speech_manifest.csv")
    df.to_csv(out, index=False)
    status = ["official_split", "quality_status"]
    summary["status_counts"] = df.groupby(status).size().unstack(fill_value=0).to_dict()
    if "auto_review" in df:
        summary["auto_review"] = df[df["auto_review"] != ""].groupby(status).size().unstack(fill_value=0).to_dict()
    summary["eligible_primary_per_split"] = df[df["eligible_primary"]].groupby("official_split").size().to_dict()
    summary["audit_flags_raised"] = _count_tokens(df["audit_flags"])
    summary["exclusion_reasons"] = _count_tokens(df["exclusion_reason"])
    save_json(summary, manifest_path(cfg, "speech_audit_summary.json"))
    log.info("Wrote %s\n%s", out, df.groupby(status).size().to_string())
    return df


def _flag_duplicate_noise(df: pd.DataFrame, eval_pool: str) -> None:
    """Identical noise files. Same category: keep one copy, the evaluation-pool one first. Different
    categories: the label is unreliable, so every copy is excluded."""
    df["duplicate_of"] = ""
    for _, g in df.groupby("content_md5"):
        if len(g) < 2:
            continue
        if g["noise_category"].nunique() > 1:
            _assign(df, df.index.isin(g.index), "invalid", "duplicate_conflicting_category")
            for idx in g.index:
                df.at[idx, "duplicate_of"] = ";".join(g.loc[g.index != idx, "noise_id"])
            continue
        keep = g.assign(other=g["noise_pool"] != eval_pool).sort_values(["other", "noise_id"]).index[0]
        drop = g.index.drop(keep)
        _assign(df, df.index.isin(drop), "invalid", "duplicate_file")
        df.loc[drop, "duplicate_of"] = df.at[keep, "noise_id"]
    dup = df[df["duplicate_of"] != ""]
    if len(dup):
        log.info("Duplicate noise files:\n%s", dup[["noise_id", "noise_category", "duplicate_of"]].to_string(index=False))


def audit_noise(cfg: dict) -> pd.DataFrame:
    a = cfg["noise_audit"]
    root = project_path(cfg["paths"]["noise_root"])
    df = pd.read_csv(manifest_path(cfg, "noise_manifest_raw.csv"), keep_default_na=False)
    _start_audit(df)
    stats = _inspect_all([root / p for p in df["noise_path"]], 0.999, a["num_workers"])
    df = df.join(stats)
    df["content_md5"] = [hashlib.md5((root / p).read_bytes()).hexdigest() for p in df["noise_path"]]

    readable = df["readable"].astype(bool)
    _assign(df, ~readable, "invalid", "unreadable")
    _assign(df, readable & (df["num_samples"] == 0), "invalid", "empty_waveform")
    _assign(df, df["noise_category"].isin(a["speech_like_categories"]), "invalid", "speech_or_babble")
    if not a.get("include_music", False):
        _assign(df, df["noise_category"].str.lower().str.contains("music"), "invalid", "music_out_of_scope")
    _assign(df, df["noise_category"].isin(a["review_categories"]), "review", "may_contain_background_speech")
    _assign(df, readable & (df["rms_dbfs"] < a["silence_rms_dbfs"]), "review", "near_silent")
    _assign(df, readable & (df["actual_sample_rate"] != cfg["audio"]["target_sample_rate"]),
            "review", "sample_rate_mismatch")
    _flag_duplicate_noise(df, cfg["mixing"]["eval_noise_pool"])

    df = _apply_manual_decisions(df, "noise", "noise_id", cfg)
    df = _finalize_reasons(df)
    out = manifest_path(cfg, "noise_manifest.csv")
    df.to_csv(out, index=False)
    log.info("Wrote %s\n%s", out,
             df.groupby(["noise_pool", "noise_category", "quality_status"]).size().to_string())
    return df


def speech_in_noise_flags(texts: list[str], durations: list[float], threshold: float) -> list[dict]:
    """Letters per second Wav2Vec2 "heard" in each noise file; at or above `threshold` the file
    probably contains speech."""
    rows = []
    for text, dur in zip(texts, durations):
        chars = len(text.replace(" ", ""))
        rate = chars / dur if dur > 0 else 0.0
        rows.append({"transcribed_chars": chars, "chars_per_sec": round(rate, 3),
                     "speech_suspected": rate >= threshold, "transcript": text})
    return rows


def split_into_chunks(audio: np.ndarray, chunk_size: int, min_size: int) -> list[np.ndarray]:
    """Cut audio into pieces of `chunk_size` samples. A tail shorter than `min_size` joins the piece
    before it, because Wav2Vec2 cannot process a few milliseconds of audio."""
    pieces = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    if len(pieces) > 1 and len(pieces[-1]) < min_size:
        last = pieces.pop()
        pieces[-1] = np.concatenate([pieces[-1], last])
    return pieces


def check_noise_for_speech(cfg: dict) -> pd.DataFrame:
    """Check valid noise files for speech with Wav2Vec2 (plan 3.2, 5.2, 16). Suspected files are listed in
    noise_speech_check.csv; excluding one is a manual decision."""
    from src.model_loader import load_model
    from src.noise_mixing import load_noise
    from src.utils import resolve_device

    noise = pd.read_csv(manifest_path(cfg, "noise_manifest.csv"), keep_default_na=False)
    noise = noise[noise["quality_status"] == "valid"].reset_index(drop=True)
    model = load_model("wav2vec2", cfg, resolve_device(cfg["inference"]["device"]))
    sr = cfg["audio"]["target_sample_rate"]
    texts, durations = [], []
    for path in noise["noise_path"]:
        audio = load_noise(cfg, path)
        pieces = split_into_chunks(audio, chunk_size=20 * sr, min_size=sr)
        texts.append(" ".join(model.transcribe([piece])[0] for piece in pieces).strip())
        durations.append(audio.size / sr)
    flags = pd.DataFrame(speech_in_noise_flags(texts, durations, cfg["noise_audit"]["speech_check_chars_per_sec"]))
    out = pd.concat([noise[["noise_id", "noise_pool", "noise_category", "noise_path"]], flags], axis=1)
    out.to_csv(manifest_path(cfg, "noise_speech_check.csv"), index=False)
    suspected = out[out["speech_suspected"]]
    log.info("Speech check: %d/%d valid noise files suspected of containing speech", len(suspected), len(out))
    if len(suspected):
        log.info("\n%s", suspected[["noise_id", "chars_per_sec", "transcript"]].to_string(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--target", default="all", choices=["speech", "noise", "all", "noise-speech"],
                        help="noise-speech: screen valid noise files for speech with Wav2Vec2 (needs the model)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.target == "noise-speech":
        check_noise_for_speech(cfg)
        return
    if args.target in ("speech", "all"):
        audit_speech(cfg, summary_only=args.summary_only)
    if args.target in ("noise", "all") and not args.summary_only:
        audit_noise(cfg)


if __name__ == "__main__":
    main()
