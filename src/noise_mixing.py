"""Mix speech with noise at a target SNR: SNR (dB) = 10 * log10(P_speech / P_noise).

Speech power is measured over active-speech frames only, so pauses do not change the SNR.
The conditions manifest stores the noise file, offset and gain of every mixture, so each
waveform can be regenerated exactly and every model gets the same input.

Usage:
    python -m src.noise_mixing build-pairs [--split test] [--pilot]
    python -m src.noise_mixing export-pilot     # pilot mixtures as WAV files, for listening
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import soundfile as sf

from src.data_loader import eligible_speech, load_audio
from src.utils import ensure_dir, get_logger, load_config, manifest_path, project_path, stable_seed

log = get_logger("noise_mixing")
EPS = 1e-12


def active_speech_mask(audio: np.ndarray, sr: int, frame_ms: float, hop_ms: float,
                       threshold_db: float) -> np.ndarray:
    """Mark which samples belong to active speech (True) and which to silence (False).

    1. Cut the audio into short frames (25 ms, a new frame every 10 ms).
    2. Measure the energy of each frame in dB.
    3. A frame is speech if it is at most `threshold_db` quieter than the loudest frame.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if len(audio) < frame_len:
        return np.ones(len(audio), dtype=bool)

    frame_starts = range(0, len(audio) - frame_len + 1, hop_len)
    energies_db = []
    for start in frame_starts:
        frame = audio[start:start + frame_len].astype(np.float64)
        energies_db.append(10 * np.log10(np.mean(frame ** 2) + EPS))

    loudest_db = max(energies_db)
    mask = np.zeros(len(audio), dtype=bool)
    for start, energy_db in zip(frame_starts, energies_db):
        if energy_db >= loudest_db - threshold_db:
            mask[start:start + frame_len] = True
    return mask


def power(audio: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = audio if mask is None else audio[mask]
    return float(np.mean(x.astype(np.float64) ** 2)) if x.size else 0.0


def snr_db(p_signal: float, p_noise: float) -> float:
    return float(10 * np.log10((p_signal + EPS) / (p_noise + EPS)))


def select_noise_segment(noise: np.ndarray, length: int, offset: int) -> np.ndarray:
    """Crop (or tile, if the noise is shorter than the speech) starting at `offset`."""
    if noise.size >= length:
        return noise[offset: offset + length]
    reps = int(np.ceil((offset + length) / noise.size))
    return np.tile(noise, reps)[offset: offset + length]


def mix_at_snr(speech: np.ndarray, noise_seg: np.ndarray, target_snr: float, mcfg: dict,
               sr: int, gain: float | None = None) -> tuple[np.ndarray, dict]:
    """Scale the noise to the target SNR, add it to the speech and report the achieved SNR.

    `gain` scales speech and noise alike (the SNR stays the same) to prevent clipping.
    None computes it from this mixture alone.
    """
    if mcfg["power_method"] == "active_speech":
        mask = active_speech_mask(speech, sr, mcfg["active_frame_ms"], mcfg["active_hop_ms"],
                                  mcfg["active_threshold_db"])
    else:
        mask = None
    p_speech = power(speech, mask)
    p_noise = power(noise_seg)
    if p_noise <= EPS:
        raise ValueError("Noise segment has zero power")
    noise_scale = np.sqrt(p_speech / (p_noise * 10 ** (target_snr / 10)))
    speech_c = speech.astype(np.float64)
    noise_c = noise_seg.astype(np.float64) * noise_scale
    mixture = speech_c + noise_c

    peak = float(np.max(np.abs(mixture)))
    clipped = peak > mcfg["clip_peak"]
    if gain is None:
        gain = mcfg["clip_peak"] / peak if (clipped and mcfg["on_clipping"] == "scale_both") else 1.0
    # Same gain on speech and noise: the SNR is unchanged, only the level drops.
    speech_c, noise_c, mixture = speech_c * gain, noise_c * gain, mixture * gain

    achieved = snr_db(power(speech_c, mask), power(noise_c))
    achieved_full = snr_db(power(speech_c), power(noise_c))
    info = {
        "target_snr_db": target_snr,
        "achieved_snr_db": round(achieved, 4),
        "achieved_snr_full_db": round(achieved_full, 4),
        "noise_scale": float(noise_scale),
        "output_gain": round(gain, 6),
        "peak_before_gain": round(peak, 5),
        "clipping_detected": bool(clipped),
        "residual_clipping": bool(np.max(np.abs(mixture)) > 1.0),
        "active_speech_fraction": round(float(mask.mean()), 4) if mask is not None else 1.0,
    }
    return mixture.astype(np.float32), info


@lru_cache(maxsize=256)
def _load_noise_cached(abs_path: str, sr: int) -> np.ndarray:
    return load_audio(abs_path, sr)[0]


def load_speech(cfg: dict, audio_path: str) -> np.ndarray:
    return load_audio(project_path(cfg["paths"]["speech_root"]) / audio_path, cfg["audio"]["target_sample_rate"])[0]


def load_noise(cfg: dict, noise_path: str) -> np.ndarray:
    return _load_noise_cached(str(project_path(cfg["paths"]["noise_root"]) / noise_path),
                              cfg["audio"]["target_sample_rate"])


def condition_name(snr: float | None) -> str:
    return "clean" if snr is None or (isinstance(snr, float) and np.isnan(snr)) else f"snr_{int(snr)}dB"


def generate_waveform(cfg: dict, row: pd.Series) -> np.ndarray:
    """Regenerate the exact waveform of one row of the conditions manifest."""
    speech = load_speech(cfg, row["audio_path"])
    gain = float(row["output_gain"])
    if row["condition"] == "clean":
        return (speech.astype(np.float64) * gain).astype(np.float32)
    noise = load_noise(cfg, row["noise_path"])
    seg = select_noise_segment(noise, speech.size, int(row["noise_offset"]))
    mixture, _ = mix_at_snr(speech, seg, float(row["target_snr_db"]), cfg["mixing"],
                            cfg["audio"]["target_sample_rate"], gain=gain)
    return mixture


def utterance_gain(speech: np.ndarray, segments: list[np.ndarray], m: dict, sr: int) -> float:
    """One gain for all conditions of an utterance (clean included), set by its loudest mixture,
    so the speech level is the same in every condition and only the noise level differs."""
    peaks = [float(np.max(np.abs(speech)))]
    for seg in segments:
        for snr in m["snr_db"]:
            peaks.append(mix_at_snr(speech, seg, float(snr), m, sr, gain=1.0)[1]["peak_before_gain"])
    return min(1.0, m["clip_peak"] / max(peaks))


def eligible_noise(cfg: dict, pool: str) -> pd.DataFrame:
    noise = pd.read_csv(manifest_path(cfg, "noise_manifest.csv"), keep_default_na=False)
    noise = noise[(noise["noise_pool"] == pool) & (noise["quality_status"] == "valid")]
    if noise.empty:
        raise ValueError(f"No valid noise files in pool '{pool}'. Run the noise audit / review first.")
    return noise.sort_values("noise_id").reset_index(drop=True)


def pairing_noise(cfg: dict, split: str) -> pd.DataFrame:
    """Noise files for mixing utterances of `split`. Only the test split uses the evaluation pool;
    other splits use the other pool (same categories), so test noise never influences a decision."""
    m = cfg["mixing"]
    noise = eligible_noise(cfg, m["eval_noise_pool"])
    if split == "test":
        return noise
    other = eligible_noise(cfg, m["augmentation_noise_pool"])
    return other[other["noise_category"].isin(set(noise["noise_category"]))].reset_index(drop=True)


def select_utterances(cfg: dict, split: str, pilot: bool) -> pd.DataFrame:
    speech = eligible_speech(cfg, split)
    if pilot:  # fixed-seed random sample
        n = min(cfg["pilot"]["n_utterances"], len(speech))
        speech = speech.sample(n=n, random_state=cfg["project"]["seed"]).sort_values("utt_id")
    return speech.reset_index(drop=True)


def build_pairs(cfg: dict, split: str = "test", pilot: bool = False) -> pd.DataFrame:
    """Pair every utterance with one noise segment (same file and offset for every SNR) and write
    one row per condition to conditions_<split>.csv."""
    m = cfg["mixing"]
    sr = cfg["audio"]["target_sample_rate"]
    seed = cfg["project"]["seed"]
    speech = select_utterances(cfg, split, pilot)
    noise = pairing_noise(cfg, split)
    categories = sorted(noise["noise_category"].unique())
    log.info("%d utterances (%s%s) x %d noise categories %s", len(speech), split,
             ", pilot" if pilot else "", len(categories), categories)

    order = np.random.default_rng(seed).permutation(len(speech))
    rows = []
    for rank, i in enumerate(order):
        utt = speech.iloc[i]
        speech_audio = load_speech(cfg, utt["audio_path"])
        base = {"utt_id": utt["utt_id"], "audio_path": utt["audio_path"], "transcript": utt["transcript"],
                "speaker_id": utt["speaker_id"], "official_split": utt["official_split"],
                "category": utt["category"], "duration_sec": round(speech_audio.size / sr, 4)}
        pairings = []
        for r in range(m["realizations_per_utterance"]):
            pair_seed = stable_seed(seed, utt["utt_id"], r)
            rng = np.random.default_rng(pair_seed)
            if m["noise_assignment"] == "round_robin_category":
                cat = categories[(rank * m["realizations_per_utterance"] + r) % len(categories)]
                pool = noise[noise["noise_category"] == cat]
            else:
                pool = noise
            nrow = pool.iloc[int(rng.integers(len(pool)))]
            noise_audio = load_noise(cfg, nrow["noise_path"])
            offset = int(rng.integers(max(1, noise_audio.size - speech_audio.size + 1))) \
                if noise_audio.size >= speech_audio.size else int(rng.integers(noise_audio.size))
            pairings.append((r, pair_seed, nrow, offset,
                             select_noise_segment(noise_audio, speech_audio.size, offset)))

        if m["on_clipping"] != "scale_both":
            gain = 1.0
        elif m.get("gain_scope", "utterance") == "utterance":
            gain = utterance_gain(speech_audio, [p[4] for p in pairings], m, sr)
        else:
            gain = None  # derived per condition
        clean_peak = float(np.max(np.abs(speech_audio)))
        clean_gain = 1.0 if gain is None else gain
        rows.append({**base, "condition": "clean", "realization": 0, "noise_id": "", "noise_path": "",
                     "noise_category": "", "noise_offset": "", "pair_seed": "", "target_snr_db": np.nan,
                     "output_gain": round(clean_gain, 6), "peak_before_gain": round(clean_peak, 5),
                     "clipping_detected": clean_peak > m["clip_peak"],
                     "residual_clipping": clean_peak * clean_gain > 1.0})
        for r, pair_seed, nrow, offset, seg in pairings:
            for snr in m["snr_db"]:
                _, info = mix_at_snr(speech_audio, seg, float(snr), m, sr, gain=gain)
                rows.append({**base, "condition": condition_name(float(snr)), "realization": r,
                             "noise_id": nrow["noise_id"], "noise_path": nrow["noise_path"],
                             "noise_category": nrow["noise_category"], "noise_offset": offset,
                             "pair_seed": pair_seed, **info})

    df = pd.DataFrame(rows)
    noisy = df[df["condition"] != "clean"]
    err = (noisy["achieved_snr_db"] - noisy["target_snr_db"]).abs()
    bad = noisy[err > m["snr_tolerance_db"]]
    log.info("Achieved-SNR max error %.4f dB | outside tolerance %d | clipping handled %d | residual %d",
             err.max(), len(bad), int(noisy["clipping_detected"].sum()), int(noisy["residual_clipping"].sum()))
    if len(bad):
        log.warning("Rows outside SNR tolerance:\n%s", bad[["utt_id", "condition", "achieved_snr_db"]].head(20))
    gains = df.drop_duplicates("utt_id")["output_gain"]
    log.info("Output gain (%s scope): %d/%d utterances attenuated, min %.3f (%.1f dB)",
             m.get("gain_scope", "utterance"), int((gains < 1).sum()), len(gains), gains.min(),
             20 * np.log10(gains.min()))

    name = f"conditions_{split}{'_pilot' if pilot else ''}.csv"
    out = manifest_path(cfg, name)
    df.to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(df))
    log.info("Rows per condition/noise category:\n%s",
             df.groupby(["condition", "noise_category"]).size().to_string())
    return df


def export_pilot(cfg: dict, split: str = "test") -> None:
    """Write pilot waveforms so they can be listened to and inspected."""
    df = pd.read_csv(manifest_path(cfg, f"conditions_{split}_pilot.csv"), keep_default_na=False)
    out_dir = ensure_dir(project_path(cfg["paths"]["pilot_cache_dir"]))
    sr = cfg["audio"]["target_sample_rate"]
    for _, row in df.iterrows():
        row = row.copy()
        row["target_snr_db"] = float(row["target_snr_db"]) if row["target_snr_db"] != "" else np.nan
        wav = generate_waveform(cfg, row)
        sf.write(out_dir / f"{row['utt_id']}__{row['condition']}__r{row['realization']}.wav", wav, sr,
                 subtype="PCM_16")
    log.info("Wrote %d pilot WAVs to %s", len(df), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-pairs")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--pilot", action="store_true")
    e = sub.add_parser("export-pilot")
    e.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "build-pairs":
        build_pairs(cfg, args.split, args.pilot)
    else:
        export_pilot(cfg, args.split)


if __name__ == "__main__":
    main()
