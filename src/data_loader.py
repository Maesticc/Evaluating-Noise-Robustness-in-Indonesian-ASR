"""Dataset download, metadata loading, audio loading and raw manifest creation.

Usage:
    python -m src.data_loader download-speech [--categories Declarative ...] [--no-audio]
    python -m src.data_loader download-noise
    python -m src.data_loader build-manifests
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.utils import ensure_dir, get_logger, load_config, manifest_path, project_path, save_json

log = get_logger("data_loader")

SPEECH_MANIFEST_COLUMNS = [
    "utt_id", "audio_path", "transcript", "speaker_id", "speaker_type", "speaker_gender",
    "official_split", "category", "sentence_id", "duration_sec", "sample_rate", "num_channels",
    "is_synthetic", "quality_status", "exclusion_reason",
]
NOISE_MANIFEST_COLUMNS = [
    "noise_id", "noise_path", "noise_pool", "noise_category", "duration_sec", "sample_rate",
    "num_channels", "quality_status", "exclusion_reason", "provenance_note",
]
SPLIT_ALIASES = {"dev": "val", "valid": "val", "validation": "val"}
NOISE_PROVENANCE = (
    "microsoft/MS-SNSD; noise sourced from Freesound (CC0) and DEMAND (CC BY-SA 3.0) "
    "per the MS-SNSD README; per-file source not specified upstream"
)


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as float32 in [-1, 1], resampled only when needed. The loudness is not normalized,
    because that would change the speech/noise relationship before mixing."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    audio = audio.T
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    audio = audio[0] if mono else audio
    if target_sr is not None and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import torch
    import torchaudio.functional as AF

    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return AF.resample(tensor, orig_sr, target_sr).numpy()


def download_speech(cfg: dict, categories: list[str] | None = None, audio: bool = True) -> None:
    """Download metadata (and optionally category tar shards) from Hugging Face, then extract."""
    from huggingface_hub import HfApi, snapshot_download

    ds = cfg["datasets"]["speech"]
    root = ensure_dir(project_path(cfg["paths"]["speech_root"]))
    patterns = ["metadata/*", "splits/*", "checksums/*", "docs/*", "README.md", "LICENSE",
                "data/transcripts/*", "data/audio_shards/audio_shards_manifest.csv"]
    if audio:
        if categories:
            patterns += [f"data/audio_shards/by_category/{c}.tar" for c in categories]
        else:
            patterns += ["data/audio_shards/by_category/*.tar"]

    log.info("Downloading %s@%s into %s", ds["hf_repo_id"], ds["revision"], root)
    snapshot_download(repo_id=ds["hf_repo_id"], repo_type="dataset", revision=ds["revision"],
                      local_dir=str(root), allow_patterns=patterns)
    commit = HfApi().dataset_info(ds["hf_repo_id"], revision=ds["revision"]).sha
    save_json({"repo_id": ds["hf_repo_id"], "requested_revision": ds["revision"],
               "resolved_commit": commit, "categories": categories or "all"},
              root / "DOWNLOAD_INFO.json")

    if audio:
        for tar_path in sorted((root / "data" / "audio_shards" / "by_category").glob("*.tar")):
            marker = tar_path.with_suffix(".extracted")
            if marker.exists():
                continue
            log.info("Extracting %s", tar_path.name)
            with tarfile.open(tar_path) as tf:
                tf.extractall(root, filter="data")
            marker.touch()
    log.info("Speech dataset ready (resolved commit %s)", commit)


def download_noise(cfg: dict) -> None:
    """Sparse-clone only the MS-SNSD noise folders (the clean-speech folders are not used)."""
    root = project_path(cfg["paths"]["noise_root"])
    pools = cfg["datasets"]["noise"]["pools"]
    if not (root / ".git").exists():
        ensure_dir(root.parent)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        cfg["datasets"]["noise"]["git_url"], str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", *pools], check=True)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    save_json({"git_url": cfg["datasets"]["noise"]["git_url"], "commit": commit, "pools": pools},
              root / "DOWNLOAD_INFO.json")
    log.info("MS-SNSD noise folders ready at %s (commit %s)", root, commit)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_speech_metadata(cfg: dict) -> pd.DataFrame:
    """Read the public metadata CSV, the dataset's row-level source of truth."""
    meta_path = project_path(cfg["paths"]["speech_metadata"])
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run: python -m src.data_loader download-speech")
    df = pd.read_csv(meta_path, keep_default_na=False, na_values=[""])
    df["split"] = df["split"].astype(str).str.strip().str.lower().replace(SPLIT_ALIASES)
    df["is_synthetic"] = _to_bool(df["is_synthetic"])
    return df


def check_split_assignment(cfg: dict, meta: pd.DataFrame) -> None:
    """Verify speaker-disjointness and agreement with the official speaker split file."""
    per_speaker = meta.groupby("speaker_id")["split"].nunique()
    leaking = per_speaker[per_speaker > 1]
    if len(leaking):
        raise ValueError(f"Speakers present in more than one split: {list(leaking.index)}")

    assign_path = project_path(cfg["paths"]["speech_split_assignment"])
    if not assign_path.exists():
        log.warning("Split assignment file not found (%s); skipped cross-check", assign_path)
        return
    assign = pd.read_csv(assign_path)
    spk_col = next((c for c in assign.columns if "speaker" in c.lower()), None)
    split_col = next((c for c in assign.columns if c.lower() == "split"), None)
    if not spk_col or not split_col:
        log.warning("Could not identify speaker/split columns in %s: %s", assign_path, list(assign.columns))
        return
    official = dict(zip(assign[spk_col].astype(str),
                        assign[split_col].astype(str).str.lower().replace(SPLIT_ALIASES)))
    observed = meta.groupby("speaker_id")["split"].first()
    mismatched = [s for s, sp in observed.items() if s in official and official[s] != sp]
    if mismatched:
        raise ValueError(f"Metadata split disagrees with split assignment for speakers: {mismatched}")
    log.info("Split check passed: %d speakers, speaker-disjoint, consistent with %s",
             len(observed), assign_path.name)


def build_speech_manifest(cfg: dict) -> pd.DataFrame:
    meta = load_speech_metadata(cfg)
    check_split_assignment(cfg, meta)
    root = project_path(cfg["paths"]["speech_root"])

    out = pd.DataFrame({
        "utt_id": meta["audio_path"].map(lambda p: re.sub(r"[^\w]+", "_", Path(p).with_suffix("").as_posix())
                                         .split("Dataset_Balanced19_")[-1]),
        "audio_path": meta["audio_path"],
        "transcript": meta["transcript"],
        "speaker_id": meta["speaker_id"],
        "speaker_type": meta.get("speaker_type"),
        "speaker_gender": meta.get("speaker_gender"),
        "official_split": meta["split"],
        "category": meta.get("category"),
        "sentence_id": meta.get("sentence_id"),
        "duration_sec": meta.get("duration_sec"),
        "sample_rate": meta.get("sample_rate"),
        "num_channels": meta.get("num_channels"),
        "is_synthetic": meta["is_synthetic"],
        "quality_status": "pending",
        "exclusion_reason": "",
    })
    if out["utt_id"].duplicated().any():
        raise ValueError("utt_id is not unique; check audio_path values")
    missing = ~out["audio_path"].map(lambda p: (root / p).exists())
    log.info("Speech metadata rows: %d | audio present locally: %d | missing: %d",
             len(out), (~missing).sum(), missing.sum())
    out.loc[missing, "exclusion_reason"] = "audio_not_downloaded"

    path = manifest_path(cfg, "speech_manifest_raw.csv")
    out[SPEECH_MANIFEST_COLUMNS].to_csv(path, index=False)
    log.info("Wrote %s", path)
    log.info("Rows per split:\n%s", out.groupby(["official_split", "is_synthetic"]).size().to_string())
    return out


def eligible_speech(cfg: dict, split: str) -> pd.DataFrame:
    """Utterances of one official split that passed the audit, sorted by utt_id."""
    df = pd.read_csv(manifest_path(cfg, "speech_manifest.csv"), keep_default_na=False)
    df = df[(df["official_split"] == split) & _to_bool(df["eligible_primary"])]
    return df.sort_values("utt_id").reset_index(drop=True)


def noise_category(filename: str, aliases: dict[str, str]) -> str:
    stem = Path(filename).stem
    cat = re.sub(r"_?\d+$", "", stem)
    return aliases.get(cat, cat)


def build_noise_manifest(cfg: dict) -> pd.DataFrame:
    root = project_path(cfg["paths"]["noise_root"])
    aliases = cfg["noise_audit"].get("category_aliases", {}) or {}
    rows = []
    for pool in cfg["datasets"]["noise"]["pools"]:
        pool_dir = root / pool
        if not pool_dir.exists():
            raise FileNotFoundError(f"{pool_dir} not found. Run: python -m src.data_loader download-noise")
        for wav in sorted(pool_dir.glob("*.wav")):
            info = sf.info(str(wav))
            rows.append({
                "noise_id": f"{pool}/{wav.stem}",
                "noise_path": wav.relative_to(root).as_posix(),
                "noise_pool": pool,
                "noise_category": noise_category(wav.name, aliases),
                "duration_sec": round(info.frames / info.samplerate, 4),
                "sample_rate": info.samplerate,
                "num_channels": info.channels,
                "quality_status": "pending",
                "exclusion_reason": "",
                "provenance_note": NOISE_PROVENANCE,
            })
    out = pd.DataFrame(rows, columns=NOISE_MANIFEST_COLUMNS)
    path = manifest_path(cfg, "noise_manifest_raw.csv")
    out.to_csv(path, index=False)
    log.info("Wrote %s (%d files)", path, len(out))
    log.info("Files per pool/category:\n%s", out.groupby(["noise_pool", "noise_category"]).size().to_string())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download-speech")
    p.add_argument("--categories", nargs="*", default=None, help="Category shards to download (default: all)")
    p.add_argument("--no-audio", action="store_true", help="Download metadata only")
    sub.add_parser("download-noise")
    sub.add_parser("build-manifests")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "download-speech":
        download_speech(cfg, args.categories, audio=not args.no_audio)
    elif args.cmd == "download-noise":
        download_noise(cfg)
    elif args.cmd == "build-manifests":
        build_speech_manifest(cfg)
        build_noise_manifest(cfg)


if __name__ == "__main__":
    main()
