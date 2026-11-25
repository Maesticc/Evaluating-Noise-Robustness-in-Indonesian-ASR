"""Run every model on the same clean and noisy utterances and save the transcripts and timing.

Timing covers the model only (no waveform generation, no warm-up batch). Predictions are saved
after every batch to a .partial.csv file, so a stopped run continues with --resume.

Usage:
    python -m src.inference [--models ...] [--pilot] [--resume] [--max-hours H]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import pandas as pd
import torch

from src.model_loader import MODEL_NAMES, load_model
from src.noise_mixing import generate_waveform
from src.utils import (ensure_dir, environment_info, get_logger, load_config, load_json, manifest_path,
                       project_path, resolve_device, save_json, set_seed)

ROW_KEY = ["utt_id", "condition", "realization"]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def remaining_rows(rows: pd.DataFrame, done: pd.DataFrame) -> pd.DataFrame:
    """Rows of the conditions manifest that have no prediction in `done` yet (order kept)."""
    if done.empty:
        return rows
    done_keys = set(done[ROW_KEY].astype(str).itertuples(index=False, name=None))
    todo = [k not in done_keys for k in rows[ROW_KEY].astype(str).itertuples(index=False, name=None)]
    return rows[todo].reset_index(drop=True)


def run_model(cfg: dict, name: str, conditions: pd.DataFrame, tag: str, device: str, resume: bool = False,
              deadline: float | None = None) -> bool:
    """Run one model on every row. Returns False if it stopped at `deadline` (time.time()) first."""
    log = get_logger("inference", project_path(cfg["paths"]["logs_dir"]) / f"inference_{tag}.log")
    out_dir = ensure_dir(project_path(cfg["paths"]["predictions_dir"]))
    sr = cfg["audio"]["target_sample_rate"]
    batch_size = cfg["models"][name]["batch_size"]
    pred_path = out_dir / f"predictions_{tag}_{name}.csv"
    partial_path = out_dir / f"predictions_{tag}_{name}.partial.csv"
    setup_path = out_dir / f"predictions_{tag}_{name}.partial.json"

    t_load = time.time()
    model = load_model(name, cfg, device)
    load_sec = time.time() - t_load
    if model.sample_rate != sr:
        raise ValueError(f"{name} expects {model.sample_rate} Hz but the pipeline produces {sr} Hz")
    log.info("Loaded %s in %.1fs: %s", name, load_sec, model.describe())
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # resume only on the same model, GPU and batch size, so the timing stays comparable
    env = environment_info()
    setup = json.loads(json.dumps({"model": model.describe(), "device": device, "gpu": env.get("gpu"),
                                   "batch_size": batch_size, "n_rows": len(conditions)}, default=str))
    done = pd.DataFrame(columns=ROW_KEY)
    if partial_path.exists() and resume:
        saved = load_json(setup_path)
        if saved != setup:
            raise ValueError(f"Cannot resume {partial_path}: it was started with another model, GPU, batch size "
                             f"or number of rows.\nthen: {saved}\nnow:  {setup}")
        done = pd.read_csv(partial_path, keep_default_na=False)
        log.info("Resuming %s: %d of %d rows already done", name, len(done), len(conditions))
    else:
        if partial_path.exists():
            log.warning("Removing unfinished %s (use --resume to continue it instead)", partial_path)
            partial_path.unlink()
        save_json(setup, setup_path)

    # sorted by duration to reduce padding (the order does not change the results)
    rows = conditions.sort_values(["condition", "duration_sec", "utt_id"]).reset_index(drop=True)
    rows = remaining_rows(rows, done)
    batches = [rows.iloc[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    for b in batches[: cfg["inference"]["warmup_batches"]]:
        model.transcribe([generate_waveform(cfg, r) for _, r in b.iterrows()])

    errors = []
    for bi, b in enumerate(batches):
        if deadline and time.time() > deadline:
            log.info("Time budget reached: %s stopped after %d of %d batches; continue with --resume",
                     name, bi, len(batches))
            del model
            return False
        waves = [generate_waveform(cfg, r) for _, r in b.iterrows()]
        failed = False
        try:
            _sync(device)
            t0 = time.perf_counter()
            hyps = model.transcribe(waves)
            _sync(device)
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # recorded, not hidden
            errors.append({"batch": bi, "utt_ids": list(b["utt_id"]), "error": repr(exc),
                           "traceback": traceback.format_exc()})
            log.error("Batch %d failed: %r", bi, exc)
            hyps, elapsed, failed = [None] * len(b), float("nan"), True
        audio_sec = sum(len(w) for w in waves) / sr
        records = []
        for (_, r), w, h in zip(b.iterrows(), waves, hyps):
            records.append({
                "model": name, "utt_id": r["utt_id"], "condition": r["condition"],
                "realization": r["realization"], "target_snr_db": r["target_snr_db"],
                "noise_id": r["noise_id"], "noise_category": r["noise_category"],
                "speaker_id": r["speaker_id"], "category": r["category"],
                "reference": r["transcript"], "hypothesis": h, "audio_sec": len(w) / sr,
                # batch time apportioned by audio duration
                "inference_sec": elapsed * (len(w) / sr) / audio_sec if audio_sec else float("nan"),
                "failed": failed,
            })
        pd.DataFrame(records).to_csv(partial_path, mode="a", header=not partial_path.exists(), index=False)
        if (bi + 1) % 50 == 0:
            log.info("%s: %d/%d batches (%d rows were done before this session)", name, bi + 1, len(batches),
                     len(done))

    pred = pd.read_csv(partial_path, keep_default_na=False)
    pred.to_csv(pred_path, index=False)
    meta = {
        "model": model.describe(), "tag": tag, "device": device, "batch_size": batch_size,
        "n_rows": len(pred), "n_failed": int(pred["failed"].astype(str).str.lower().eq("true").sum()),
        "rows_from_earlier_sessions": len(done),
        "warmup_batches": cfg["inference"]["warmup_batches"], "model_load_sec": round(load_sec, 2),
        "timing_method": "time.perf_counter around model.transcribe with CUDA synchronize; "
                         "waveform generation and warm-up excluded",
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "environment": env, "errors": errors, "config_path": cfg["_config_path"],
    }
    save_json(meta, out_dir / f"run_{tag}_{name}.json")
    partial_path.unlink()
    setup_path.unlink()
    log.info("Wrote %s (%d rows, %d failed)", pred_path, len(pred), meta["n_failed"])

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an unfinished run (skips rows already done)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop cleanly after this many hours (e.g. 11 on Kaggle); continue later with --resume")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: first N utterances")
    args = parser.parse_args()
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["inference"]["device"])
    tag = f"{args.split}{'_pilot' if args.pilot else ''}"
    conditions = pd.read_csv(manifest_path(cfg, f"conditions_{tag}.csv"))
    conditions["noise_id"] = conditions["noise_id"].fillna("")
    conditions["noise_category"] = conditions["noise_category"].fillna("")
    if args.limit:
        keep = sorted(conditions["utt_id"].unique())[: args.limit]
        conditions = conditions[conditions["utt_id"].isin(keep)]
        tag += f"_limit{args.limit}"
    for name in args.models:
        if not run_model(cfg, name, conditions, tag, device, resume=args.resume, deadline=deadline):
            print(f"Stopped at the time budget during {name}. Run the same command with --resume to continue.")
            break


if __name__ == "__main__":
    main()
