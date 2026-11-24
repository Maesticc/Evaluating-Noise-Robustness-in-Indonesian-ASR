"""Train the CNN-BiLSTM-CTC model from scratch on the training split.

The validation split picks the best checkpoint and stops training early; the test split is not used.
The training state is saved to last.pt after every epoch, so a stopped run continues with --resume.

Usage:
    python -m src.train_custom_model [--max-epochs N] [--resume] [--max-hours H]
"""

from __future__ import annotations

import argparse
import math
import time

import jiwer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.custom_cnn_bilstm_ctc import CharVocab, CustomASR, pad_batch, save_checkpoint
from src.data_loader import eligible_speech
from src.noise_mixing import load_speech
from src.preprocessing import normalize_text
from src.utils import (ensure_dir, environment_info, get_logger, load_config, project_path, resolve_device,
                       save_json, set_seed)


class SpeechDataset(Dataset):
    def __init__(self, cfg: dict, rows: pd.DataFrame, vocab: CharVocab):
        self.cfg, self.rows, self.vocab = cfg, rows.reset_index(drop=True), vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        wave = load_speech(self.cfg, row["audio_path"])
        text = normalize_text(row["transcript"], self.cfg)
        return wave, torch.tensor(self.vocab.encode(text), dtype=torch.long), text


def collate(batch):
    waves, labels, texts = zip(*batch)
    return list(waves), list(labels), list(texts)


@torch.no_grad()
def evaluate(model: CustomASR, loader: DataLoader, device: str) -> dict:
    model.eval()
    refs, hyps = [], []
    for waves, _, texts in loader:
        wave, lengths = pad_batch(waves, device)
        log_probs, out_lengths = model(wave, lengths)
        hyps += model.greedy_decode(log_probs, out_lengths)
        refs += texts
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "example": (refs[0], hyps[0])}


def save_training_state(path, settings: dict, epoch: int, best: float, patience: int, history: list,
                        model, optim, sched, scaler) -> None:
    """Save everything needed to continue training. Written to a temporary file first, so a crash
    while saving never leaves a broken last.pt."""
    state = {"settings": settings, "epoch": epoch, "best": best, "patience": patience, "history": history,
             "model": model.state_dict(), "optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
             "scaler": scaler.state_dict()}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_training_state(path, settings: dict, model, optim, sched, scaler, device: str):
    """Restore a run saved by save_training_state. Returns (next epoch, best, patience, history)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: there is no interrupted run to resume")
    state = torch.load(path, map_location=device, weights_only=False)
    if state["settings"] != settings:
        raise ValueError("The saved run used other settings (epochs, number of training utterances or "
                         "vocabulary). Resume with exactly the same command as the interrupted run.")
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return state["epoch"] + 1, state["best"], state["patience"], state["history"]


def train(cfg: dict, max_epochs: int | None = None, max_train_utt: int | None = None, resume: bool = False,
          max_hours: float | None = None) -> None:
    """`max_hours`: stop after the last epoch that fits in this time (e.g. a Kaggle session)."""
    session_start = time.time()
    mcfg = cfg["models"]["custom_cnn_bilstm_ctc"]
    tcfg = mcfg["training"]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = resolve_device(cfg["inference"]["device"])
    out_dir = ensure_dir(project_path(mcfg["checkpoint"]).parent)
    log = get_logger("train_custom", project_path(cfg["paths"]["logs_dir"]) / "train_custom_model.log")

    train_df, val_df = eligible_speech(cfg, "train"), eligible_speech(cfg, "val")
    dur = pd.to_numeric(train_df["duration_sec"], errors="coerce")
    train_df = train_df[dur <= tcfg["max_train_duration_sec"]]
    if max_train_utt or tcfg.get("max_train_utterances"):
        n = max_train_utt or tcfg["max_train_utterances"]
        train_df = train_df.sample(n=min(n, len(train_df)), random_state=seed)
    if tcfg.get("max_val_utterances") and len(val_df) > tcfg["max_val_utterances"]:
        val_df = val_df.sample(n=tcfg["max_val_utterances"], random_state=seed)
    log.info("Train utterances: %d | val utterances: %d | device: %s", len(train_df), len(val_df), device)

    vocab = CharVocab.from_texts(normalize_text(t, cfg) for t in train_df["transcript"])
    log.info("Vocabulary (%d incl. blank): %s", len(vocab), "".join(vocab.itos[1:]))

    train_set = SpeechDataset(cfg, train_df, vocab)
    steps_per_epoch = math.ceil(len(train_set) / tcfg["batch_size"])
    val_loader = DataLoader(SpeechDataset(cfg, val_df, vocab), batch_size=tcfg["batch_size"], shuffle=False,
                            collate_fn=collate, num_workers=tcfg["num_workers"])

    model = CustomASR(vocab, mcfg["features"], mcfg["architecture"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %.2fM", n_params / 1e6)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    epochs = max_epochs or tcfg["max_epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=tcfg["learning_rate"], epochs=epochs,
                                                steps_per_epoch=max(1, steps_per_epoch), pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = bool(tcfg["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metric = tcfg["selection_metric"]
    last_path = out_dir / "last.pt"
    # A run can only be resumed with the same settings it was started with.
    settings = {"epochs": epochs, "n_train": len(train_df), "vocab": vocab.itos}
    start_epoch, best, patience, history = 1, math.inf, 0, []
    if resume:
        start_epoch, best, patience, history = load_training_state(last_path, settings, model, optim, sched,
                                                                   scaler, device)
        log.info("Resumed from %s: continuing at epoch %d (best val %s so far %.4f)",
                 last_path, start_epoch, metric, best)
    elif last_path.exists():
        log.warning("%s will be overwritten by this new run (use --resume to continue the old run)", last_path)

    stopped_by_time = False
    for epoch in range(start_epoch, epochs + 1):
        if patience >= tcfg["early_stopping_patience"]:
            log.info("Early stopping after %d epochs without improvement", patience)
            break
        if max_hours and epoch > start_epoch:
            # stop when one more epoch of average length would not fit
            spent = time.time() - session_start
            if spent + spent / (epoch - start_epoch) > max_hours * 3600:
                log.info("Time budget of %.1f h reached after epoch %d; continue with --resume", max_hours, epoch - 1)
                stopped_by_time = True
                break
        # shuffle order depends only on the epoch, so a resumed run sees the same order
        train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate,
                                  num_workers=tcfg["num_workers"],
                                  generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        t0, losses = time.time(), []
        for step, (waves, labels, _) in enumerate(train_loader, 1):
            wave, lengths = pad_batch(waves, device)
            targets = torch.cat(labels).to(device)
            target_lengths = torch.tensor([len(l) for l in labels], device=device)
            with torch.autocast(device_type=device, enabled=use_amp):
                log_probs, out_lengths = model(wave, lengths)
            loss = ctc(log_probs.float().transpose(0, 1), targets, out_lengths, target_lengths)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if step % 200 == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(train_loader), np.mean(losses[-200:]))

        train_sec = time.time() - t0
        val = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_wer": val["wer"],
                  "val_cer": val["cer"], "train_sec": round(train_sec, 1), "epoch_sec": round(time.time() - t0, 1)}
        history.append(record)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)  # saved every epoch
        log.info("epoch %d | loss %.4f | val WER %.4f | val CER %.4f | train %.0fs, total %.0fs | ref=%r hyp=%r",
                 epoch, record["train_loss"], val["wer"], val["cer"], record["train_sec"], record["epoch_sec"],
                 *val["example"])

        score = val[metric]
        if score < best:
            best, patience = score, 0
            save_checkpoint(out_dir / "best.pt", model, mcfg,
                            {"epoch": epoch, "val": {k: val[k] for k in ("wer", "cer")}, "seed": seed})
            log.info("New best val %s %.4f -> saved %s", metric, score, out_dir / "best.pt")
        else:
            patience += 1
        save_training_state(last_path, settings, epoch, best, patience, history, model, optim, sched, scaler)

    save_json({"best_val_" + metric: best, "epochs_run": len(history), "finished": not stopped_by_time,
               "n_train": len(train_df), "n_val": len(val_df), "n_params": n_params, "config": mcfg,
               "environment": environment_info()},
              out_dir / "training_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-utterances", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue the interrupted run saved in last.pt")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop cleanly before this many hours (e.g. 11 on Kaggle); continue later with --resume")
    args = parser.parse_args()
    train(load_config(args.config), args.max_epochs, args.max_train_utterances, args.resume, args.max_hours)


if __name__ == "__main__":
    main()
