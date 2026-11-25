"""Compute WER, CER, WER degradation and RTFx; produce result tables and plots.

WER  = (S + D + I) / N_ref_words * 100      (corpus-level: errors summed over utterances)
CER  = (S + D + I) / N_ref_chars * 100      (spaces count as characters)
dWER = WER_noisy - WER_clean                 (percentage points)
rel  = (WER_noisy - WER_clean) / WER_clean * 100
RTFx = audio duration / inference time       (higher = faster than real time)

The same normalization (src.preprocessing.normalize_text) is applied to references and
hypotheses of every model. Confidence intervals: utterance-level bootstrap.

Usage:
    python -m src.evaluate [--pilot] [--split test] [--models custom_cnn_bilstm_ctc wav2vec2 whisper]
"""

from __future__ import annotations

import argparse
import re

import jiwer
import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from src.model_loader import MODEL_NAMES
from src.preprocessing import normalize_text
from src.utils import ensure_dir, get_logger, load_config, load_json, manifest_path, project_path, save_json

log = get_logger("evaluate")
CONDITION_ORDER = ["clean", "snr_20dB", "snr_10dB", "snr_5dB", "snr_0dB"]


def error_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(word_errors, ref_words, char_errors, ref_chars) for one utterance."""
    hyp = hyp or ""
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    w_err = w.substitutions + w.deletions + w.insertions
    c_err = c.substitutions + c.deletions + c.insertions
    return w_err, w.hits + w.substitutions + w.deletions, c_err, c.hits + c.substitutions + c.deletions


def score_predictions(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pred = pred.copy()
    pred["ref_norm"] = pred["reference"].map(lambda t: normalize_text(t, cfg))
    pred["hyp_norm"] = pred["hypothesis"].map(lambda t: normalize_text(t, cfg))
    empty = pred["ref_norm"] == ""
    if empty.any():
        log.warning("Dropping %d rows with empty normalized reference", int(empty.sum()))
        pred = pred[~empty]
    counts = [error_counts(r, h) for r, h in zip(pred["ref_norm"], pred["hyp_norm"])]
    pred[["word_errors", "ref_words", "char_errors", "ref_chars"]] = np.array(counts)
    return pred


def bootstrap_ci(err: np.ndarray, n: np.ndarray, resamples: int, level: float, seed: int) -> tuple[float, float]:
    """Bootstrap confidence interval of an error rate. err[i] = errors in utterance i, n[i] = its
    reference words (or characters). Each resample draws the utterances again with replacement."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(err), size=(resamples, len(err)))
    rates = []
    for chosen in picks:
        rates.append(err[chosen].sum() / n[chosen].sum() * 100)
    lower_pct = (1 - level) / 2 * 100   # e.g. 2.5 for a 95 % interval
    upper_pct = (1 + level) / 2 * 100   # e.g. 97.5
    return float(np.percentile(rates, lower_pct)), float(np.percentile(rates, upper_pct))


def summarize(scored: pd.DataFrame, keys: list[str], cfg: dict, with_ci: bool = True) -> pd.DataFrame:
    ecfg = cfg["evaluation"]
    rows = []
    for k, g in scored.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row.update({
            "n_utterances": g["utt_id"].nunique(), "n_rows": len(g),
            "wer": g["word_errors"].sum() / g["ref_words"].sum() * 100,
            "cer": g["char_errors"].sum() / g["ref_chars"].sum() * 100,
            "utt_wer_std": float((g["word_errors"] / g["ref_words"] * 100).std()),
            "audio_sec": g["audio_sec"].sum(), "inference_sec": g["inference_sec"].sum(),
        })
        row["rtfx"] = row["audio_sec"] / row["inference_sec"] if row["inference_sec"] > 0 else np.nan
        if with_ci and len(g) > 1:
            seed = cfg["project"]["seed"]
            row["wer_ci_low"], row["wer_ci_high"] = bootstrap_ci(
                g["word_errors"].values, g["ref_words"].values, ecfg["bootstrap_resamples"], ecfg["confidence_level"], seed)
            row["cer_ci_low"], row["cer_ci_high"] = bootstrap_ci(
                g["char_errors"].values, g["ref_chars"].values, ecfg["bootstrap_resamples"], ecfg["confidence_level"], seed)
        rows.append(row)
    return pd.DataFrame(rows)


def add_degradation(table: pd.DataFrame, clean: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Add the clean WER/CER of the same utterances (matched on `on`) and the degradation.
    Relative degradation is NaN (flagged in `rel_wer_undefined`) when the clean WER is 0."""
    ref = clean.set_index(on)[["wer", "cer"]].rename(columns={"wer": "wer_clean", "cer": "cer_clean"})
    t = table.join(ref, on=on)
    t["abs_wer_degradation"] = t["wer"] - t["wer_clean"]
    t["rel_wer_undefined"] = t["wer_clean"] == 0
    t["rel_wer_degradation"] = np.where(t["rel_wer_undefined"], np.nan,
                                        t["abs_wer_degradation"] / t["wer_clean"].replace(0, np.nan) * 100)
    t["abs_cer_degradation"] = t["cer"] - t["cer_clean"]
    if t["rel_wer_undefined"].any():
        log.warning("Clean WER is 0 for %d rows; relative WER degradation set to NaN (see rel_wer_undefined)",
                    int(t["rel_wer_undefined"].sum()))
    return t


def clean_by_noise_category(scored: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean WER/CER of the utterances paired with each noise category (each category has its own
    utterances, so its degradation must use their clean score)."""
    pairing = (scored.loc[scored["condition"] != "clean", ["model", "utt_id", "noise_category"]]
               .drop_duplicates())
    clean_rows = (scored[scored["condition"] == "clean"].drop(columns="noise_category")
                  .merge(pairing, on=["model", "utt_id"]))
    return summarize(clean_rows, ["model", "noise_category"], cfg, with_ci=False)


def clean_mismatch_candidates(scored: pd.DataFrame, min_cer: float) -> pd.DataFrame:
    """Clean utterances with CER >= min_cer for every model: probably a transcript/audio mismatch."""
    clean = scored[scored["condition"] == "clean"].assign(cer=lambda d: d["char_errors"] / d["ref_chars"] * 100)
    table = clean.pivot_table(index="utt_id", columns="model", values="cer")
    flagged = table[(table >= min_cer).all(axis=1)]
    refs = clean.drop_duplicates("utt_id").set_index("utt_id")["reference"]
    return flagged.add_prefix("clean_cer_").join(refs).reset_index()


def runtime_table(rtfx: pd.DataFrame, pred_dir, tag: str) -> pd.DataFrame:
    """RTFx next to the hardware/runtime configuration recorded by src.inference (run_<tag>_<model>.json)."""
    rows = []
    for model in rtfx["model"]:
        path = pred_dir / f"run_{tag}_{model}.json"
        meta = load_json(path) if path.exists() else {}
        env, desc = meta.get("environment", {}), meta.get("model", {})
        rows.append({"model": model, "checkpoint": desc.get("checkpoint"), "revision": desc.get("revision"),
                     "dtype": desc.get("dtype"),
                     "device": meta.get("device"), "gpu": env.get("gpu"), "batch_size": meta.get("batch_size"),
                     "peak_gpu_memory_gb": meta.get("peak_gpu_memory_gb"), "torch": env.get("torch"),
                     "transformers": env.get("transformers")})
    return rtfx.merge(pd.DataFrame(rows), on="model", how="left")


# Plots: every model keeps the same colour and position in every plot.
MODEL_LABELS = {"custom_cnn_bilstm_ctc": "CNN-BiLSTM-CTC", "wav2vec2": "Wav2Vec2 XLSR",
                "whisper": "Whisper Large-v3"}
MODEL_COLORS = {"custom_cnn_bilstm_ctc": "#2a78d6", "wav2vec2": "#eb6834", "whisper": "#1baf7a"}
CONDITION_LABELS = {"clean": "Clean", "snr_20dB": "20 dB", "snr_10dB": "10 dB", "snr_5dB": "5 dB",
                    "snr_0dB": "0 dB"}
TEXT, TEXT_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HEATMAP_COLORS = ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light = low WER
PLOT_STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
              "axes.labelcolor": TEXT, "axes.titlesize": 12, "axes.titlecolor": TEXT, "xtick.color": TEXT_SOFT,
              "ytick.color": TEXT_SOFT, "font.size": 10, "legend.frameon": False}


def model_order(models) -> list[str]:
    """CTC, Wav2Vec2, Whisper (then any other model), independent of the order in the data."""
    known = [m for m in MODEL_LABELS if m in set(models)]
    return known + sorted(set(models) - set(known))


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#8a8984")


def noise_label(category: str) -> str:
    """'CopyMachine' -> 'Copy machine'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", category).capitalize()


def plot_metric_vs_snr(by_cond: pd.DataFrame, order: list[str], metric: str):
    """One line per model from clean to 0 dB; the shaded band is the 95 % CI."""
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in model_order(by_cond["model"].unique()):
        g = by_cond[by_cond["model"] == m].set_index("condition").loc[order]
        if f"{metric}_ci_low" in g:
            ax.fill_between(x, g[f"{metric}_ci_low"], g[f"{metric}_ci_high"], color=model_color(m), alpha=0.2,
                            linewidth=0)
        ax.plot(x, g[metric], color=model_color(m), linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=model_label(m))
        ax.annotate(f"{model_label(m)}  {g[metric].iloc[-1]:.1f}", (x[-1], g[metric].iloc[-1]), xytext=(10, 0),
                    textcoords="offset points", va="center", color=TEXT)
    ax.set_xticks(x, [CONDITION_LABELS.get(c, c) for c in order])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Noise level (SNR, lower = louder noise)")
    ax.set_ylabel(f"{metric.upper()} (%)")
    ax.set_title(f"{metric.upper()} per noise level (lower is better)")
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper left")
    return fig


def plot_degradation(by_cond: pd.DataFrame, order: list[str]):
    """How much worse each model gets compared with clean audio: in percentage points and relative (%)."""
    noisy = [c for c in order if c != "clean"]
    models = model_order(by_cond["model"].unique())
    x, width = np.arange(len(noisy)), 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [("abs_wer_degradation", "WER increase vs clean (percentage points)"),
              ("rel_wer_degradation", "WER increase vs clean (relative, %)")]
    for ax, (col, title) in zip(axes, panels):
        for i, m in enumerate(models):
            g = by_cond[by_cond["model"] == m].set_index("condition").loc[noisy]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, g[col], width * 0.9, color=model_color(m),
                   label=model_label(m))
        ax.set_xticks(x, [CONDITION_LABELS.get(c, c) for c in noisy])
        ax.set_xlabel("Noise level (SNR, lower = louder noise)")
        ax.set_title(title)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_noise_heatmap(by_cat: pd.DataFrame, order: list[str]):
    """WER per noise type and noise level, one panel per model, all on the same colour scale (0-100 %)."""
    models = model_order(by_cat["model"].unique())
    levels = [c for c in order if c in set(by_cat["condition"])]
    rows = by_cat.groupby("noise_category")["wer"].mean().sort_values(ascending=False).index  # hardest on top
    cmap = LinearSegmentedColormap.from_list("wer", HEATMAP_COLORS)
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.4 * len(models) + 1.4, 4.8),
                             gridspec_kw={"width_ratios": [1] * len(models) + [0.05]})
    for i, (ax, m) in enumerate(zip(axes, models)):
        d = by_cat[by_cat["model"] == m]
        table = d.pivot(index="noise_category", columns="condition", values="wer")[levels]
        if "wer_clean" in d:  # clean WER of the same utterances that were mixed with this noise type
            table.insert(0, "clean", d.groupby("noise_category")["wer_clean"].first())
        table = table.reindex(rows)
        table.index = [noise_label(c) for c in table.index]
        table.columns = [CONDITION_LABELS.get(c, c) for c in table.columns]
        sns.heatmap(table, annot=True, fmt=".1f", cmap=cmap, vmin=0, vmax=100, linewidths=2, linecolor="white",
                    ax=ax, cbar=i == 0, cbar_ax=axes[-1] if i == 0 else None, cbar_kws={"label": "WER (%)"})
        ax.set_title(model_label(m))
        ax.set_xlabel("Noise level (SNR)")
        ax.set_ylabel("")
        ax.tick_params(length=0, axis="both")
        ax.tick_params(axis="y", rotation=0)
        if i > 0:
            ax.set_yticklabels([])
    fig.suptitle("WER (%) per noise type, same colour scale for every model", color=TEXT, fontsize=12)
    fig.tight_layout()
    return fig


def plot_accuracy_vs_speed(trade_off: pd.DataFrame):
    """RTFx against WER on clean audio (circle) and at 0 dB (square)."""
    t = trade_off.set_index("model")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for m in model_order(t.index):
        r = t.loc[m]
        points = [(c, marker) for c, marker in (("clean", "o"), ("snr_0dB", "s")) if c in r and pd.notna(r[c])]
        ys = [r[c] for c, _ in points]
        ax.plot([r["rtfx"]] * len(ys), ys, color=model_color(m), linewidth=2, alpha=0.4)
        for (_, marker), y in zip(points, ys):
            ax.scatter(r["rtfx"], y, marker=marker, s=80, color=model_color(m), edgecolor="white", linewidth=1.5,
                       zorder=3)
            ax.annotate(f"{y:.1f}", (r["rtfx"], y), xytext=(9, 0), textcoords="offset points", va="center",
                        color=TEXT_SOFT, fontsize=9)
        speed = f"{r['rtfx']:,.1f}" if r["rtfx"] < 10 else f"{r['rtfx']:,.0f}"
        ax.annotate(f"{model_label(m)}\n{speed}× real time", (r["rtfx"], max(ys)), xytext=(0, 12),
                    textcoords="offset points", ha="center", color=TEXT)
    ax.set_xscale("log")
    ax.set_xlim(t["rtfx"].min() / 2.5, t["rtfx"].max() * 2.5)
    ax.set_ylim(0, t[[c for c in ("clean", "snr_0dB") if c in t]].max().max() * 1.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("Speed: RTFx (log scale, higher = faster)")
    ax.set_ylabel("WER (%) (lower is better)")
    ax.set_title("Accuracy vs speed")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color="#8a8984", markersize=8, label="Clean"),
                       Line2D([], [], marker="s", linestyle="", color="#8a8984", markersize=8,
                              label="0 dB (loudest noise)")], loc="lower center")
    return fig


def save_figure(fig, path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(by_cond: pd.DataFrame, by_cat: pd.DataFrame, plots_dir,
               trade_off: pd.DataFrame | None = None) -> None:
    order = [c for c in CONDITION_ORDER if c in set(by_cond["condition"])]
    with plt.rc_context(PLOT_STYLE):
        for metric in ("wer", "cer"):
            save_figure(plot_metric_vs_snr(by_cond, order, metric), plots_dir / f"{metric}_vs_snr.png")
        if (by_cond["condition"] != "clean").any():
            save_figure(plot_degradation(by_cond, order), plots_dir / "wer_degradation.png")
        if not by_cat.empty:
            save_figure(plot_noise_heatmap(by_cat, order), plots_dir / "wer_heatmap_noise_category.png")
        if trade_off is not None and not trade_off.empty:
            save_figure(plot_accuracy_vs_speed(trade_off), plots_dir / "accuracy_vs_speed.png")


def accuracy_vs_speed(by_cond: pd.DataFrame, rtfx: pd.DataFrame) -> pd.DataFrame:
    """WER per condition next to RTFx (plan objective: accuracy-versus-inference-speed trade-off)."""
    wer = by_cond.pivot(index="model", columns="condition", values="wer")
    wer = wer[[c for c in CONDITION_ORDER if c in wer.columns]].reset_index()
    return rtfx[["model", "rtfx"]].merge(wer, on="model")


def check_references(pred: pd.DataFrame, conditions: pd.DataFrame | None) -> None:
    """Every model must be scored against the same, unchanged reference (plan 11.6)."""
    per_utt = pred.groupby("utt_id")["reference"].nunique()
    if (per_utt > 1).any():
        raise ValueError(f"Models disagree on the reference of {list(per_utt[per_utt > 1].index[:5])}")
    if conditions is not None:
        source = conditions.drop_duplicates("utt_id").set_index("utt_id")["transcript"]
        refs = pred.drop_duplicates("utt_id").set_index("utt_id")["reference"]
        changed = refs[refs != source.reindex(refs.index)]
        if len(changed):
            raise ValueError(f"References differ from the conditions manifest for {list(changed.index[:5])}")


def evaluate(cfg: dict, tag: str, models: list[str] | None = None) -> None:
    pred_dir = project_path(cfg["paths"]["predictions_dir"])
    frames = [pd.read_csv(pred_dir / f"predictions_{tag}_{m}.csv", keep_default_na=False)
              for m in (models or MODEL_NAMES) if (pred_dir / f"predictions_{tag}_{m}.csv").exists()]
    if not frames:
        raise FileNotFoundError(f"No predictions_{tag}_*.csv in {pred_dir}. Run src.inference first.")
    pred = pd.concat(frames, ignore_index=True)
    failed = pred["hypothesis"].eq("")
    log.info("Loaded %d predictions for models %s (%d empty hypotheses)", len(pred),
             sorted(pred["model"].unique()), int(failed.sum()))
    for col in ("audio_sec", "inference_sec"):
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    cond_path = manifest_path(cfg, f"conditions_{tag}.csv")
    check_references(pred, pd.read_csv(cond_path, keep_default_na=False) if cond_path.exists() else None)

    # keep only rows that every model has
    key = ["utt_id", "condition", "realization"]
    common = pred.groupby(key)["model"].nunique()
    common = common[common == pred["model"].nunique()].index
    pred = pred.set_index(key).loc[common].reset_index()

    scored = score_predictions(pred, cfg)
    by_cond = summarize(scored, ["model", "condition"], cfg)
    clean = by_cond[by_cond["condition"] == "clean"]
    by_cond = add_degradation(by_cond, clean, ["model"])
    by_cond["condition"] = pd.Categorical(by_cond["condition"], CONDITION_ORDER, ordered=True)
    by_cond = by_cond.sort_values(["model", "condition"]).reset_index(drop=True)
    by_cond["condition"] = by_cond["condition"].astype(str)

    by_cat = summarize(scored[scored["condition"] != "clean"], ["model", "noise_category", "condition"], cfg,
                       with_ci=False)
    if not by_cat.empty:
        by_cat = add_degradation(by_cat, clean_by_noise_category(scored, cfg), ["model", "noise_category"])

    mismatch = clean_mismatch_candidates(scored, cfg["evaluation"]["mismatch_clean_cer"])
    if not mismatch.empty:
        log.warning("%d clean utterances have CER >= %.0f%% for every model (possible transcript/audio "
                    "mismatch, see clean_mismatch_candidates.csv)", len(mismatch),
                    cfg["evaluation"]["mismatch_clean_cer"])

    rank = by_cond.pivot(index="condition", columns="model", values="wer").rank(axis=1).astype(int)
    rtfx = summarize(scored, ["model"], cfg, with_ci=False)[["model", "audio_sec", "inference_sec", "rtfx"]]
    rtfx = runtime_table(rtfx, pred_dir, tag)
    trade_off = accuracy_vs_speed(by_cond, rtfx)

    out_dir = ensure_dir(pred_dir / f"results_{tag}")
    scored.to_csv(out_dir / "scored_utterances.csv", index=False)
    by_cond.to_csv(out_dir / "summary_by_model_condition.csv", index=False)
    by_cat.to_csv(out_dir / "summary_by_model_noise_category.csv", index=False)
    rank.to_csv(out_dir / "model_rank_by_condition.csv")
    rtfx.to_csv(out_dir / "rtfx.csv", index=False)
    mismatch.to_csv(out_dir / "clean_mismatch_candidates.csv", index=False)
    trade_off.to_csv(out_dir / "accuracy_vs_speed.csv", index=False)
    runs = {m: str(pred_dir / f"run_{tag}_{m}.json") for m in scored["model"].unique()}
    save_json({"tag": tag, "normalization": cfg.get("text_normalization"), "run_metadata": runs,
               "n_common_rows": len(common)}, out_dir / "evaluation_info.json")

    plots_dir = ensure_dir(project_path(cfg["paths"]["plots_dir"]) / tag)
    make_plots(by_cond, by_cat, plots_dir, trade_off)

    cols = ["model", "condition", "n_utterances", "wer", "cer", "abs_wer_degradation", "rel_wer_degradation"]
    log.info("Results (%s):\n%s", tag, by_cond[cols].round(2).to_string(index=False))
    log.info("RTFx:\n%s", rtfx[["model", "rtfx", "gpu", "batch_size", "dtype"]].round(2).to_string(index=False))
    log.info("Tables in %s, plots in %s", out_dir, plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--tag", default=None, help="Override the predictions tag, e.g. test_pilot_limit5")
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_NAMES,
                        help="Models to compare (default: every model with predictions). The primary "
                             "comparison uses exactly one Whisper checkpoint.")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.tag or f"{args.split}{'_pilot' if args.pilot else ''}", args.models)


if __name__ == "__main__":
    main()
