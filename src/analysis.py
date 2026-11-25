"""Extra analysis of the scored test results, used to answer the research questions.

Reads `output/predictions/results_<tag>/scored_utterances.csv` (written by src.evaluate) and writes tables to
`output/predictions/results_<tag>/analysis/` plus one plot to `output/plots/<tag>/error_types.png`:

- error_types.csv            substitutions, deletions and insertions per model and condition, and how many
                             utterances are perfect, empty, or much longer than the reference (repetition loops)
- pairwise_differences.csv   WER difference between every pair of models per condition with a paired
                             bootstrap 95 % CI: is the ranking real or within sampling noise?
- noise_category_ranking.csv noise types from hardest to easiest per model and SNR, and the spread between them
- by_speaker.csv             WER per test speaker
- examples.csv               three random utterances with the output of every model in every condition
- longest_outputs.csv        per model, the outputs that are longest compared with their reference

Usage:
    python -m src.evaluate [--pilot]      # first: writes scored_utterances.csv
    python -m src.analysis [--pilot]
"""

from __future__ import annotations

import argparse
import itertools

import jiwer
import numpy as np
import pandas as pd

from src.evaluate import (CONDITION_LABELS, CONDITION_ORDER, GRID, PLOT_STYLE, model_label, model_order, plt,
                          save_figure)
from src.utils import ensure_dir, get_logger, load_config, project_path

log = get_logger("analysis")
ERROR_TYPE_COLORS = {"substitution": "#6250d6", "deletion": "#eda100", "insertion": "#e87ba4"}


def sort_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Rows in the fixed model order (CTC, Wav2Vec2, Whisper) and condition order (clean ... 0 dB)."""
    keys = {}
    if "model" in table:
        keys["model"] = {m: i for i, m in enumerate(model_order(table["model"].unique()))}
    if "condition" in table:
        keys["condition"] = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return table.sort_values(list(keys), key=lambda col: col.map(keys[col.name]), kind="stable").reset_index(drop=True)


def edit_operations(ref: str, hyp: str) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) at word level for one utterance."""
    w = jiwer.process_words(ref, hyp or "")
    return w.substitutions, w.deletions, w.insertions


def error_types(scored: pd.DataFrame) -> pd.DataFrame:
    """What kind of mistakes each model makes, per condition. Rates are % of reference words, so
    substitution + deletion + insertion rate = WER."""
    s = scored.copy()
    s[["sub", "del", "ins"]] = np.array([edit_operations(r, h) for r, h in zip(s["ref_norm"], s["hyp_norm"])])
    if not (s[["sub", "del", "ins"]].sum(axis=1) == s["word_errors"]).all():
        raise ValueError("Substitutions + deletions + insertions do not add up to the word errors")
    s["hyp_words"] = s["hyp_norm"].str.split().str.len()
    s["utt_wer"] = s["word_errors"] / s["ref_words"] * 100
    rows = []
    for (model, condition), g in s.groupby(["model", "condition"], sort=False):
        n = g["ref_words"].sum()
        rows.append({
            "model": model, "condition": condition,
            "wer": g["word_errors"].sum() / n * 100,
            "substitution_rate": g["sub"].sum() / n * 100,
            "deletion_rate": g["del"].sum() / n * 100,
            "insertion_rate": g["ins"].sum() / n * 100,
            "output_length_ratio": g["hyp_words"].sum() / n,  # words written / words spoken
            "pct_perfect": (g["utt_wer"] == 0).mean() * 100,
            "pct_empty_output": (g["hyp_words"] == 0).mean() * 100,
            "pct_wer_100_or_more": (g["utt_wer"] >= 100).mean() * 100,
            "pct_output_over_2x_reference": (g["hyp_words"] > 2 * g["ref_words"]).mean() * 100,
        })
    return sort_rows(pd.DataFrame(rows))


def paired_bootstrap_difference(err_a: np.ndarray, err_b: np.ndarray, n: np.ndarray, resamples: int,
                                level: float, seed: int) -> tuple[float, float, float]:
    """WER(a) - WER(b) on the same utterances, with a paired bootstrap CI.

    Each resample draws utterances with replacement and uses the *same* draw for both models, so the
    difference between the models is measured on identical audio every time.
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(n), len(n))
        diffs.append((err_a[chosen].sum() - err_b[chosen].sum()) / n[chosen].sum() * 100)
    low, high = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return (err_a.sum() - err_b.sum()) / n.sum() * 100, float(low), float(high)


def pairwise_differences(scored: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ecfg = cfg["evaluation"]
    models = model_order(scored["model"].unique())
    rows = []
    for condition, d in scored.groupby("condition", sort=False):
        errors = d.pivot(index=["utt_id", "realization"], columns="model", values="word_errors")
        n = d.drop_duplicates(["utt_id", "realization"]).set_index(["utt_id", "realization"]).loc[
            errors.index, "ref_words"].values
        for a, b in itertools.combinations(models, 2):
            diff, low, high = paired_bootstrap_difference(errors[a].values, errors[b].values, n,
                                                          ecfg["bootstrap_resamples"], ecfg["confidence_level"],
                                                          cfg["project"]["seed"])
            better = a if high < 0 else b if low > 0 else "no clear difference"  # CI excludes 0 or not
            rows.append({"condition": condition, "model_a": a, "model_b": b,
                         "wer_a": errors[a].sum() / n.sum() * 100, "wer_b": errors[b].sum() / n.sum() * 100,
                         "wer_difference_a_minus_b": diff, "ci_low": low, "ci_high": high, "better_model": better})
    return sort_rows(pd.DataFrame(rows))


def wer_table(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = scored.groupby(keys)[["word_errors", "ref_words"]].sum()
    return (g["word_errors"] / g["ref_words"] * 100).rename("wer").reset_index()


def noise_category_ranking(scored: pd.DataFrame) -> pd.DataFrame:
    """Noise types ordered from hardest (highest WER) to easiest, per model and SNR."""
    wer = wer_table(scored[scored["condition"] != "clean"], ["model", "condition", "noise_category"])
    rows = []
    for (model, condition), g in wer.groupby(["model", "condition"]):
        g = g.sort_values("wer", ascending=False)
        rows.append({"model": model, "condition": condition,
                     "hardest": g["noise_category"].iloc[0], "hardest_wer": g["wer"].iloc[0],
                     "easiest": g["noise_category"].iloc[-1], "easiest_wer": g["wer"].iloc[-1],
                     "spread_pp": g["wer"].iloc[0] - g["wer"].iloc[-1],
                     "ranking_hardest_first": " > ".join(g["noise_category"])})
    return sort_rows(pd.DataFrame(rows))


def by_speaker(scored: pd.DataFrame) -> pd.DataFrame:
    wer = wer_table(scored, ["model", "condition", "speaker_id"])
    return sort_rows(wer.pivot(index=["model", "condition"], columns="speaker_id", values="wer").reset_index())


def examples(scored: pd.DataFrame, seed: int, n: int = 3) -> pd.DataFrame:
    """The output of every model for a few randomly chosen utterances, in every condition."""
    utts = scored["utt_id"].drop_duplicates().sample(n=n, random_state=seed)
    d = scored[scored["utt_id"].isin(utts)].assign(wer=lambda x: x["word_errors"] / x["ref_words"] * 100)
    wide = d.pivot(index=["utt_id", "condition"], columns="model", values=["hyp_norm", "wer"])
    wide.columns = [f"{'output' if v == 'hyp_norm' else 'wer'}_{m}" for v, m in wide.columns]
    info = d.drop_duplicates(["utt_id", "condition"])[["utt_id", "condition", "noise_category", "ref_norm"]]
    table = info.merge(wide.reset_index(), on=["utt_id", "condition"]).rename(columns={"ref_norm": "reference"})
    return table.sort_values(["utt_id", "condition"], key=lambda c: c.map(
        {k: i for i, k in enumerate(CONDITION_ORDER)}) if c.name == "condition" else c).reset_index(drop=True)


def longest_outputs(scored: pd.DataFrame, per_model: int = 3) -> pd.DataFrame:
    """Outputs that are much longer than what was said (e.g. a phrase repeated over and over)."""
    s = scored.assign(output_words=scored["hyp_norm"].str.split().str.len(),
                      wer=scored["word_errors"] / scored["ref_words"] * 100)
    s["length_ratio"] = s["output_words"] / s["ref_words"]
    top = s.sort_values("length_ratio", ascending=False).groupby("model").head(per_model)
    cols = ["model", "condition", "noise_category", "utt_id", "ref_norm", "hyp_norm", "length_ratio", "wer"]
    return sort_rows(top[cols].rename(columns={"ref_norm": "reference", "hyp_norm": "output"}))


def plot_error_types(errors: pd.DataFrame):
    """Stacked substitution / deletion / insertion rates (together = WER), one panel per model."""
    models = model_order(errors["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.6), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        g = errors[errors["model"] == m].set_index("condition")
        conditions = [c for c in CONDITION_ORDER if c in g.index]
        x, bottom = np.arange(len(conditions)), np.zeros(len(conditions))
        for kind, color in ERROR_TYPE_COLORS.items():
            values = g.loc[conditions, f"{kind}_rate"].values
            ax.bar(x, values, 0.7, bottom=bottom, color=color, label=kind.capitalize(), edgecolor="white",
                   linewidth=1)
            bottom += values
        for xi, total in zip(x, bottom):
            ax.annotate(f"{total:.1f}", (xi, total), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9)
        ax.set_xticks(x, [CONDITION_LABELS.get(c, c) for c in conditions])
        ax.set_title(model_label(m))
        ax.set_xlabel("Noise level (SNR)")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Errors (% of reference words) = WER")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Type of word errors per noise level", fontsize=12)
    fig.tight_layout()
    return fig


def analyse(cfg: dict, tag: str) -> None:
    results_dir = project_path(cfg["paths"]["predictions_dir"]) / f"results_{tag}"
    scored_path = results_dir / "scored_utterances.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"{scored_path} not found. Run src.evaluate first.")
    scored = pd.read_csv(scored_path, keep_default_na=False, low_memory=False)
    scored["noise_category"] = scored["noise_category"].astype(str)
    out_dir = ensure_dir(results_dir / "analysis")
    seed = cfg["project"]["seed"]

    errors = error_types(scored)
    tables = {"error_types": errors, "pairwise_differences": pairwise_differences(scored, cfg),
              "noise_category_ranking": noise_category_ranking(scored), "by_speaker": by_speaker(scored),
              "examples": examples(scored, seed), "longest_outputs": longest_outputs(scored)}
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    with plt.rc_context(PLOT_STYLE):
        save_figure(plot_error_types(errors), ensure_dir(project_path(cfg["paths"]["plots_dir"]) / tag)
                    / "error_types.png")

    show = ["model", "condition", "wer", "substitution_rate", "deletion_rate", "insertion_rate",
            "pct_output_over_2x_reference"]
    log.info("Error types (%s):\n%s", tag, errors[show].round(2).to_string(index=False))
    log.info("Pairwise WER differences:\n%s", tables["pairwise_differences"].round(2).to_string(index=False))
    log.info("Tables in %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    analyse(load_config(args.config), f"{args.split}{'_pilot' if args.pilot else ''}")


if __name__ == "__main__":
    main()
