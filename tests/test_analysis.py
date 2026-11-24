import numpy as np
import pandas as pd

from src.analysis import error_types, noise_category_ranking, paired_bootstrap_difference


def test_error_types_add_up_to_wer():
    scored = pd.DataFrame({
        "model": ["m"] * 2, "condition": ["clean"] * 2,
        "ref_norm": ["saya makan nasi", "dia pergi"],
        "hyp_norm": ["saya minum nasi goreng", ""],   # 1 substitution + 1 insertion, then 2 deletions
        "word_errors": [2, 2], "ref_words": [3, 2],
    })
    row = error_types(scored).iloc[0]
    assert np.isclose(row["substitution_rate"], 20) and np.isclose(row["insertion_rate"], 20)
    assert np.isclose(row["deletion_rate"], 40)
    assert np.isclose(row["substitution_rate"] + row["deletion_rate"] + row["insertion_rate"], row["wer"])
    assert row["pct_empty_output"] == 50


def test_paired_bootstrap_difference():
    n = np.full(200, 10)
    same = np.arange(200) % 5
    diff, low, high = paired_bootstrap_difference(same, same, n, 200, 0.95, 0)
    assert diff == low == high == 0                      # identical models: no difference at all
    diff, low, high = paired_bootstrap_difference(same, same + 2, n, 200, 0.95, 0)
    assert np.isclose(diff, -20) and high < 0            # a is clearly better: the CI excludes 0


def test_noise_category_ranking_orders_hardest_first():
    scored = pd.DataFrame({
        "model": ["m"] * 3, "condition": ["snr_0dB"] * 3, "noise_category": ["Typing", "CopyMachine", "Munching"],
        "word_errors": [2, 9, 5], "ref_words": [10, 10, 10],
    })
    row = noise_category_ranking(scored).iloc[0]
    assert row["hardest"] == "CopyMachine" and row["easiest"] == "Typing"
    assert row["ranking_hardest_first"] == "CopyMachine > Munching > Typing"
    assert np.isclose(row["spread_pp"], 70)
