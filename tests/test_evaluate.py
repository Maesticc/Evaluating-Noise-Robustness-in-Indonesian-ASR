import numpy as np
import pandas as pd

from src.evaluate import add_degradation, clean_mismatch_candidates, error_counts


def test_error_counts():
    # one substitution, one deletion over 4 reference words
    w_err, n_words, c_err, n_chars = error_counts("saya makan nasi goreng", "saya minum nasi")
    assert (w_err, n_words) == (2, 4)
    assert n_chars == len("saya makan nasi goreng") and c_err > 0
    assert error_counts("halo", None)[:2] == (1, 1)


def test_relative_degradation_with_zero_clean_wer():
    table = pd.DataFrame({"model": ["a", "b"], "wer": [10.0, 5.0], "cer": [4.0, 2.0]})
    clean = pd.DataFrame({"model": ["a", "b"], "wer": [0.0, 2.5], "cer": [0.0, 1.0]})
    t = add_degradation(table, clean, ["model"]).set_index("model")
    assert bool(t.loc["a", "rel_wer_undefined"]) and np.isnan(t.loc["a", "rel_wer_degradation"])
    assert t.loc["a", "abs_wer_degradation"] == 10.0
    assert t.loc["b", "rel_wer_degradation"] == 100.0


def test_clean_mismatch_needs_every_model():
    d = pd.DataFrame({"model": ["a", "b", "a", "b", "a"], "utt_id": ["u1", "u1", "u2", "u2", "u1"],
                      "condition": ["clean"] * 4 + ["snr_0dB"], "char_errors": [5, 4, 5, 0, 9],
                      "ref_chars": [10] * 5, "reference": ["r1", "r1", "r2", "r2", "r1"]})
    out = clean_mismatch_candidates(d, 30.0)
    assert out["utt_id"].tolist() == ["u1"]
    assert out.loc[0, "clean_cer_a"] == 50 and out.loc[0, "reference"] == "r1"
    assert clean_mismatch_candidates(d, 60.0).empty


def test_runtime_table_joins_run_metadata(tmp_path):
    import json

    from src.evaluate import runtime_table

    meta = {"device": "cuda", "batch_size": 8, "peak_gpu_memory_gb": 1.9,
            "model": {"checkpoint": "openai/whisper-large-v3", "revision": "abc123", "dtype": "torch.float16"},
            "environment": {"gpu": "RTX", "torch": "2.11", "transformers": "5.23"}}
    (tmp_path / "run_t_whisper.json").write_text(json.dumps(meta))
    rtfx = pd.DataFrame({"model": ["whisper", "wav2vec2"], "rtfx": [5.0, 20.0]})
    t = runtime_table(rtfx, tmp_path, "t").set_index("model")
    assert t.loc["whisper", "gpu"] == "RTX" and t.loc["whisper", "batch_size"] == 8
    assert t.loc["whisper", "revision"] == "abc123"
    assert pd.isna(t.loc["wav2vec2", "gpu"])  # missing run metadata does not drop the row


def test_check_references_detects_changes():
    import pytest

    from src.evaluate import check_references

    pred = pd.DataFrame({"model": ["a", "b"], "utt_id": ["u1", "u1"], "reference": ["Halo dunia", "Halo dunia"]})
    check_references(pred, pd.DataFrame({"utt_id": ["u1"], "transcript": ["Halo dunia"]}))
    with pytest.raises(ValueError):
        check_references(pred, pd.DataFrame({"utt_id": ["u1"], "transcript": ["Halo, dunia"]}))
    with pytest.raises(ValueError):
        check_references(pred.assign(reference=["Halo dunia", "Halo"]), None)


def test_accuracy_vs_speed_table():
    from src.evaluate import accuracy_vs_speed

    by_cond = pd.DataFrame({"model": ["a", "a"], "condition": ["snr_0dB", "clean"], "wer": [20.0, 5.0]})
    t = accuracy_vs_speed(by_cond, pd.DataFrame({"model": ["a"], "rtfx": [3.0]}))
    assert t.columns.tolist() == ["model", "rtfx", "clean", "snr_0dB"]
    assert t.iloc[0].tolist() == ["a", 3.0, 5.0, 20.0]
