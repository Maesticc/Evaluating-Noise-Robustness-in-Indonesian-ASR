import pandas as pd

from src.inference import remaining_rows


def test_remaining_rows_skips_rows_already_done():
    rows = pd.DataFrame({"utt_id": ["a", "a", "b", "b"], "condition": ["clean", "snr_0dB", "clean", "snr_0dB"],
                         "realization": [0, 0, 0, 0], "duration_sec": [1.0, 1.0, 2.0, 2.0]})
    # a partial file read back from CSV holds strings; the keys must still match
    done = pd.DataFrame({"utt_id": ["a", "b"], "condition": ["snr_0dB", "clean"], "realization": ["0", "0"]})
    todo = remaining_rows(rows, done)
    assert list(zip(todo["utt_id"], todo["condition"])) == [("a", "clean"), ("b", "snr_0dB")]
    assert len(remaining_rows(rows, pd.DataFrame(columns=["utt_id", "condition", "realization"]))) == 4
