import numpy as np

from src.audio_validation import speech_in_noise_flags, split_into_chunks


def test_split_into_chunks_joins_a_tiny_tail():
    # 300.004 s at 16 kHz: fifteen 20-s pieces plus a 64-sample tail that used to crash Wav2Vec2
    audio = np.zeros(16000 * 300 + 64, dtype=np.float32)
    pieces = split_into_chunks(audio, chunk_size=16000 * 20, min_size=16000)
    assert len(pieces) == 15
    assert len(pieces[-1]) == 16000 * 20 + 64
    assert sum(len(p) for p in pieces) == len(audio)  # no sample lost
    # a long enough tail stays a separate piece
    assert len(split_into_chunks(np.zeros(16000 * 25), 16000 * 20, 16000)) == 2


def test_speech_in_noise_flags():
    rows = speech_in_noise_flags(["", "a i", "saya pergi ke pasar hari ini"], [10.0, 10.0, 2.0], 2.0)
    assert [r["speech_suspected"] for r in rows] == [False, False, True]
    assert rows[1]["transcribed_chars"] == 2 and rows[2]["chars_per_sec"] == 11.5
    assert speech_in_noise_flags(["x"], [0.0], 2.0)[0]["chars_per_sec"] == 0.0

