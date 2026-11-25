import numpy as np
import pytest

from src.noise_mixing import mix_at_snr, pairing_noise, select_noise_segment
from src.utils import load_config

SR = 16000


@pytest.fixture(scope="module")
def cfg():
    return load_config(None)


def _speech(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (rng.random(t.size) > 0.0)
    x[: SR // 4] = 0.0  # leading silence must not change the active-speech SNR
    return x.astype(np.float32)


@pytest.mark.parametrize("snr", [20.0, 10.0, 5.0, 0.0])
def test_mix_reaches_target_snr(cfg, snr):
    speech = _speech()
    noise = np.random.default_rng(1).normal(0, 0.1, speech.size).astype(np.float32)
    mixture, info = mix_at_snr(speech, noise, snr, cfg["mixing"], SR)
    assert mixture.shape == speech.shape
    assert abs(info["achieved_snr_db"] - snr) <= cfg["mixing"]["snr_tolerance_db"]
    assert not info["residual_clipping"]


def test_output_gain_keeps_snr(cfg):
    speech = _speech()
    noise = np.random.default_rng(2).normal(0, 0.5, speech.size).astype(np.float32)
    _, ref = mix_at_snr(speech, noise, 0.0, cfg["mixing"], SR, gain=1.0)
    _, scaled = mix_at_snr(speech, noise, 0.0, cfg["mixing"], SR, gain=0.5)
    assert scaled["achieved_snr_db"] == pytest.approx(ref["achieved_snr_db"], abs=1e-3)


def test_select_noise_segment_crops_and_tiles():
    noise = np.arange(10, dtype=np.float32)
    assert select_noise_segment(noise, 4, 3).tolist() == [3, 4, 5, 6]
    tiled = select_noise_segment(noise, 25, 7)
    assert tiled.size == 25 and tiled[:3].tolist() == [7, 8, 9] and tiled[3] == 0


def test_test_noise_only_used_for_test_split(cfg):
    test = pairing_noise(cfg, "test")
    dev = pairing_noise(cfg, "val")
    assert set(test["noise_pool"]) == {cfg["mixing"]["eval_noise_pool"]}
    assert set(dev["noise_pool"]) == {cfg["mixing"]["augmentation_noise_pool"]}
    assert set(dev["noise_category"]) <= set(test["noise_category"])
    assert not set(dev["noise_path"]) & set(test["noise_path"])
