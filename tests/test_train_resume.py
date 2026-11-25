import pytest
import torch

from src.custom_cnn_bilstm_ctc import CharVocab, CustomASR
from src.train_custom_model import load_training_state, save_training_state

FEATURES = {"sample_rate": 16000, "n_fft": 400, "win_length": 400, "hop_length": 160, "n_mels": 80,
            "log_offset": 1e-6, "per_utterance_norm": True}
ARCH = {"cnn_channels": [4, 4], "lstm_hidden": 8, "lstm_layers": 1, "dropout": 0.0}


def _new_run():
    """A tiny model with optimizer, learning-rate schedule and gradient scaler (CPU only)."""
    model = CustomASR(CharVocab(list("ab ")), FEATURES, ARCH)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-3, epochs=2, steps_per_epoch=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optim, sched, scaler


def test_resume_restores_the_saved_run(tmp_path):
    settings = {"epochs": 2, "n_train": 10, "vocab": ["<blank>", " ", "a", "b"]}
    model, optim, sched, scaler = _new_run()
    optim.step()
    sched.step()  # one step, so the schedule is no longer at its start
    history = [{"epoch": 1, "val_cer": 0.5}]
    save_training_state(tmp_path / "last.pt", settings, 1, 0.5, 0, history, model, optim, sched, scaler)

    fresh = _new_run()  # different random weights
    start, best, patience, hist = load_training_state(tmp_path / "last.pt", settings, *fresh, "cpu")
    assert (start, best, patience, hist) == (2, 0.5, 0, history)
    assert fresh[2].last_epoch == sched.last_epoch  # learning-rate schedule continues where it stopped
    for saved, loaded in zip(model.state_dict().values(), fresh[0].state_dict().values()):
        assert torch.equal(saved, loaded)


def test_resume_refuses_other_settings(tmp_path):
    settings = {"epochs": 2, "n_train": 10, "vocab": ["<blank>", "a"]}
    save_training_state(tmp_path / "last.pt", settings, 1, 0.5, 0, [], *_new_run())
    with pytest.raises(ValueError):
        load_training_state(tmp_path / "last.pt", {**settings, "epochs": 50}, *_new_run(), "cpu")
    with pytest.raises(FileNotFoundError):
        load_training_state(tmp_path / "missing.pt", settings, *_new_run(), "cpu")
