"""CNN-BiLSTM-CTC model, trained from scratch.

Input:  log-Mel features (80 bins, 25 ms window, 10 ms hop)
CNN:    local time-frequency patterns (time downsampled x2)
BiLSTM: context in both directions
CTC:    one character per frame plus a blank, greedy decoding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

BLANK = "<blank>"


class CharVocab:
    """Character vocabulary built from *normalized* training transcripts. Index 0 is the CTC blank."""

    def __init__(self, chars: list[str]):
        self.itos = [BLANK] + sorted(set(chars) - {BLANK})
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts) -> "CharVocab":
        return cls([c for t in texts for c in t])

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode_ctc(self, ids: list[int]) -> str:
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.itos[i])
            prev = i
        return " ".join("".join(out).split())

    def __len__(self) -> int:
        return len(self.itos)


class LogMelFeatures(nn.Module):
    """Log-Mel filterbank with optional per-utterance mean/variance normalization."""

    def __init__(self, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80,
                 log_offset=1e-6, per_utterance_norm=True):
        super().__init__()
        self.hop_length = hop_length
        self.log_offset = log_offset
        self.per_utterance_norm = per_utterance_norm
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=n_mels, power=2.0)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wave: (B, T) -> features (B, n_mels, frames), frame lengths (B,)"""
        with torch.autocast(device_type=wave.device.type, enabled=False):
            feats = torch.log(self.mel(wave.float()) + self.log_offset)
        frame_lengths = lengths // self.hop_length + 1
        if self.per_utterance_norm:
            # per utterance, every Mel bin to mean 0 and variance 1 (padding frames ignored, stay 0)
            normalized = torch.zeros_like(feats)
            for i in range(feats.size(0)):
                n_frames = int(frame_lengths[i])
                real = feats[i, :, :n_frames]                      # (n_mels, n_frames)
                mean = real.mean(dim=-1, keepdim=True)
                var = ((real - mean) ** 2).mean(dim=-1, keepdim=True)
                normalized[i, :, :n_frames] = (real - mean) / torch.sqrt(var + 1e-5)
            feats = normalized
        return feats, frame_lengths


class CNNBiLSTMCTC(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80, cnn_channels=(32, 32), lstm_hidden: int = 256,
                 lstm_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        c1, c2 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=(2, 2), padding=1), nn.BatchNorm2d(c1), nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=(2, 1), padding=1), nn.BatchNorm2d(c2), nn.GELU(),
            nn.Dropout(dropout),
        )
        freq_out = (((n_mels + 1) // 2) + 1) // 2
        self.proj = nn.Linear(c2 * freq_out, lstm_hidden)
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * lstm_hidden, vocab_size)

    @staticmethod
    def output_lengths(frame_lengths: torch.Tensor) -> torch.Tensor:
        return (frame_lengths - 1) // 2 + 1  # conv1 time stride 2, kernel 3, padding 1

    def forward(self, feats: torch.Tensor, frame_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats: (B, n_mels, T) -> log-probs (B, T', vocab), lengths (B,)"""
        x = self.cnn(feats.unsqueeze(1))                 # (B, C, F', T')
        b, c, f, t = x.shape
        x = self.proj(x.permute(0, 3, 1, 2).reshape(b, t, c * f))
        out_lengths = self.output_lengths(frame_lengths).clamp(max=t)
        packed = nn.utils.rnn.pack_padded_sequence(x, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=t)
        logits = self.classifier(self.dropout(x))
        return logits.log_softmax(dim=-1), out_lengths


class CustomASR(nn.Module):
    """Feature extractor + acoustic model packaged together for training and inference."""

    def __init__(self, vocab: CharVocab, feature_cfg: dict, arch_cfg: dict):
        super().__init__()
        self.vocab = vocab
        self.features = LogMelFeatures(**feature_cfg)
        self.model = CNNBiLSTMCTC(len(vocab), n_mels=feature_cfg["n_mels"],
                                  cnn_channels=tuple(arch_cfg["cnn_channels"]),
                                  lstm_hidden=arch_cfg["lstm_hidden"], lstm_layers=arch_cfg["lstm_layers"],
                                  dropout=arch_cfg["dropout"])

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor):
        feats, frame_lengths = self.features(wave, lengths)
        return self.model(feats, frame_lengths)

    @torch.no_grad()
    def greedy_decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
        ids = log_probs.argmax(dim=-1).cpu()
        return [self.vocab.decode_ctc(ids[i, : int(lengths[i])].tolist()) for i in range(ids.size(0))]


def pad_batch(waves: list, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(w) for w in waves], dtype=torch.long)
    batch = torch.zeros(len(waves), int(lengths.max()))
    for i, w in enumerate(waves):
        batch[i, : len(w)] = torch.as_tensor(w)
    return batch.to(device), lengths.to(device)


def save_checkpoint(path, model: CustomASR, cfg_section: dict, extra: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "vocab": model.vocab.itos[1:],
                "config": cfg_section, **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> CustomASR:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CustomASR(CharVocab(ckpt["vocab"]), cfg["features"], cfg["architecture"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
