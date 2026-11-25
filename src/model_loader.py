"""The three ASR models behind one interface: `transcribe(list of waveforms) -> list of texts`.
Each pretrained model uses its own official processor; no features are shared between models."""

from __future__ import annotations

import numpy as np
import torch

from src.custom_cnn_bilstm_ctc import load_checkpoint, pad_batch
from src.utils import project_path

MODEL_NAMES = ("custom_cnn_bilstm_ctc", "wav2vec2", "whisper")
WHISPER_MAX_SEC = 30.0


class ASRModel:
    name: str
    sample_rate: int = 16000

    def transcribe(self, waves: list[np.ndarray]) -> list[str]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class CustomModel(ASRModel):
    name = "custom_cnn_bilstm_ctc"

    def __init__(self, cfg: dict, device: str):
        self.mcfg = cfg["models"][self.name]
        self.device = device
        self.checkpoint = project_path(self.mcfg["checkpoint"])
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"{self.checkpoint} not found. Train it with: python -m src.train_custom_model")
        self.model = load_checkpoint(self.checkpoint, device)
        self.sample_rate = self.mcfg["features"]["sample_rate"]

    @torch.no_grad()
    def transcribe(self, waves):
        wave, lengths = pad_batch(waves, self.device)
        log_probs, out_lengths = self.model(wave, lengths)
        return self.model.greedy_decode(log_probs, out_lengths)

    def describe(self):
        return {"name": self.name, "checkpoint": str(self.checkpoint), "decoding": "ctc_greedy",
                "features": self.mcfg["features"], "architecture": self.mcfg["architecture"]}


class PretrainedModel(ASRModel):
    """A Hugging Face checkpoint, loaded at the exact commit (`revision`) given in the config."""

    def __init__(self, cfg: dict, device: str, processor_class, model_class):
        self.mcfg = cfg["models"][self.name]
        self.device = device
        self.dtype = torch.float16 if (self.mcfg.get("fp16") and device == "cuda") else torch.float32
        self.revision = self.mcfg.get("revision")
        self.processor = processor_class.from_pretrained(self.mcfg["checkpoint"], revision=self.revision)
        self.model = model_class.from_pretrained(self.mcfg["checkpoint"], revision=self.revision,
                                                 torch_dtype=self.dtype).to(device).eval()
        self.sample_rate = self.processor.feature_extractor.sampling_rate


class Wav2Vec2Model(PretrainedModel):
    name = "wav2vec2"

    def __init__(self, cfg: dict, device: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        super().__init__(cfg, device, Wav2Vec2Processor, Wav2Vec2ForCTC)

    @torch.no_grad()
    def transcribe(self, waves):
        inputs = self.processor(waves, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        kwargs = {"input_values": inputs.input_values.to(self.device, self.dtype)}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        logits = self.model(**kwargs).logits
        return self.processor.batch_decode(logits.argmax(dim=-1))

    def describe(self):
        return {"name": self.name, "checkpoint": self.mcfg["checkpoint"], "revision": self.revision,
                "decoding": "ctc_greedy", "dtype": str(self.dtype), "sample_rate": self.sample_rate,
                "return_attention_mask": self.processor.feature_extractor.return_attention_mask}


class WhisperModel(PretrainedModel):
    name = "whisper"

    def __init__(self, cfg: dict, device: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        super().__init__(cfg, device, WhisperProcessor, WhisperForConditionalGeneration)
        self.gen_kwargs = {"language": self.mcfg["language"], "task": self.mcfg["task"],
                           "num_beams": self.mcfg["num_beams"], "max_new_tokens": self.mcfg["max_new_tokens"]}

    def _generate(self, inputs, long_form: bool) -> list[str]:
        kwargs = dict(self.gen_kwargs)
        if long_form:  # sequential long-form decoding for audio > 30 s
            kwargs.update({"attention_mask": inputs.attention_mask.to(self.device), "return_timestamps": True})
            kwargs.pop("max_new_tokens")
        ids = self.model.generate(inputs.input_features.to(self.device, self.dtype), **kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def transcribe(self, waves):
        out: list[str | None] = [None] * len(waves)
        short = [i for i, w in enumerate(waves) if len(w) / self.sample_rate <= WHISPER_MAX_SEC]
        if short:
            inputs = self.processor([waves[i] for i in short], sampling_rate=self.sample_rate, return_tensors="pt")
            for i, text in zip(short, self._generate(inputs, long_form=False)):
                out[i] = text
        for i in (i for i in range(len(waves)) if out[i] is None):
            inputs = self.processor(waves[i], sampling_rate=self.sample_rate, return_tensors="pt",
                                    truncation=False, padding="longest", return_attention_mask=True)
            out[i] = self._generate(inputs, long_form=True)[0]
        return [t.strip() for t in out]

    def describe(self):
        return {"name": self.name, "checkpoint": self.mcfg["checkpoint"], "revision": self.revision,
                "dtype": str(self.dtype), "generation": self.gen_kwargs, "sample_rate": self.sample_rate,
                "long_form": f"sequential with timestamps for audio > {WHISPER_MAX_SEC:.0f} s"}


def load_model(name: str, cfg: dict, device: str) -> ASRModel:
    classes = {"custom_cnn_bilstm_ctc": CustomModel, "wav2vec2": Wav2Vec2Model, "whisper": WhisperModel}
    if name not in classes:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    return classes[name](cfg, device)
