"""Shared configuration, reproducibility, logging and helper functions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML experiment configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def project_path(relative: str | os.PathLike) -> Path:
    """Resolve a path from the config relative to the repository root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_path(cfg: dict, name: str) -> Path:
    return ensure_dir(project_path(cfg["paths"]["manifests_dir"])) / name


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary values (independent of PYTHONHASHSEED)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def get_logger(name: str, log_file: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def git_info() -> dict:
    """The Git commit of the code that produced a result, and whether the code (`src/`, `config/`) was
    changed after that commit. None when the project is not a Git checkout."""
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_code_changed": git("status", "--porcelain", "--", "src", "config") != ""}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_code_changed": None}


def environment_info() -> dict:
    """Software and hardware details recorded with every run."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        **git_info(),
    }
    from importlib.metadata import PackageNotFoundError, version

    # Installed distribution metadata: also covers packages without __version__ (e.g. jiwer).
    for pkg in ("numpy", "pandas", "torch", "torchaudio", "transformers", "jiwer", "librosa", "soundfile"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        pass
    return info


def save_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
