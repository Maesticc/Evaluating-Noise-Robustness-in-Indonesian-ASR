"""Text normalization for WER/CER, applied to references and hypotheses only when scoring.
The transcripts in the manifests are never changed."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

_UNITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
          "sepuluh", "sebelas"]
_SCALES = [(10**12, "triliun"), (10**9, "miliar"), (10**6, "juta"), (1000, "ribu")]


def number_to_words(n: int) -> str:
    """Convert a non-negative integer to Indonesian words (e.g. 2018 -> 'dua ribu delapan belas')."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 12:
        return _UNITS[n]
    if n < 20:
        return f"{_UNITS[n - 10]} belas"
    if n < 100:
        tens, rest = divmod(n, 10)
        return f"{_UNITS[tens]} puluh" + (f" {_UNITS[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "seratus" if hundreds == 1 else f"{_UNITS[hundreds]} ratus"
        return head + (f" {number_to_words(rest)}" if rest else "")
    for value, name in _SCALES:
        if n >= value:
            count, rest = divmod(n, value)
            head = "seribu" if (value == 1000 and count == 1) else f"{number_to_words(count)} {name}"
            return head + (f" {number_to_words(rest)}" if rest else "")
    raise ValueError(n)


def _expand_number(match: re.Match) -> str:
    token = match.group(0)
    if "," in token or "." in token:
        # Indonesian uses '.' as thousands separator and ',' as decimal separator.
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", token):
            return number_to_words(int(token.replace(".", "")))
        integer, _, frac = token.replace(".", ",").partition(",")
        words = number_to_words(int(integer)) if integer else "nol"
        return f"{words} koma " + " ".join(_UNITS[int(d)] for d in frac)
    return number_to_words(int(token))


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_text(text: str, cfg: dict | None = None) -> str:
    """NFKC -> lowercase -> '%' to 'persen' -> digits to words -> hyphens to spaces -> strip accents
    -> remove punctuation -> collapse spaces. Each step can be switched off in config.yaml."""
    opts = {"lowercase": True, "strip_accents": True, "hyphen_to_space": True,
            "remove_punctuation": True, "numbers_to_words": True, "percent_to_word": True}
    if cfg:
        opts.update(cfg.get("text_normalization", {}))

    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    if opts["lowercase"]:
        t = t.lower()
    if opts["percent_to_word"]:
        t = re.sub(r"\s*%", " persen", t)
    if opts["numbers_to_words"]:
        t = _NUMBER_RE.sub(lambda m: f" {_expand_number(m)} ", t)
    if opts["hyphen_to_space"]:
        t = re.sub(r"[-\u2010-\u2015]", " ", t)
    if opts["strip_accents"]:
        t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    if opts["remove_punctuation"]:
        t = t.replace("'", "")  # e.g. "jum'at" -> "jumat"
        t = _PUNCT_RE.sub(" ", t).replace("_", " ")
    return _SPACE_RE.sub(" ", t).strip()
