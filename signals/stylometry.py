import re
import statistics

from config import STYLO_VAR_WEIGHT, STYLO_PUNCT_WEIGHT, STYLO_TTR_WEIGHT

_PUNCT_CHARS = [",", ";", ":", ".", "!", "?", "-", "—", "(", ")", '"', "'", "..."]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def sentence_length_subscore(text: str) -> float:
    """Uniform sentence lengths read AI-ward (1.0); high variance human-ward (0.0)."""
    sentences = _sentences(text)
    if len(sentences) < 2:
        return 0.5  # edge case B: not enough data to judge
    lengths = [len(_words(s)) for s in sentences]
    std = statistics.pstdev(lengths)
    return _clamp((8.0 - std) / (8.0 - 2.0))


def type_token_ratio_subscore(text: str) -> float:
    """Lower vocabulary diversity nudges AI-ward. Lightly weighted (length-sensitive)."""
    words = _words(text)
    if not words:
        return 0.5
    ttr = len(set(words)) / len(words)
    return _clamp((0.8 - ttr) / (0.8 - 0.3))


def punctuation_subscore(text: str) -> float:
    """Few distinct punctuation types reads AI-ward (regular/sparse)."""
    present = {p for p in _PUNCT_CHARS if p in text}
    variety = len(present)
    return _clamp((5 - variety) / 5)


def stylometry_score(text: str) -> float:
    """Weighted average of three sub-scores. Returns P(AI) in [0,1]."""
    var = sentence_length_subscore(text)
    ttr = type_token_ratio_subscore(text)
    punct = punctuation_subscore(text)
    score = (
        STYLO_VAR_WEIGHT * var
        + STYLO_PUNCT_WEIGHT * punct
        + STYLO_TTR_WEIGHT * ttr
    )
    return _clamp(score)
