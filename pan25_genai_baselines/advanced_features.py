import bz2
import lzma
import re
import zlib

import numpy as np
import pyppmd

__all__ = ["extract_features"]


def get_ppmd_len(text_bytes: bytes) -> int:
    return len(pyppmd.compress(text_bytes))


def get_zlib_len(text_bytes: bytes) -> int:
    return len(zlib.compress(text_bytes))


def get_bz2_len(text_bytes: bytes) -> int:
    return len(bz2.compress(text_bytes))


def get_lzma_len(text_bytes: bytes) -> int:
    return len(lzma.compress(text_bytes))


def _split_text_halves(text: str) -> tuple[bytes, bytes]:
    words = text.split()

    # Fallback for very short texts where word split is unreliable.
    if len(words) < 2:
        mid = len(text) // 2
        return text[:mid].encode("utf-8"), text[mid:].encode("utf-8")

    mid = len(words) // 2
    return " ".join(words[:mid]).encode("utf-8"), " ".join(words[mid:]).encode("utf-8")


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(max(den, 1e-8))


def _token_repetition_rate(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]
    unique = len(set(ngrams))
    return 1.0 - _safe_ratio(unique, len(ngrams))


def extract_features(text: str) -> list[float]:
    """Extract an enriched compression and text-statistics feature vector for one text."""
    if not text.strip():
        return [0.0] * 18

    text_bytes = text.encode("utf-8")
    half1, half2 = _split_text_halves(text)
    words = text.split()

    c_h1_p = get_ppmd_len(half1)
    c_h2_p = get_ppmd_len(half2)
    c_all_p = get_ppmd_len(text_bytes)

    eps = 1e-8
    ppmd_cosine = (c_h1_p + c_h2_p - c_all_p) / np.sqrt(max(c_h1_p * c_h2_p, eps))
    ppmd_ncd = (c_all_p - min(c_h1_p, c_h2_p)) / max(max(c_h1_p, c_h2_p), 1)
    ppmd_ratio = c_all_p / max(len(text_bytes), 1)

    c_h1_z = get_zlib_len(half1)
    c_h2_z = get_zlib_len(half2)
    c_all_z = get_zlib_len(text_bytes)

    zlib_ncd = (c_all_z - min(c_h1_z, c_h2_z)) / max(max(c_h1_z, c_h2_z), 1)
    zlib_ratio = c_all_z / max(len(text_bytes), 1)

    c_h1_b = get_bz2_len(half1)
    c_h2_b = get_bz2_len(half2)
    c_all_b = get_bz2_len(text_bytes)
    bz2_ncd = (c_all_b - min(c_h1_b, c_h2_b)) / max(max(c_h1_b, c_h2_b), 1)
    bz2_ratio = c_all_b / max(len(text_bytes), 1)

    c_h1_l = get_lzma_len(half1)
    c_h2_l = get_lzma_len(half2)
    c_all_l = get_lzma_len(text_bytes)
    lzma_ncd = (c_all_l - min(c_h1_l, c_h2_l)) / max(max(c_h1_l, c_h2_l), 1)
    lzma_ratio = c_all_l / max(len(text_bytes), 1)

    word_count = float(len(words))

    sent_split = [s.strip() for s in re.split(r"[.!?。！？]+", text) if s.strip()]
    sent_lengths = [len(s.split()) for s in sent_split]
    mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sent_len = float(np.std(sent_lengths)) if sent_lengths else 0.0

    unique_word_ratio = _safe_ratio(len(set(words)), len(words))
    punct_count = float(sum(text.count(ch) for ch in ",.;:!?，。；：！？"))
    punct_ratio = _safe_ratio(punct_count, len(text))
    digit_ratio = _safe_ratio(sum(ch.isdigit() for ch in text), len(text))
    upper_ratio = _safe_ratio(sum(ch.isupper() for ch in text), len(text))
    repeat_bigram = _token_repetition_rate(words, 2)
    repeat_trigram = _token_repetition_rate(words, 3)

    return [
        # Keep the original 6 baseline features first for backward compatibility.
        float(ppmd_cosine),
        float(ppmd_ncd),
        float(ppmd_ratio),
        float(zlib_ncd),
        float(zlib_ratio),
        word_count,
        float(bz2_ncd),
        float(bz2_ratio),
        float(lzma_ncd),
        float(lzma_ratio),
        float(mean_sent_len),
        float(std_sent_len),
        float(unique_word_ratio),
        float(punct_ratio),
        float(digit_ratio),
        float(upper_ratio),
        float(repeat_bigram),
        float(repeat_trigram),
    ]