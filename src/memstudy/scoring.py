"""Deterministic answer metrics, computed beside the judge verdict and never instead of it.

Normalization follows the usual QA convention (lowercase, no punctuation including curly quotes
and dashes, no articles, collapsed whitespace). Every accepted answer of an item is tried and the
best score is kept.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def _is_punct(ch: str) -> bool:
    return ch in _PUNCT or unicodedata.category(ch).startswith("P")  # also curly quotes, dashes


def normalize(text: str) -> str:
    text = "".join(ch for ch in text.lower() if not _is_punct(ch))
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, golds: list[str]) -> bool:
    pred = normalize(prediction)
    return any(pred == normalize(g) for g in golds)


def substring_match(prediction: str, golds: list[str]) -> bool:
    """True when a normalized accepted answer appears inside the normalized prediction."""
    pred = normalize(prediction)
    return any(normalize(g) and normalize(g) in pred for g in golds)


def _f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = normalize(prediction).split(), normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: list[str]) -> float:
    return max((_f1(prediction, g) for g in golds), default=0.0)


def gold_in_context(context: str, golds: list[str]) -> bool:
    """Retrieval proxy: an accepted answer appears verbatim in the context the reader saw. Short
    answers can match by chance, so read it as a rough signal, not as recall."""
    return substring_match(context, golds)


def answer_metrics(prediction: str, context: str, golds: list[str]) -> dict[str, float | bool]:
    return {
        "exact_match": exact_match(prediction, golds),
        "substring_match": substring_match(prediction, golds),
        "token_f1": token_f1(prediction, golds),
        "gold_in_context": gold_in_context(context, golds),
    }


def gold_in_store(store_text: str, golds: list[str]) -> bool:
    """Extraction proxy: an accepted answer appears verbatim (after normalization) in the text of
    everything a memory system stored for the history. Extracted facts paraphrase the source, so
    this under-counts storage for extraction systems; PREREG.md Section 6 sizes that bias."""
    return substring_match(store_text, golds)
