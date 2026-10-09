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


_LABEL = re.compile(r"label\s*:\s*(-?\d+)", re.IGNORECASE)
_INTEGER = re.compile(r"-?\d+")
_OPTION_FIELD = re.compile(r'"answer"\s*:\s*"\s*([A-D])\b')
_OPTION = re.compile(r"(?:^|\s|\()([A-D])[.):]")


def parse_label(prediction: str) -> str | None:
    """The numeric label of an in-context-learning answer: "label: 17", else the first integer."""
    match = _LABEL.search(prediction) or _INTEGER.search(prediction)
    return match.group(1 if match.re is _LABEL else 0) if match else None


def parse_option(text: str) -> str | None:
    """The option letter of a multiple-choice answer: the JSON "answer" field, else the first
    "C." / "(C)" style letter."""
    match = _OPTION_FIELD.search(text) or _OPTION.search(text.strip())
    return match.group(1) if match else None


def native_match(task: str, prediction: str, golds: list[str]) -> bool | None:
    """MemoryAgentBench's own pass test for each source. icl: the parsed label equals the gold
    label (substring would let gold 1 match label 17). detective_qa: same option letter. eventqa,
    ruler_qa and factconsolidation: the accepted answer appears in the prediction. None for
    infbench_sum, which is graded on its summary scores."""
    if task == "icl_":
        label = parse_label(prediction)
        return label is not None and any(label == g.strip() for g in golds)
    if task == "detective_qa":
        letter = parse_option(prediction)
        return letter is not None and any(letter == parse_option(g) for g in golds)
    if task == "infbench_sum":
        return None
    return substring_match(prediction, golds)


def answer_metrics(
    prediction: str, context: str, golds: list[str], task: str | None = None
) -> dict[str, float | bool | None]:
    metrics: dict[str, float | bool | None] = {
        "exact_match": exact_match(prediction, golds),
        "substring_match": substring_match(prediction, golds),
        "token_f1": token_f1(prediction, golds),
        "gold_in_context": gold_in_context(context, golds),
    }
    if task is not None:
        metrics["native_match"] = native_match(task, prediction, golds)
    return metrics


def gold_in_store(store_text: str, golds: list[str]) -> bool:
    """Extraction proxy: an accepted answer appears verbatim (after normalization) in the text of
    everything a memory system stored for the history. Extracted facts paraphrase the source, so
    this under-counts storage for extraction systems; PREREG.md Section 6 sizes that bias."""
    return substring_match(store_text, golds)
