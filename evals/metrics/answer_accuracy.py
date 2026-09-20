"""Deterministic answer scoring helpers."""

import re


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def exact_match(answer: str | None, gold: str) -> float:
    return float(answer is not None and normalize_answer(answer) == normalize_answer(gold))


def token_f1(answer: str | None, gold: str) -> float:
    if not answer:
        return 0.0
    predicted = normalize_answer(answer).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    predicted_counts = {token: predicted.count(token) for token in set(predicted)}
    expected_counts = {token: expected.count(token) for token in set(expected)}
    overlap = sum(min(count, expected_counts.get(token, 0))
                  for token, count in predicted_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)
