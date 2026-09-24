"""Official deterministic scoring rules for the selected benchmark releases."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import editdistance
from nltk.stem import PorterStemmer

from evals.schemas import ExternalBenchmarkExample

_STEMMER = PorterStemmer()


def _locomo_normalize(text: str) -> str:
    text = text.replace(",", "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _locomo_f1(prediction: str, truth: str) -> float:
    predicted = [_STEMMER.stem(word) for word in _locomo_normalize(prediction).split()]
    expected = [_STEMMER.stem(word) for word in _locomo_normalize(truth).split()]
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def locomo_score(prediction: str, answer: str, category: str) -> float:
    """Mirror `task_eval/evaluation.py::eval_question_answering`."""
    if category in {"2", "3", "4"}:
        expected = answer.split(";", maxsplit=1)[0].strip() if category == "3" else answer
        return _locomo_f1(prediction, expected)
    if category == "1":
        predictions = [item.strip() for item in prediction.split(",")]
        truths = [item.strip() for item in answer.split(",")]
        return sum(max(_locomo_f1(item, truth) for item in predictions) for truth in truths) / len(
            truths
        )
    if category == "5":
        lowered = prediction.lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    raise ValueError(f"Unknown LoCoMo category: {category}")


def _mab_normalize(text: str) -> str:
    lowered = text.lower()
    unpunctuated = "".join(
        character for character in lowered if character not in string.punctuation
    )
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", unpunctuated).split())


def _flatten_answers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [
            str(item)
            for group in value
            for item in (group if isinstance(group, list) else [group])
        ]
    return [str(value)]


def _parsed_output(output: str) -> str:
    match = re.search(r"(?:Answer:)(.*)(?:\n|$)", output, flags=re.IGNORECASE)
    candidate = match.group(1).strip() if match else output.splitlines()[0].strip()
    return re.sub(r"^Answer:", "", candidate, flags=re.IGNORECASE).strip()


def memoryagentbench_score(
    prediction: str,
    answers: list[str],
    source: str,
    *,
    entity_map_path: Path | None = None,
) -> tuple[str, float] | None:
    """Return the benchmark's primary deterministic metric, or None for LLM-judged tasks."""
    if "longmemeval" in source or "infbench" in source:
        return None
    parsed = _parsed_output(prediction)
    flattened = _flatten_answers(answers)
    if "recsys" in source:
        if entity_map_path is None:
            raise ValueError("MemoryAgentBench RecSys scoring requires entity2id.json")
        return "recsys_recall@5", _recsys_recall_at_5(prediction, flattened, entity_map_path)
    normalized_prediction = _mab_normalize(parsed)
    if "ruler_niah" in source:
        score = sum(answer.lower() in prediction.lower() for answer in flattened) / len(flattened)
        return "ruler_recall", score
    if "eventqa" in source or "ruler" in source or "factconsolidation" in source:
        score = float(any(_mab_normalize(answer) in normalized_prediction for answer in flattened))
        return "substring_exact_match", score
    score = float(any(normalized_prediction == _mab_normalize(answer) for answer in flattened))
    return "exact_match", score


def _movie_name(raw: str) -> str:
    filename = raw.split("/")[-1].replace("_", " ").replace("-", " ").replace(">", " ")
    return " ".join(re.sub(r"\([^()]*\)", "", filename).split())


def _recsys_recall_at_5(prediction: str, answers: list[str], mapping_path: Path) -> float:
    mapping = json.loads(mapping_path.read_text())
    id_to_name = {int(entity_id): _movie_name(name) for name, entity_id in mapping.items()}
    try:
        _, recommendation_text = prediction.split("1.", maxsplit=1)
    except ValueError:
        recommendation_text = prediction.replace(",", "\n")
    raw_items = [
        re.sub(r"^(?:\d+[.、)]?\s*[-—–]?\s*)?", "", re.sub(r"\([^()]*\)", "", item)).strip()
        for item in recommendation_text.splitlines()
    ]
    candidates = list(set(id_to_name.values()))
    predicted = [
        min(candidates, key=lambda candidate: editdistance.eval(item.lower(), candidate.lower()))
        for item in raw_items
    ]
    truth_ids = [int(item.strip()) for answer in answers for item in answer.split(",")]
    truths = [id_to_name[item] for item in truth_ids]
    return sum(movie in predicted[:5] for movie in truths) / len(truths)


def official_score(
    dataset: str,
    prediction: str | None,
    example: ExternalBenchmarkExample,
    *,
    data_path: Path,
) -> tuple[str, float] | None:
    if prediction is None:
        return None
    if dataset == "locomo":
        return "locomo_f1", locomo_score(
            prediction, example.answer, str(example.metadata["category"])
        )
    if dataset == "memoryagentbench":
        source = str(example.metadata["source"])
        answers = [str(item) for item in example.metadata.get("answers", [example.answer])]
        mapping = (
            data_path / "entity2id.json"
            if data_path.is_dir()
            else data_path.parent / "entity2id.json"
        )
        return memoryagentbench_score(prediction, answers, source, entity_map_path=mapping)
    return None
