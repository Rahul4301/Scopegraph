"""Score raw JSONL independently from benchmark execution."""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from evals.metrics.answer_accuracy import exact_match, token_f1
from evals.metrics.latency import latency_summary
from evals.metrics.retrieval_precision import precision_at_k
from evals.metrics.retrieval_recall import recall_at_k
from evals.metrics.scope_contamination import (
    cross_scope_contamination,
    scope_classification_accuracy,
)
from evals.metrics.stale_memory import stale_memory_error_rate
from evals.schemas import EvaluationRecord, ScoredRecord


def score_record(record: EvaluationRecord, *, k: int = 8) -> ScoredRecord:
    metrics = {
        "exact_match": exact_match(record.answer, record.gold_answer),
        "token_f1": token_f1(record.answer, record.gold_answer),
        f"precision_at_{k}": precision_at_k(record.retrieved_memory_ids, record.gold_memory_ids, k),
        f"recall_at_{k}": recall_at_k(record.retrieved_memory_ids, record.gold_memory_ids, k),
        "cross_scope_contamination": cross_scope_contamination(
            [{"scope_id": scope_id} for scope_id in record.retrieved_scope_ids],
            record.gold_scope_ids,
        ),
        "scope_classification_accuracy": scope_classification_accuracy(
            record.current_scope_id, record.gold_scope_ids
        ),
        "stale_memory_error_rate": stale_memory_error_rate(
            [{"status": status} for status in record.retrieved_statuses]
        ),
    }
    return ScoredRecord(**record.model_dump(), metrics=metrics)


def load_jsonl(paths: Iterable[str | Path]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(EvaluationRecord.model_validate_json(line))
    return records


def aggregate_records(records: Iterable[EvaluationRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[ScoredRecord]] = defaultdict(list)
    for record in records:
        grouped[record.system].append(score_record(record))
    summary: dict[str, dict[str, float]] = {}
    for system, values in grouped.items():
        metric_names = sorted({name for value in values for name in value.metrics})
        summary[system] = {
            name: sum(float(value.metrics.get(name) or 0.0) for value in values) / len(values)
            for name in metric_names
        }
        summary[system]["query_count"] = float(len(values))
        summary[system]["retrieval_p50_ms"] = latency_summary(
            value.retrieval_latency_ms for value in values
        )["p50"]
        summary[system]["retrieval_p95_ms"] = latency_summary(
            value.retrieval_latency_ms for value in values
        )["p95"]
    return summary


def aggregate_files(
    paths: Iterable[str | Path], output: str | Path | None = None
) -> dict[str, dict[str, float]]:
    summary = aggregate_records(load_jsonl(paths))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
