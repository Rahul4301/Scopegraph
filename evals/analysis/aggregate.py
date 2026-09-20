"""Score raw JSONL independently from benchmark execution."""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from evals.analysis.statistics import bootstrap_mean_difference
from evals.metrics.answer_accuracy import exact_match, token_f1
from evals.metrics.latency import latency_summary
from evals.metrics.retrieval_precision import precision_at_k
from evals.metrics.retrieval_recall import recall_at_k
from evals.metrics.scope_contamination import cross_scope_contamination
from evals.metrics.stale_memory import stale_memory_error_rate
from evals.schemas import EvaluationRecord, ScoredRecord


def score_record(record: EvaluationRecord, *, k: int = 8) -> ScoredRecord:
    gold_sources = set(record.gold_source_ids)
    retrieved_sources = record.retrieved_source_ids[:k]
    source_recall = (
        len(gold_sources & {source for sources in retrieved_sources for source in sources})
        / len(gold_sources) if gold_sources else None
    )
    source_precision = (
        sum(bool(gold_sources & set(sources)) for sources in retrieved_sources)
        / len(retrieved_sources) if retrieved_sources else 0.0
    )
    metrics = {
        "exact_match": exact_match(record.answer, record.gold_answer)
        if record.answer_evaluated else None,
        "token_f1": token_f1(record.answer, record.gold_answer)
        if record.answer_evaluated else None,
        f"precision_at_{k}": source_precision if gold_sources else precision_at_k(
            record.retrieved_memory_ids, record.gold_memory_ids, k),
        f"recall_at_{k}": source_recall if gold_sources else recall_at_k(
            record.retrieved_memory_ids, record.gold_memory_ids, k),
        "cross_scope_contamination": (
            sum(bool(set(scopes) - set(record.allowed_scope_ids or record.gold_scope_ids))
                for scopes in record.retrieved_origin_scope_ids)
            / len(record.retrieved_origin_scope_ids)
        ) if record.retrieved_origin_scope_ids else cross_scope_contamination(
            [{"scope_id": scope_id} for scope_id in record.retrieved_scope_ids],
            record.allowed_scope_ids or record.gold_scope_ids,
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
    identities: set[tuple[str, str, str, str]] = set()
    protocols: set[tuple[str, str, str, str]] = set()
    for record in records:
        identity = (record.dataset, record.system, record.scenario_id, record.question_id)
        if identity in identities:
            raise ValueError(f"Duplicate evaluation question: {identity}; select one run batch")
        identities.add(identity)
        protocols.add((record.dataset, record.protocol_version,
                       record.evaluation_mode, record.config_hash))
        if len(protocols) > 1:
            raise ValueError("Cannot aggregate incompatible datasets, protocols, modes or configs")
        grouped[record.system].append(score_record(record))
    summary: dict[str, dict[str, float]] = {}
    for system, values in grouped.items():
        metric_names = sorted({name for value in values for name in value.metrics})
        summary[system] = {}
        for name in metric_names:
            measured = [value.metrics[name] for value in values
                        if value.metrics.get(name) is not None]
            if measured:
                summary[system][name] = sum(float(value) for value in measured) / len(measured)
        summary[system]["query_count"] = float(len(values))
        summary[system]["retrieval_p50_ms"] = latency_summary(
            value.retrieval_latency_ms for value in values
        )["p50"]
        summary[system]["retrieval_p95_ms"] = latency_summary(
            value.retrieval_latency_ms for value in values
        )["p95"]
        summary[system]["retrieved_tokens_mean"] = sum(
            value.retrieved_tokens for value in values
        ) / len(values)
        summary[system]["retrieved_tokens_p95"] = sorted(
            value.retrieved_tokens for value in values
        )[min(len(values) - 1, int(len(values) * 0.95))]
        measured_tokens = [value.output_tokens for value in values
                           if value.output_tokens is not None]
        if measured_tokens:
            summary[system]["answer_tokens_mean"] = sum(measured_tokens) / len(measured_tokens)
        storage = [value.storage_stats for value in values if value.storage_stats]
        if storage:
            summary[system]["logical_bytes_mean"] = sum(
                float(item.get("logical_bytes", 0)) for item in storage
            ) / len(storage)
            summary[system]["nodes_mean"] = sum(
                float(item.get("nodes", 0)) for item in storage
            ) / len(storage)
            summary[system]["edges_mean"] = sum(
                float(item.get("edges", 0)) for item in storage
            ) / len(storage)
    return summary


def aggregate_files(
    paths: Iterable[str | Path], output: str | Path | None = None
) -> dict[str, dict[str, float]]:
    summary = aggregate_records(load_jsonl(paths))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def confidence_intervals(
    records: Iterable[EvaluationRecord], *, reference_system: str = "scopegraph",
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return paired bootstrap intervals against a reference system."""
    scored = [score_record(record) for record in records]
    by_system = defaultdict(dict)
    for record in scored:
        key = (record.scenario_id, record.question_id)
        if key in by_system[record.system]:
            raise ValueError("Duplicate question in confidence interval inputs")
        by_system[record.system][key] = record
    reference = by_system.get(reference_system, {})
    result: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = sorted({name for record in scored for name in record.metrics})
    for system, values in by_system.items():
        if system == reference_system:
            continue
        if set(reference) != set(values):
            raise ValueError("Paired comparisons require identical question sets")
        keys = sorted(reference)
        system_result: dict[str, dict[str, float]] = {}
        for metric in metric_names:
            paired = [key for key in keys if reference[key].metrics.get(metric) is not None
                      and values[key].metrics.get(metric) is not None]
            if not paired:
                continue
            # Resample scenarios, not dependent questions from the same history.
            clusters = sorted({key[0] for key in paired})
            left, right = [], []
            for scenario in clusters:
                cluster = [key for key in paired if key[0] == scenario]
                left.append(sum(float(reference[key].metrics[metric]) for key in cluster)
                            / len(cluster))
                right.append(sum(float(values[key].metrics[metric]) for key in cluster)
                             / len(cluster))
            system_result[metric] = bootstrap_mean_difference(left, right, seed=seed)
        result[system] = system_result
    return result
