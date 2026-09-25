"""Score raw JSONL independently from benchmark execution."""

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from evals.metrics.answer_accuracy import exact_match, token_f1
from evals.metrics.latency import latency_summary
from evals.metrics.retrieval_precision import precision_at_k
from evals.metrics.retrieval_recall import recall_at_k
from evals.metrics.scope_contamination import cross_scope_contamination
from evals.metrics.stale_memory import stale_memory_error_rate
from evals.schemas import EvaluationRecord, ScoredRecord


def _cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]], *, samples: int = 10_000
) -> tuple[float, float]:
    """Bootstrap whole histories/accounts so related questions stay together."""
    if not values_by_cluster:
        raise ValueError("Cannot bootstrap an empty sample")
    clusters = sorted(values_by_cluster)
    values = [value for cluster in clusters for value in values_by_cluster[cluster]]
    if len(clusters) == 1:
        mean = sum(values) / len(values)
        return mean, mean
    generator = random.Random(42)
    size = len(clusters)
    means: list[float] = []
    for _ in range(samples):
        sampled = [
            value
            for _ in range(size)
            for value in values_by_cluster[clusters[generator.randrange(size)]]
        ]
        means.append(sum(sampled) / len(sampled))
    means.sort()
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


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
    if record.official_metric is not None:
        metrics[f"official_{record.official_metric}"] = record.official_score
    if record.question_type == "abstention":
        metrics[f"precision_at_{k}"] = None
        metrics[f"recall_at_{k}"] = None
    if record.question_type in {"temporal_historical", "temporal_historical_state"}:
        metrics["stale_memory_error_rate"] = None
    return ScoredRecord(**record.model_dump(), metrics=metrics)


def load_jsonl(paths: Iterable[str | Path]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(EvaluationRecord.model_validate_json(line))
    return records


def aggregate_records(records: Iterable[EvaluationRecord]) -> dict[str, dict[str, float]]:
    records = list(records)
    multiple_datasets = len({record.dataset for record in records}) > 1
    grouped: dict[str, list[ScoredRecord]] = defaultdict(list)
    identities: set[tuple[str, str, str, str]] = set()
    protocols: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for record in records:
        identity = (
            record.dataset,
            f"{record.system}/{record.ablation}",
            record.scenario_id,
            record.question_id,
        )
        if identity in identities:
            raise ValueError(f"Duplicate evaluation question: {identity}; select one run batch")
        identities.add(identity)
        protocol_group = (record.dataset, record.system, record.ablation)
        protocols[protocol_group].add(
            (record.protocol_version, record.evaluation_mode, record.config_hash)
        )
        if len(protocols[protocol_group]) > 1:
            raise ValueError("Cannot aggregate incompatible datasets, protocols, modes or configs")
        system_name = (
            record.system
            if record.ablation == "full"
            else f"{record.system}/{record.ablation}"
        )
        group_name = f"{record.dataset}/{system_name}" if multiple_datasets else system_name
        grouped[group_name].append(score_record(record))
    summary: dict[str, dict[str, float]] = {}
    for system, values in grouped.items():
        metric_names = sorted({name for value in values for name in value.metrics})
        summary[system] = {}
        for name in metric_names:
            measured_records = [
                value for value in values if value.metrics.get(name) is not None
            ]
            measured = [value.metrics[name] for value in measured_records]
            if measured:
                numeric = [float(value) for value in measured]
                summary[system][name] = sum(numeric) / len(numeric)
                clusters: dict[str, list[float]] = defaultdict(list)
                for value in measured_records:
                    metric = value.metrics[name]
                    assert metric is not None
                    clusters[value.scenario_id].append(float(metric))
                lower, upper = _cluster_bootstrap_ci(dict(clusters))
                summary[system][f"{name}_ci95_low"] = lower
                summary[system][f"{name}_ci95_high"] = upper
        summary[system]["query_count"] = float(len(values))
        summary[system]["failure_count"] = float(
            sum(value.failure_type is not None for value in values)
        )
        summary[system]["failure_rate"] = summary[system]["failure_count"] / len(values)
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
            summary[system]["answer_tokens_total"] = float(sum(measured_tokens))
        prompt_tokens = [value.input_tokens for value in values if value.input_tokens is not None]
        if prompt_tokens:
            summary[system]["answer_input_tokens_total"] = float(sum(prompt_tokens))
        judge_tokens = [
            (value.judge_input_tokens or 0) + (value.judge_output_tokens or 0)
            for value in values
            if value.judge_input_tokens is not None or value.judge_output_tokens is not None
        ]
        if judge_tokens:
            summary[system]["judge_tokens_total"] = float(sum(judge_tokens))
        usage_keys = sorted({key for value in values for key in value.token_usage})
        for key in usage_keys:
            summary[system][f"{key}_total"] = float(
                sum(value.token_usage.get(key, 0) for value in values)
            )
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


def _exact_mcnemar_p(full_wins: int, ablation_wins: int) -> float:
    discordant = full_wins + ablation_wins
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * 0.5**discordant
        for index in range(min(full_wins, ablation_wins) + 1)
    )
    return min(1.0, 2.0 * tail)


def paired_ablation_comparisons(
    records: Iterable[EvaluationRecord],
) -> dict[str, dict[str, float]]:
    """Compare each ablation with full ScopeGraph on identical questions."""
    scored = [score_record(record) for record in records]
    grouped: dict[tuple[str, str, str], dict[tuple[str, str], ScoredRecord]] = defaultdict(dict)
    for record in scored:
        grouped[(record.dataset, record.system, record.ablation)][
            (record.scenario_id, record.question_id)
        ] = record
    output: dict[str, dict[str, float]] = {}
    architectures = sorted({(dataset, system) for dataset, system, _ in grouped})
    for dataset, system in architectures:
        full = grouped.get((dataset, system, "full"), {})
        if not full:
            continue
        ablations = sorted(
            ablation
            for ds, candidate_system, ablation in grouped
            if ds == dataset and candidate_system == system and ablation != "full"
        )
        for ablation in ablations:
            comparison = grouped[(dataset, system, ablation)]
            paired_keys = sorted(set(full) & set(comparison))
            label = f"{dataset}/{system}/full-minus-{ablation}"
            values: dict[str, float] = {"paired_question_count": float(len(paired_keys))}
            metric_names = sorted(
                {
                    metric
                    for key in paired_keys
                    for metric in set(full[key].metrics) & set(comparison[key].metrics)
                }
            )
            for metric in metric_names:
                pairs = [
                    (key, full[key].metrics.get(metric), comparison[key].metrics.get(metric))
                    for key in paired_keys
                ]
                measured = [
                    (key, float(full_value), float(ablation_value))
                    for key, full_value, ablation_value in pairs
                    if full_value is not None and ablation_value is not None
                ]
                if not measured:
                    continue
                differences_by_account: dict[str, list[float]] = defaultdict(list)
                for (scenario_id, _), full_value, ablation_value in measured:
                    differences_by_account[scenario_id].append(full_value - ablation_value)
                differences = [
                    difference
                    for account in differences_by_account.values()
                    for difference in account
                ]
                values[f"{metric}_mean_difference"] = sum(differences) / len(differences)
                lower, upper = _cluster_bootstrap_ci(dict(differences_by_account))
                values[f"{metric}_difference_ci95_low"] = lower
                values[f"{metric}_difference_ci95_high"] = upper
                if all(
                    full_value in {0.0, 1.0} and ablation_value in {0.0, 1.0}
                    for _, full_value, ablation_value in measured
                ):
                    full_wins = sum(
                        full_value == 1.0 and ablation_value == 0.0
                        for _, full_value, ablation_value in measured
                    )
                    ablation_wins = sum(
                        full_value == 0.0 and ablation_value == 1.0
                        for _, full_value, ablation_value in measured
                    )
                    values[f"{metric}_mcnemar_full_wins"] = float(full_wins)
                    values[f"{metric}_mcnemar_ablation_wins"] = float(ablation_wins)
                    values[f"{metric}_mcnemar_exact_p"] = _exact_mcnemar_p(
                        full_wins, ablation_wins
                    )
            output[label] = values
    return output
def aggregate_files(
    paths: Iterable[str | Path], output: str | Path | None = None
) -> dict[str, dict[str, float]]:
    summary = aggregate_records(load_jsonl(paths))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
