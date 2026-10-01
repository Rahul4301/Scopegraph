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

# Per-question 0/1 outcomes: the only metrics McNemar is valid for. Averaged metrics such as
# recall_at_8 or the stale-memory rate can coincidentally be 0/1 and must not be tested.
BINARY_METRICS = frozenset({
    "exact_match", "gold_hit_at_8", "any_cross_scope_contamination",
    "official_llm_judge_accuracy", "official_substring_exact_match", "official_exact_match",
})
MIN_PILOT_ACCOUNTS = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]], *, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float] | None:
    """Percentile-bootstrap 95% CI resampling whole accounts, never single questions.

    Returns ``None`` for a single account: resampling one cluster can only reproduce
    the mean, so a degenerate ``mean == lower == upper`` interval would be misleading.
    """
    if not values_by_cluster:
        raise ValueError("Cannot bootstrap an empty sample")
    clusters = sorted(values_by_cluster)
    if len(clusters) == 1:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
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


def _latency_keys(values: list[ScoredRecord]) -> dict[str, float]:
    """Percentiles for each separately measured stage; stages are never summed or blended.

    ``retrieval_*`` is the timed retrieval call and includes embedding lookups made inside
    it. ``retrieval_core_*`` subtracts that embedding time. ``answer_*`` and ``judge_*`` are
    provider round trips. Latencies are only comparable within one dataset and protocol.
    """
    stages: dict[str, list[float]] = {
        "retrieval": [v.retrieval_latency_ms for v in values],
        "retrieval_embedding": [
            v.retrieval_embedding_ms for v in values if v.retrieval_embedding_ms is not None
        ],
        "retrieval_core": [
            v.retrieval_latency_ms - v.retrieval_embedding_ms
            for v in values if v.retrieval_embedding_ms is not None
        ],
        "embedding_preparation": [
            v.embedding_preparation_ms for v in values if v.embedding_preparation_ms is not None
        ],
        "answer": [v.answer_latency_ms for v in values if v.answer_latency_ms is not None],
        "judge": [v.judge_latency_ms for v in values if v.judge_latency_ms is not None],
    }
    keys: dict[str, float] = {}
    for stage, samples in stages.items():
        if samples:
            summary = latency_summary(samples)
            keys[f"{stage}_p50_ms"] = summary["p50"]
            keys[f"{stage}_p95_ms"] = summary["p95"]
    return keys


def score_record(record: EvaluationRecord, *, k: int = 8) -> ScoredRecord:
    """Attach the scored metrics for one raw record."""
    if record.failure_type is not None:
        metrics: dict[str, float | None] = {}
        if record.answer_evaluated:
            metrics["exact_match"] = 0.0
            metrics["token_f1"] = 0.0
        if record.official_metric is not None:
            metrics[f"official_{record.official_metric}"] = 0.0
        metrics.update(
            {
                f"official_{metric}": 0.0
                for metric in record.official_secondary_scores
            }
        )
        return ScoredRecord(**record.model_dump(), metrics=metrics)
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
        "gold_hit_at_8": float(source_recall > 0) if source_recall is not None else None,
        "any_cross_scope_contamination": float(any(
            set(scopes) - set(record.allowed_scope_ids or record.gold_scope_ids)
            for scopes in record.retrieved_origin_scope_ids[:k]
        )) if record.retrieved_origin_scope_ids else None,
    }
    if record.official_metric is not None:
        metrics[f"official_{record.official_metric}"] = record.official_score
    metrics.update(
        {
            f"official_{metric}": score
            for metric, score in record.official_secondary_scores.items()
        }
    )
    if record.question_type == "abstention":
        metrics[f"precision_at_{k}"] = None
        metrics[f"recall_at_{k}"] = None
        metrics["gold_hit_at_8"] = None
    if record.question_type in {"temporal_historical", "temporal_historical_state"}:
        metrics["stale_memory_error_rate"] = None
    return ScoredRecord(**record.model_dump(), metrics=metrics)


def load_jsonl(paths: Iterable[str | Path]) -> list[EvaluationRecord]:
    """Read EvaluationRecord lines from one or more JSONL files."""
    records: list[EvaluationRecord] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(EvaluationRecord.model_validate_json(line))
    return records


def aggregate_records(records: Iterable[EvaluationRecord]) -> dict[str, dict[str, float]]:
    """Summarize records per system/condition with account-clustered 95% intervals."""
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
                interval = _cluster_bootstrap_ci(dict(clusters))
                if interval is not None:
                    summary[system][f"{name}_ci95_low"], summary[system][f"{name}_ci95_high"] = (
                        interval
                    )
        summary[system]["account_count"] = float(len({value.scenario_id for value in values}))
        summary[system]["query_count"] = float(len(values))
        summary[system]["failure_count"] = float(
            sum(value.failure_type is not None for value in values)
        )
        summary[system]["failure_rate"] = summary[system]["failure_count"] / len(values)
        summary[system].update(_latency_keys(values))
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
    """Compare each ablation with full ScopeGraph on identical questions.

    Sign convention: every difference is ``full minus control``. For metrics where lower
    is better (contamination, stale-memory rate) a negative difference favours full.
    """
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
            values: dict[str, float] = {
                "paired_question_count": float(len(paired_keys)),
                "paired_account_count": float(len({key[0] for key in paired_keys})),
            }
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
                account_means = [sum(item) / len(item) for item in differences_by_account.values()]
                positive = sum(mean > 0 for mean in account_means)
                negative = sum(mean < 0 for mean in account_means)
                values[f"{metric}_accounts_positive"] = float(positive)
                values[f"{metric}_accounts_negative"] = float(negative)
                values[f"{metric}_account_sign_test_p"] = _exact_mcnemar_p(positive, negative)
                interval = _cluster_bootstrap_ci(dict(differences_by_account))
                if interval is not None:
                    values[f"{metric}_difference_ci95_low"] = interval[0]
                    values[f"{metric}_difference_ci95_high"] = interval[1]
                if metric in BINARY_METRICS and all(
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
    """Aggregate JSONL files and optionally write the summary JSON."""
    summary = aggregate_records(load_jsonl(paths))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


LOWER_IS_BETTER = frozenset({
    "cross_scope_contamination", "any_cross_scope_contamination", "stale_memory_error_rate",
})
SIGN_CONVENTION = (
    "Every difference is full minus control. For metrics where lower is better "
    "(cross_scope_contamination, any_cross_scope_contamination, stale_memory_error_rate) "
    "a negative difference favours full; "
    "for all other metrics a positive difference favours full."
)


def _favours(metric: str, low: float | None, high: float | None) -> str:
    """Which side the 95% interval supports, or why it supports neither."""
    if low is None or high is None:
        return "undetermined (single account: no interval)"
    if low <= 0.0 <= high:
        return "neither (interval includes 0)"
    full_higher = low > 0
    if metric in LOWER_IS_BETTER:
        return "control" if full_higher else "full"
    return "full" if full_higher else "control"


def _split_metric_keys(values: dict[str, float], suffix: str) -> dict[str, float]:
    return {key[: -len(suffix)]: value for key, value in values.items() if key.endswith(suffix)}


def confidence_interval_report(records: Iterable[EvaluationRecord]) -> dict[str, object]:
    """Structured CI output: per-system intervals and paired full-minus-control differences.

    Intervals are 95% percentile bootstrap intervals that resample whole accounts (CrossScopeMem
    scenarios or LoCoMo conversations). Anything under ``MIN_PILOT_ACCOUNTS`` accounts is
    flagged ``underpowered``; a single account yields no interval at all.
    """
    records = list(records)
    summary = aggregate_records(records)
    systems: dict[str, object] = {}
    for system, values in summary.items():
        means = {key: value for key, value in values.items() if _is_quality_metric(key)}
        accounts = int(values["account_count"])
        systems[system] = {
            "n_accounts": accounts,
            "n_questions": int(values["query_count"]),
            "underpowered": accounts < MIN_PILOT_ACCOUNTS,
            "metrics": {
                metric: {
                    "mean": mean,
                    "ci95_low": values.get(f"{metric}_ci95_low"),
                    "ci95_high": values.get(f"{metric}_ci95_high"),
                }
                for metric, mean in sorted(means.items())
            },
        }
    paired: dict[str, object] = {}
    for label, values in paired_ablation_comparisons(records).items():
        means = _split_metric_keys(values, "_mean_difference")
        accounts = int(values["paired_account_count"])
        entry: dict[str, object] = {
            "n_accounts": accounts,
            "n_pairs": int(values["paired_question_count"]),
            "underpowered": accounts < MIN_PILOT_ACCOUNTS,
            "metrics": {},
        }
        for metric, mean in sorted(means.items()):
            low = values.get(f"{metric}_difference_ci95_low")
            high = values.get(f"{metric}_difference_ci95_high")
            metrics = entry["metrics"]
            assert isinstance(metrics, dict)
            metrics[metric] = {
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "lower_is_better": metric in LOWER_IS_BETTER,
                "favours": _favours(metric, low, high),
                "accounts_positive": int(values[f"{metric}_accounts_positive"]),
                "accounts_negative": int(values[f"{metric}_accounts_negative"]),
                "account_sign_test_p": values[f"{metric}_account_sign_test_p"],
            }
        paired[label.replace("/full-minus-", "/")] = entry
    return {
        "protocol": {
            "resampling_unit": "account (CrossScopeMem scenario / LoCoMo conversation)",
            "method": "percentile bootstrap, whole-account resampling",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": 0.95,
            "min_pilot_accounts": MIN_PILOT_ACCOUNTS,
            "proposal_target_accounts": "40-60",
            "sign_convention": SIGN_CONVENTION,
        },
        "systems": systems,
        "paired_differences_full_minus_control": paired,
    }


def _is_quality_metric(key: str) -> bool:
    """True for scored quality metrics; false for counts, latencies, tokens and CI bounds."""
    if key.endswith(("_ci95_low", "_ci95_high")):
        return False
    return key in {"exact_match", "token_f1", "cross_scope_contamination",
                   "any_cross_scope_contamination", "stale_memory_error_rate",
                   "gold_hit_at_8"} or key.startswith(
        ("precision_at_", "recall_at_", "official_")
    )


def mcnemar_report(records: Iterable[EvaluationRecord]) -> list[dict[str, object]]:
    """Exact McNemar rows (full vs control) for every binary metric, one row per pair.

    McNemar treats questions as independent; questions inside one account are correlated,
    so read ``exact_p`` as descriptive and prefer the account-level sign test.
    """
    rows: list[dict[str, object]] = []
    for label, values in sorted(paired_ablation_comparisons(records).items()):
        for metric in sorted(_split_metric_keys(values, "_mcnemar_exact_p")):
            full_wins = int(values[f"{metric}_mcnemar_full_wins"])
            control_wins = int(values[f"{metric}_mcnemar_ablation_wins"])
            rows.append({
                "comparison": label,
                "metric": metric,
                "n_pairs": int(values["paired_question_count"]),
                "n_accounts": int(values["paired_account_count"]),
                "full_correct_control_wrong": full_wins,
                "control_correct_full_wrong": control_wins,
                "exact_p": values[f"{metric}_mcnemar_exact_p"],
                "accounts_full_higher": int(values[f"{metric}_accounts_positive"]),
                "accounts_control_higher": int(values[f"{metric}_accounts_negative"]),
                "account_sign_test_p": values[f"{metric}_account_sign_test_p"],
            })
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _gold_recall(gold: set[str], lists: list[list[str]]) -> float:
    return len(gold & {source for sources in lists for source in sources}) / len(gold)


def _interval(by_conversation: dict[str, list[float]]) -> dict[str, float | None]:
    values = [value for items in by_conversation.values() for value in items]
    interval = _cluster_bootstrap_ci(by_conversation) if by_conversation else None
    return {
        "mean": _mean(values),
        "ci95_low": interval[0] if interval else None,
        "ci95_high": interval[1] if interval else None,
        "n_conversations": float(len(by_conversation)),
        "n_questions": float(len(values)),
        "underpowered": float(len(by_conversation) < MIN_PILOT_ACCOUNTS),
    }


def _failure_stage(record: EvaluationRecord, correct: bool) -> str:
    """Where a LoCoMo question was lost, judged against the annotated gold evidence turns."""
    if correct:
        return "correct"
    gold = set(record.gold_source_ids)
    if not gold:
        return "wrong_no_gold_annotation"
    if not gold & {s for sources in record.retrieved_source_ids for s in sources}:
        return "wrong_gold_not_retrieved"
    if not gold & {s for sources in record.delivered_source_ids for s in sources}:
        return "wrong_gold_retrieved_not_delivered"
    return "wrong_gold_delivered"


def locomo_diagnostics(records: Iterable[EvaluationRecord]) -> dict[str, object]:
    """Per-category LoCoMo accuracy and gold-evidence recall, with and without category 5.

    Category 5 (adversarial) is graded by abstention, so its accuracy says nothing about
    retrieval; it is reported separately and excluded from the ``excluding_category_5``
    aggregates. Gold-evidence recall is the share of annotated evidence turns present in the
    retrieved (``retrieved_source_ids``) or delivered (``delivered_source_ids``) sources. It
    is a strict proxy: a question can be answered from other turns, so the report also gives
    accuracy when the gold turns were missed.
    """
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.dataset == "locomo":
            grouped[f"{record.system}/{record.ablation}"].append(record)
    output: dict[str, object] = {}
    for system, rows in sorted(grouped.items()):
        by_category: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for row in rows:
            by_category[str(row.benchmark_metadata.get("category", "unknown"))].append(row)

        def judged(row: EvaluationRecord) -> float | None:
            if row.official_metric != "llm_judge_accuracy":
                return None
            return 0.0 if row.failure_type is not None else row.official_score

        def f1(row: EvaluationRecord) -> float | None:
            return row.official_secondary_scores.get("locomo_f1")

        categories: dict[str, object] = {}
        for category, items in sorted(by_category.items()):
            annotated = [row for row in items if row.gold_source_ids]
            retrieved = [_gold_recall(set(r.gold_source_ids), r.retrieved_source_ids)
                         for r in annotated]
            delivered = [_gold_recall(set(r.gold_source_ids), r.delivered_source_ids)
                         for r in annotated]
            scores = [judged(row) for row in items]
            complete = [
                judged(r) for r in annotated
                if set(r.gold_source_ids) <= {s for ss in r.delivered_source_ids for s in ss}
            ]
            partial = [
                judged(r) for r in annotated
                if not set(r.gold_source_ids) <= {s for ss in r.delivered_source_ids for s in ss}
            ]
            hit = [judged(r) for r, rec in zip(annotated, retrieved, strict=True) if rec > 0]
            miss = [judged(r) for r, rec in zip(annotated, retrieved, strict=True) if rec == 0]
            stages: dict[str, int] = defaultdict(int)
            for row in items:
                score = judged(row)
                stages[_failure_stage(row, score is not None and score >= 1.0)] += 1
            categories[category] = {
                "n": len(items),
                "scored_by": "abstention rule" if category == "5" else "gpt-4o rubric judge",
                "judge_accuracy": _mean([s for s in scores if s is not None]),
                "locomo_f1": _mean([v for v in map(f1, items) if v is not None]),
                "no_gold_annotation": len(items) - len(annotated),
                "gold_source_recall_retrieved": _mean(retrieved),
                "gold_source_recall_delivered": _mean(delivered),
                "any_gold_retrieved_rate": _mean([float(r > 0) for r in retrieved]),
                "zero_gold_retrieved_count": sum(r == 0 for r in retrieved),
                "zero_gold_retrieved_rate": _mean([float(r == 0) for r in retrieved]),
                "all_gold_delivered_rate": (
                    len(complete) / len(annotated) if annotated else None
                ),
                "accuracy_when_all_gold_delivered": _mean([v for v in complete if v is not None]),
                "accuracy_when_gold_partly_or_not_delivered": _mean(
                    [v for v in partial if v is not None]
                ),
                "accuracy_when_gold_retrieved": _mean([v for v in hit if v is not None]),
                "accuracy_when_gold_missed": _mean([v for v in miss if v is not None]),
                "outcome_stages": dict(sorted(stages.items())),
            }

        def overall(rows_: list[EvaluationRecord], metric: str) -> dict[str, float | None]:
            clusters: dict[str, list[float]] = defaultdict(list)
            for row in rows_:
                value = judged(row) if metric == "judge" else f1(row)
                if value is not None:
                    clusters[row.scenario_id].append(value)
            return _interval(dict(clusters))

        without_5 = [row for row in rows if row.benchmark_metadata.get("category") != "5"]
        chain: dict[str, object] = {}
        category_1 = by_category.get("1", [])
        by_evidence: dict[int, list[float]] = defaultdict(list)
        for row in category_1:
            value = judged(row)
            if value is not None:
                by_evidence[len(row.gold_source_ids)].append(value)
        chain["category_1_accuracy_by_gold_evidence_count"] = {
            str(count): {"n": len(values), "judge_accuracy": _mean(values)}
            for count, values in sorted(by_evidence.items())
        }
        def gold_coverage(rows_: list[EvaluationRecord]) -> dict[str, float | None]:
            annotated_ = [row for row in rows_ if row.gold_source_ids]
            zero_retrieved = sum(
                _gold_recall(set(r.gold_source_ids), r.retrieved_source_ids) == 0
                for r in annotated_
            )
            zero_delivered = sum(
                _gold_recall(set(r.gold_source_ids), r.delivered_source_ids) == 0
                for r in annotated_
            )
            total = len(annotated_)
            return {
                "annotated_questions": float(total),
                "zero_gold_retrieved_count": float(zero_retrieved),
                "zero_gold_retrieved_rate": zero_retrieved / total if total else None,
                "zero_gold_delivered_count": float(zero_delivered),
                "zero_gold_delivered_rate": zero_delivered / total if total else None,
            }

        output[system] = {
            "n_questions": len(rows),
            "n_conversations": len({row.scenario_id for row in rows}),
            "gold_evidence_all_categories": gold_coverage(rows),
            "gold_evidence_excluding_category_5": gold_coverage(without_5),
            "all_categories": {
                "judge_accuracy": overall(rows, "judge"), "locomo_f1": overall(rows, "f1"),
            },
            "excluding_category_5": {
                "judge_accuracy": overall(without_5, "judge"),
                "locomo_f1": overall(without_5, "f1"),
            },
            "categories": categories,
            **chain,
        }
    return output
