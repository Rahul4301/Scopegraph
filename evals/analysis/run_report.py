"""Create processed JSON, Markdown, and SVG reports from raw JSONL."""

import argparse
import json
from pathlib import Path
from typing import Any

from evals.analysis.aggregate import (
    aggregate_records,
    confidence_interval_report,
    load_jsonl,
    locomo_diagnostics,
    mcnemar_report,
    paired_ablation_comparisons,
)
from evals.analysis.scope_classification import write_scope_classification_report
from evals.analysis.tables import (
    rows_table,
    write_markdown_table,
    write_readable_report,
    write_text,
)
from evals.runners.checkpoint import save_json
from evals.schemas import EvaluationRecord, ScopeClassificationEvaluation

MCNEMAR_COLUMNS = (
    "comparison", "metric", "n_pairs", "n_accounts", "full_correct_control_wrong",
    "control_correct_full_wrong", "exact_p", "accounts_full_higher",
    "accounts_control_higher", "account_sign_test_p",
)


def _latency_by_dataset(records: list[EvaluationRecord]) -> dict[str, dict[str, Any]]:
    """Latency summaries kept per dataset and protocol; they are never blended together."""
    result: dict[str, dict[str, Any]] = {}
    for dataset in sorted({record.dataset for record in records}):
        subset = [record for record in records if record.dataset == dataset]
        protocols = sorted({record.latency_protocol for record in subset})
        summary = aggregate_records(subset)
        result[dataset] = {
            "latency_protocols": protocols,
            "systems": {
                system: {k: v for k, v in values.items() if k.endswith("_ms")}
                for system, values in summary.items()
            },
        }
    return result


def _locomo_gold_text(path: Path) -> dict[str, str]:
    """Map ``<sample_id>:<dia_id>`` to the turn text, to explain missed gold evidence."""
    texts: dict[str, str] = {}
    for sample in json.loads(path.read_text()):
        for turns in sample["conversation"].values():
            if isinstance(turns, list):
                for turn in turns:
                    texts[f"{sample['sample_id']}:{turn['dia_id']}"] = turn.get("text", "")
    return texts


def _zero_gold_questions(
    records: list[EvaluationRecord], gold_text: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        gold = set(record.gold_source_ids)
        if record.dataset != "locomo" or not gold:
            continue
        if gold & {source for sources in record.retrieved_source_ids for source in sources}:
            continue
        rows.append({
            "system": f"{record.system}/{record.ablation}",
            "question_id": record.question_id,
            "category": record.benchmark_metadata.get("category"),
            "question": record.question,
            "gold_answer": record.gold_answer,
            "answer": record.answer,
            "judge_score": record.official_score,
            "gold_source_ids": sorted(gold),
            "gold_turn_text": {source: gold_text.get(source) for source in sorted(gold)},
        })
    return rows


def main() -> None:
    """CLI: score raw JSONL and write processed JSON, tables and report.md under --output-root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--locomo-data", type=Path, help="locomo10.json, to attach gold turn text")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing report (existing results are otherwise immutable)",
    )
    args = parser.parse_args()
    root: Path = args.output_root
    if (root / "processed" / "summary.json").exists() and not args.overwrite:
        parser.error(f"{root} already holds a report; choose a new --output-root or --overwrite")
    records = load_jsonl(args.inputs)
    summary = aggregate_records(records)
    save_json(root / "processed" / "summary.json", summary)
    save_json(
        root / "processed" / "summary_by_question_type.json",
        {
            question_type: aggregate_records(
                record for record in records if record.question_type == question_type
            )
            for question_type in sorted({record.question_type for record in records})
        },
    )
    save_json(root / "processed" / "paired_ablation_comparisons.json",
              paired_ablation_comparisons(records))
    save_json(root / "processed" / "confidence_intervals.json",
              confidence_interval_report(records))
    mcnemar = mcnemar_report(records)
    save_json(root / "processed" / "mcnemar.json", mcnemar)
    write_text(
        root / "tables" / "mcnemar.md",
        "Exact McNemar (full vs control, binary metrics). Differences are full minus control; "
        "questions inside one account are correlated, so prefer the account sign test.\n\n"
        + (rows_table(mcnemar, MCNEMAR_COLUMNS) if mcnemar else "No binary paired metrics.\n"),
    )
    latency = _latency_by_dataset(records)
    save_json(root / "processed" / "latency.json", latency)
    for dataset, block in latency.items():
        write_text(
            root / "tables" / f"latency_{dataset}.md",
            f"Latency for dataset `{dataset}` under protocol(s) {block['latency_protocols']}. "
            "Retrieval includes embedding lookups made inside it; retrieval_core excludes them; "
            "answer and judge are provider round trips. Do not compare across datasets.\n\n"
            + _latency_markdown(block["systems"]),
        )
    write_markdown_table(summary, root / "tables" / "summary.md", exclude_suffixes=("_ms",))
    if any(record.dataset == "locomo" for record in records):
        save_json(root / "processed" / "locomo_diagnostics.json", locomo_diagnostics(records))
        gold_text = _locomo_gold_text(args.locomo_data) if args.locomo_data else {}
        save_json(root / "processed" / "locomo_zero_gold_questions.json",
                  _zero_gold_questions(records, gold_text))
    classification_candidates = []
    if len({path.parent for path in args.inputs}) == 1:
        classification_candidates.append(args.inputs[0].parent / "classification.json")
    if len(args.inputs) == 1:
        classification_candidates.append(args.inputs[0].with_suffix(".classification.json"))
    classification_path = next(
        (path for path in classification_candidates if path.exists()), None
    )
    if classification_path is not None:
        write_scope_classification_report(
            ScopeClassificationEvaluation.model_validate_json(classification_path.read_text()),
            root,
        )
    write_readable_report(args.inputs, root)
    print(root / "report.md")


def _latency_markdown(systems: dict[str, dict[str, float]]) -> str:
    stages = sorted({key for values in systems.values() for key in values})
    rows = [{"system": system, **values} for system, values in systems.items()]
    return rows_table(rows, ["system", *stages])


if __name__ == "__main__":
    main()
