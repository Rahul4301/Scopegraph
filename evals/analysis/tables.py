"""Markdown tables for reports, plus the results index and claims-ledger check."""

import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from evals.analysis.aggregate import (
    aggregate_records,
    load_jsonl,
    locomo_diagnostics,
    score_record,
)
from evals.schemas import EvaluationRecord


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(
    summary: Mapping[str, Mapping[str, float]], *, exclude_suffixes: tuple[str, ...] = ()
) -> str:
    """Render ``system x metric``; metrics a system did not measure show "—", never 0."""
    metrics = sorted({
        metric for values in summary.values() for metric in values
        if not metric.endswith(exclude_suffixes)
    })
    lines = ["| system | " + " | ".join(metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for system, values in summary.items():
        lines.append("| " + system + " | " + " | ".join(
            _cell(values.get(metric)) for metric in metrics
        ) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(
    summary: Mapping[str, Mapping[str, float]], path: str | Path,
    *, exclude_suffixes: tuple[str, ...] = (),
) -> None:
    """Write :func:`markdown_table` to ``path``, creating parent directories."""
    write_text(path, markdown_table(summary, exclude_suffixes=exclude_suffixes))


def rows_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    """Render dict rows as a Markdown table with the given column order."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(row.get(column)) for column in columns) + " |"
              for row in rows]
    return "\n".join(lines) + "\n"


def write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


# --- run report -----------------------------------------------------------------------------

QUESTION_COLUMNS = (
    "condition", "account", "question_id", "type", "question", "expected", "answer", "correct",
    "gold_found", "other_scope_memories", "retrieval_ms", "answer_ms",
)


def _expected(record: EvaluationRecord) -> str:
    if record.benchmark_metadata.get("category") == "5":
        return "(should abstain)"
    return record.gold_answer


def _is_correct(record: EvaluationRecord) -> bool | None:
    """Whether the answer was right; None when no answer was evaluated."""
    if record.failure_type is not None:
        return False
    if record.official_metric in {"llm_judge_accuracy", "substring_exact_match", "exact_match"}:
        return None if record.official_score is None else record.official_score >= 1.0
    if record.answer_evaluated:
        return bool(score_record(record).metrics.get("exact_match"))
    return None


def question_rows(records: list[EvaluationRecord]) -> list[dict[str, object]]:
    """One flat, spreadsheet-friendly row per question and condition."""
    rows = []
    for record in records:
        metrics = score_record(record).metrics
        correct = _is_correct(record)
        gold = metrics.get("gold_hit_at_8")
        other = metrics.get("any_cross_scope_contamination")
        rows.append({
            "condition": f"{record.dataset}/{record.system}/{record.ablation}",
            "account": record.scenario_id, "question_id": record.question_id,
            "type": record.question_type, "question": record.question,
            "expected": _expected(record), "answer": record.answer,
            "correct": None if correct is None else ("yes" if correct else "no"),
            "gold_found": None if gold is None else ("yes" if gold else "no"),
            "other_scope_memories": None if other is None else ("yes" if other else "no"),
            "retrieval_ms": round(record.retrieval_latency_ms, 1),
            "answer_ms": None if record.answer_latency_ms is None
            else round(record.answer_latency_ms, 1),
        })
    return rows


# Scope-isolation and string-match metrics are not meaningful on one-scope external datasets
# (LoCoMo category 5 even stores the wrong answer as gold), so they are not listed for them.
EXTERNAL_EXCLUDED = frozenset({
    "exact_match", "token_f1", "cross_scope_contamination", "any_cross_scope_contamination",
    "stale_memory_error_rate",
})


def _value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}"


def write_readable_report(
    jsonl_paths: Sequence[Path], destination: Path, *, stem: str | None = None
) -> Path:
    """Write ``report.md`` (``metric: value`` lines per condition) and ``questions.csv``.

    With ``stem`` the files are named ``<stem>.report.md`` / ``<stem>.questions.csv`` so they
    can sit beside a single JSONL file. Datasets are listed separately, never mixed.
    """
    prefix = f"{stem}." if stem else ""
    records = [r for path in jsonl_paths for r in load_jsonl([path])]
    destination.mkdir(parents=True, exist_ok=True)
    first = records[0]
    title = stem or (destination.parent.name if destination.name == "report" else destination.name)
    lines = [f"# Run {title}", "", f"mode: {first.evaluation_mode}",
             f"date (UTC): {str(first.timestamp)[:16]}", f"commit: {str(first.git_commit)[:7]}", ""]
    for dataset in sorted({r.dataset for r in records}):
        subset = [r for r in records if r.dataset == dataset]
        lines += [f"## {dataset}", ""]
        locomo = locomo_diagnostics(subset) if dataset == "locomo" else {}
        for system, values in aggregate_records(subset).items():
            lines.append(f"### {system}")
            for name in sorted(values):
                if name.endswith(("_ci95_low", "_ci95_high")):
                    continue
                if dataset != "cross_scope_mem" and name in EXTERNAL_EXCLUDED:
                    continue
                lines.append(f"{name}: {_value(values[name])}")
            block = locomo.get(system if "/" in system else f"{system}/full")
            if isinstance(block, dict):
                without_5 = block["excluding_category_5"]["judge_accuracy"]["mean"]
                lines.append(f"judge_accuracy_without_category_5: {_value(without_5)}")
                for category, stats in block["categories"].items():
                    lines.append(f"category_{category}_judge_accuracy: "
                                 f"{_value(stats['judge_accuracy'])}")
            lines.append("")
    report = destination / f"{prefix}report.md"
    report.write_text("\n".join(lines))
    with (destination / f"{prefix}questions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_COLUMNS)
        writer.writeheader()
        writer.writerows(question_rows(records))
    return report


# --- results index and claims ledger -------------------------------------------------------

INDEX_COLUMNS = (
    "run_id", "date", "commit", "dataset", "systems", "seeds", "judge", "metric", "N",
    "status", "path",
)
STATUS_VALUES = ("official", "smoke", "superseded")
CLAIM_STATUS_VALUES = ("supported", "partial", "unsupported", "superseded")
LEDGER_COLUMNS = ("id", "claim", "number", "source", "command", "caveat", "status")
BANNED_PHRASES = (
    "state-of-the-art", "outperform", "superior to", "production-ready", "production ready",
    "best-in-class", "beats ",
)


def _first_record(path: Path) -> dict[str, object] | None:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                parsed: dict[str, object] = json.loads(line)
                return parsed
    return None


def results_index(root: Path) -> list[dict[str, object]]:
    """One row per raw result file (batches grouped by directory); status is filled in later."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.jsonl")):
        if path.name.endswith(".bulk.jsonl"):
            continue
        key = str(path.parent) if "batches" in path.parts else str(path)
        groups[key].append(path)
    rows: list[dict[str, object]] = []
    for key, paths in groups.items():
        records = [json.loads(line) for path in paths for line in path.read_text().splitlines()
                   if line.strip()]
        if not records:
            continue
        first = records[0]
        is_batch = "batches" in paths[0].parts
        systems = sorted({
            f"{r.get('system')}/{r.get('ablation') or 'full'}" if r.get("ablation")
            else str(r.get("system")) for r in records
        })
        judges = sorted({str(r["judge_model"]) for r in records if r.get("judge_model")})
        metrics = sorted({str(r["official_metric"]) for r in records if r.get("official_metric")})
        if not metrics:
            metrics = ["recall_at_8", *(["exact_match"] if first.get("answer_evaluated") else [])]
        rows.append({
            "run_id": Path(key).name if is_batch else str(first.get("run_id")),
            "date": str(first.get("timestamp", ""))[:10],
            "commit": str(first.get("git_commit") or "")[:7],
            "dataset": first.get("dataset"),
            "systems": ", ".join(systems),
            "seeds": ", ".join(str(s) for s in sorted({r.get("seed") for r in records})),
            "judge": ", ".join(judges) or "none",
            "metric": ", ".join(metrics),
            "N": len(records),
            "path": str(paths[0].parent if is_batch else paths[0]),
        })
    return rows


def readme_statuses(readme: Path) -> dict[str, str]:
    """Map ``path`` -> status from the hand-curated run table in ``results/README.md``."""
    statuses: dict[str, str] = {}
    if not readme.exists():
        return statuses
    for line in readme.read_text().splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= len(INDEX_COLUMNS) and cells[-2] in STATUS_VALUES:
            statuses[cells[-1]] = cells[-2]
    return statuses


def print_results_index(root: Path) -> int:
    """Print the derived run table; return 1 if any run lacks a status in results/README.md."""
    statuses = readme_statuses(root / "README.md")
    rows = results_index(root)
    for row in rows:
        row["status"] = statuses.get(str(row["path"]), "UNLABELLED")
    print(rows_table(rows, INDEX_COLUMNS))
    missing = [str(row["path"]) for row in rows if row["status"] == "UNLABELLED"]
    if missing:
        print("Runs missing from the results/README.md table:", file=sys.stderr)
        print(*missing, sep="\n  ", file=sys.stderr)
    return 1 if missing else 0


def _table_after(lines: list[str], header: tuple[str, ...]) -> list[dict[str, str]]:
    wanted = "| " + " | ".join(header) + " |"
    rows: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        if line.strip() == wanted:
            for body in lines[index + 2:]:
                if not body.startswith("|"):
                    break
                cells = [cell.strip() for cell in body.strip().strip("|").split("|")]
                rows.append(dict(zip(header, cells, strict=False)))
    return rows


def _json_value(file: Path, dotted: str) -> object:
    node: object = json.loads(file.read_text())
    for part in dotted.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]  # type: ignore[index]
    return node


def check_claims(ledger: Path, repo: Path) -> list[str]:
    """Return problems found in the RESULTS.md ledger (empty list = consistent)."""
    problems: list[str] = []
    lines = ledger.read_text().splitlines()
    claims = _table_after(lines, LEDGER_COLUMNS)
    if not claims:
        return [f"{ledger}: no claims table with columns {LEDGER_COLUMNS}"]
    ids = set()
    for row in claims:
        claim_id = row["id"]
        ids.add(claim_id)
        for column in LEDGER_COLUMNS:
            if not row.get(column):
                problems.append(f"{claim_id}: empty '{column}'")
        if row.get("status") not in CLAIM_STATUS_VALUES:
            problems.append(f"{claim_id}: status must be one of {CLAIM_STATUS_VALUES}")
        if row.get("status") in {"supported", "partial"}:
            for path in re.findall(r"`([^`]+\.(?:json|jsonl|md))`", row.get("source", "")):
                if not (repo / path).exists():
                    problems.append(f"{claim_id}: source file missing: {path}")
    for raw_check in _table_after(lines, ("id", "file", "path", "expected")):
        check = {key: value.strip("`") for key, value in raw_check.items()}
        if check["id"] not in ids:
            problems.append(f"{check['id']}: machine-checked value has no ledger row")
            continue
        try:
            actual = _json_value(repo / check["file"], check["path"])
        except (OSError, KeyError, IndexError, ValueError) as exc:
            problems.append(f"{check['id']}: cannot read {check['file']}:{check['path']} ({exc})")
            continue
        expected = float(check["expected"])
        tolerance = 1e-6 * max(1.0, abs(expected))
        if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > tolerance:
            problems.append(f"{check['id']}: expected {expected}, file has {actual}")
    for path in [ledger, repo / "README.md", *sorted((repo / "docs").glob("*.md"))]:
        text = path.read_text().casefold()
        problems += [f"{path.name}: banned claim wording {phrase!r}"
                     for phrase in BANNED_PHRASES if phrase in text]
    return problems


def main() -> None:
    """CLI: ``index [results_dir]`` prints the run table; ``claims [RESULTS.md]`` checks it."""
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "index":
        raise SystemExit(print_results_index(Path(sys.argv[2] if len(sys.argv) > 2 else "results")))
    if command == "claims":
        ledger = Path(sys.argv[2] if len(sys.argv) > 2 else "RESULTS.md")
        problems = check_claims(ledger, Path("."))
        print("\n".join(problems) if problems else f"{ledger}: ledger consistent")
        raise SystemExit(1 if problems else 0)
    raise SystemExit("usage: python -m evals.analysis.tables index [results] | claims [RESULTS.md]")


if __name__ == "__main__":
    main()
