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
    confidence_interval_report,
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


# --- human-readable run report -------------------------------------------------------------

HEADLINE_METRICS = (
    ("recall_at_8", "Found the evidence (Recall@8)", False),
    ("cross_scope_contamination", "Pulled in other scopes' memories", True),
    ("stale_memory_error_rate", "Returned outdated memories", True),
    ("exact_match", "Answered correctly (exact match)", False),
    ("official_llm_judge_accuracy", "Answered correctly (judge)", False),
)
QUESTION_COLUMNS = (
    "condition", "account", "question_id", "type", "question", "expected", "answer", "correct",
    "gold_found", "other_scope_memories", "retrieval_ms", "answer_ms",
)


def _pct(value: object) -> str:
    return "—" if value is None else f"{float(value):.1%}"  # type: ignore[arg-type]


def _short(text: object, width: int = 70) -> str:
    flat = " ".join(str(text or "").split()).replace("|", "/")
    return flat if len(flat) <= width else flat[: width - 1] + "…"


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


def _dataset_section(dataset: str, records: list[EvaluationRecord]) -> list[str]:
    summary = aggregate_records(records)
    out = [f"## {dataset}", ""]
    metrics = [m for m in HEADLINE_METRICS if any(m[0] in v for v in summary.values())]
    if dataset != "cross_scope_mem":
        # Scope contamination, stale rate and source-recall mean little on one-scope datasets.
        metrics = [(key, label, lower) for key, label, lower in metrics
                   if key.startswith("official_")]
        metrics += [(key, key.removeprefix("official_"), False)
                    for key in sorted({k for v in summary.values() for k in v})
                    if key.startswith("official_") and not key.endswith(("_ci95_low", "_ci95_high"))
                    and key not in {m[0] for m in metrics}]
    columns = [label for _, label, _ in metrics] + ["Typical retrieval time (ms)"]
    out += ["| Condition | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for system, values in summary.items():
        cells = [_pct(values.get(key)) for key, _, _ in metrics]
        latency = values.get("retrieval_p50_ms")
        out.append(f"| {system} | " + " | ".join(cells)
                   + f" | {'—' if latency is None else f'{latency:,.0f}'} |")
    accounts = {int(v["account_count"]) for v in summary.values()}
    out += ["", f"Accounts / conversations: {', '.join(map(str, sorted(accounts)))}. "
            "Retrieval time is the median of the timed retrieval call only.", ""]
    ci = confidence_interval_report(records)["paired_differences_full_minus_control"]
    assert isinstance(ci, dict)
    if ci:
        out += ["### Full ScopeGraph compared with each control", "",
                "Difference = full minus control, with a 95% interval across accounts. "
                "For the 'pulled in' and 'outdated' rows, negative is good for full.", "",
                "| Control | Measure | Difference | 95% interval | Verdict |",
                "|---|---|---|---|---|"]
        for label, entry in ci.items():
            for key, name, _ in metrics:
                m = entry["metrics"].get(key)
                if m is None:
                    continue
                low, high = m["ci95_low"], m["ci95_high"]
                interval = "—" if low is None else f"{low:+.3f} to {high:+.3f}"
                verdict = {"full": "full is better", "control": "control is better"}.get(
                    m["favours"], "no clear difference")
                out.append(f"| {label.split('/')[-1]} | {name} | {m['mean_difference']:+.3f} "
                           f"| {interval} | {verdict} |")
        out.append("")
    if dataset == "locomo":
        out += _locomo_lines(records)
    return out


def _locomo_lines(records: list[EvaluationRecord]) -> list[str]:
    out = ["### LoCoMo by question category", ""]
    for system, block in locomo_diagnostics(records).items():
        assert isinstance(block, dict)
        a, b = block["all_categories"], block["excluding_category_5"]
        out += [f"**{system}** — accuracy {_pct(a['judge_accuracy']['mean'])} with all "
                f"categories, {_pct(b['judge_accuracy']['mean'])} without category 5 "
                "(category 5 is graded by whether the model abstains).", "",
                "| Category | Questions | Accuracy | Gold evidence found (retrieved) | "
                "Gold evidence shown to the model | Found none of it |",
                "|---|---|---|---|---|---|"]
        for category, st in block["categories"].items():
            out.append(
                f"| {category} | {st['n']} | {_pct(st['judge_accuracy'])} "
                f"| {_pct(st['gold_source_recall_retrieved'])} "
                f"| {_pct(st['gold_source_recall_delivered'])} "
                f"| {_pct(st['zero_gold_retrieved_rate'])} |"
            )
        out.append("")
    return out


def _mistakes(records: list[EvaluationRecord], limit: int = 15) -> list[str]:
    full = [r for r in records if r.ablation == "full"]
    wrong = [r for r in full if _is_correct(r) is False]
    heading = "answered wrong"
    if not wrong:
        wrong = [r for r in full if score_record(r).metrics.get("gold_hit_at_8") == 0.0]
        heading = "did not retrieve the evidence"
    if not wrong:
        return ["## Mistakes", "", f"None: full ScopeGraph retrieved the evidence for all "
                f"{len(full)} questions.", ""]
    out = [f"## Where full ScopeGraph {heading} ({len(wrong)} of {len(full)} questions)", ""]
    if wrong:
        out += ["| Account | Question | Expected | Got |", "|---|---|---|---|"]
        out += [f"| {r.scenario_id} | {_short(r.question)} | {_short(_expected(r), 45)} "
                f"| {_short(r.answer or '—', 45)} |" for r in wrong[:limit]]
        if len(wrong) > limit:
            out.append(f"\n…and {len(wrong) - limit} more; all rows are in `questions.csv`.")
    return [*out, ""]


def write_readable_report(
    jsonl_paths: Sequence[Path], destination: Path, *, stem: str | None = None
) -> Path:
    """Write ``report.md`` (read this) and ``questions.csv`` (every question) for a run.

    With ``stem`` the files are named ``<stem>.report.md`` / ``<stem>.questions.csv`` so they
    can sit beside a single JSONL file.
    """
    prefix = f"{stem}." if stem else ""
    records = [r for path in jsonl_paths for r in load_jsonl([path])]
    destination.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {}
    for candidate in (jsonl_paths[0].parent / "run.json", jsonl_paths[0].with_suffix(".run.json")):
        if candidate.exists():
            meta = json.loads(candidate.read_text())
            break
    git = meta.get("git") if isinstance(meta.get("git"), dict) else {}
    first = records[0]
    title = stem or (destination.parent.name if destination.name == "report" else destination.name)
    lines = [
        f"# Run {title}",
        "",
        f"- **Date (UTC):** {str(first.timestamp)[:16]}  ",
        f"- **Commit:** {str(git.get('commit', first.git_commit))[:7]}"  # type: ignore[union-attr]
        + (" (uncommitted changes)" if git.get("dirty") else "") + "  ",  # type: ignore[union-attr]
        f"- **Mode:** {first.evaluation_mode}  ",
        f"- **Questions:** {len(records):,} across "
        f"{len({r.system + '/' + r.ablation for r in records})} condition(s)",
        "",
        "How to read this: every table is one dataset; datasets are never mixed. "
        "Percentages are averages over questions. Detail for every question is in "
        "`questions.csv`; field definitions are in `results/SCHEMA.md`.", "",
    ]
    for dataset in sorted({r.dataset for r in records}):
        lines += _dataset_section(dataset, [r for r in records if r.dataset == dataset])
    lines += _mistakes(records)
    notes = []
    if str(first.evaluation_mode).find("oracle") >= 0 or "answer=not-evaluated" in str(
        first.evaluation_mode
    ):
        notes.append("Offline run: facts come from the generator's oracle labels, embeddings are "
                     "a local hash, and no answer model ran, so 'answered correctly' is blank.")
    if any(r.ablation == "vector_scope_filter" for r in records):
        notes.append("`vector_scope_filter` is 0% 'pulled in other scopes' by construction: it "
                     "filters on the same allowed scopes the measure uses.")
    if len({r.scenario_id for r in records}) < 10:
        notes.append("Fewer than 10 accounts/conversations: intervals are not pilot-grade.")
    if notes:
        lines += ["## Caveats", ""] + [f"- {n}" for n in notes] + [""]
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
