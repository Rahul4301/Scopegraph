"""Keep the documentation, Makefile and claims ledger consistent with the code."""

import json
import re
from pathlib import Path

from evals.analysis.tables import check_claims, readme_statuses
from evals.schemas import EvaluationRecord

ROOT = Path(__file__).resolve().parents[2]


def test_claude_and_agents_files_are_identical():
    assert (ROOT / "CLAUDE.md").read_bytes() == (ROOT / "AGENTS.md").read_bytes()


def test_every_record_field_is_defined_in_the_results_schema():
    schema = (ROOT / "results" / "SCHEMA.md").read_text()
    missing = [name for name in EvaluationRecord.model_fields if f"`{name}`" not in schema]
    assert not missing, f"results/SCHEMA.md does not define: {missing}"


def test_docs_index_lists_every_doc():
    index = (ROOT / "docs" / "README.md").read_text()
    for doc in sorted((ROOT / "docs").glob("*.md")):
        if doc.name != "README.md":
            assert f"({doc.name})" in index, f"docs/README.md is missing {doc.name}"


def test_makefile_recipes_use_tabs_and_define_required_targets():
    makefile = (ROOT / "Makefile").read_text()
    for number, line in enumerate(makefile.splitlines(), start=1):
        assert not re.match(r" +\S", line) or line.lstrip().startswith("#"), (
            f"Makefile:{number} is space-indented; recipes need tabs"
        )
    for target in ("eval-ablation", "results-index", "claims-check", "eval-smoke", "check"):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE), target
    smoke = makefile[makefile.index("\neval-smoke:"):]
    assert "$$(date +%m_%d__%H_%M).jsonl" in smoke[:1200]


def test_results_readme_table_assigns_a_valid_status():
    statuses = readme_statuses(ROOT / "results" / "README.md")
    assert statuses
    assert set(statuses.values()) <= {"official", "smoke", "superseded"}


def test_claims_check_catches_wrong_numbers_missing_files_and_banned_wording(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("A plain readme.\n")
    (tmp_path / "out.json").write_text(json.dumps({"a": {"b": 0.5}}))
    ledger = tmp_path / "RESULTS.md"
    row = "| X1 | claim | 0.5 | `out.json` | `make x` | caveat | supported |"
    header = (
        "| id | claim | number | source | command | caveat | status |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    checks = (
        "\n| id | file | path | expected |\n|---|---|---|---|\n"
        "| X1 | `out.json` | `a.b` | {} |\n"
    )
    ledger.write_text(header + row + "\n" + checks.format("0.5"))
    assert check_claims(ledger, tmp_path) == []
    ledger.write_text(header + row + "\n" + checks.format("0.6"))
    assert any("expected 0.6" in problem for problem in check_claims(ledger, tmp_path))
    ledger.write_text(header + row.replace("out.json", "gone.json") + "\n")
    assert any("source file missing" in problem for problem in check_claims(ledger, tmp_path))
    ledger.write_text(header + row.replace("claim", "it outperforms rivals") + "\n")
    assert any("banned claim wording" in problem for problem in check_claims(ledger, tmp_path))
