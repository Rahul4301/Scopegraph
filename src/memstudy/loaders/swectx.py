"""SWE Context Bench loader (coding benchmark).

VibeMemBench was not usable: its code repository is a stub (README only) and no trajectories are
published (checked 2026-10-01). SWE Context Bench is the fallback, with this caveat: it also
ships no trajectories. The experience pool here is each prior task's issue and gold patch, a
verified-correct experience, not an agent trajectory.

Scope mapping: one history per repository. user_id = "swectx_<owner>__<repo>". A related task
only ever queries the history of its own repository, so memories from repo A are never
retrievable in repo B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from memstudy.schema import History, Item, Session, Turn

BENCH = "swectx"
INSTANCE_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "hints_text",
    "created_at",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
)


def repo_history_id(repo: str) -> str:
    return repo.replace("/", "__")


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(pq.read_table(path).to_pylist())


def build_swectx(
    experience: list[dict[str, Any]],
    related: list[dict[str, Any]],
    relationship: list[dict[str, Any]],
) -> tuple[dict[str, History], list[Item]]:
    experience_ids = {r["instance_id"] for r in experience}
    sessions: dict[str, list[Session]] = {}
    for row in sorted(experience, key=lambda r: (r["created_at"], r["instance_id"])):
        sessions.setdefault(repo_history_id(row["repo"]), []).append(
            Session(
                session_id=row["instance_id"],
                timestamp=row["created_at"],
                turns=[
                    Turn(speaker="issue", text=row["problem_statement"]),
                    Turn(speaker="fix", text=row["patch"]),
                ],
            )
        )
    links: dict[str, list[str]] = {}
    for rel in relationship:
        links.setdefault(rel["related_instance_id"], []).append(rel["experience_instance_id"])

    histories = {
        hid: History(history_id=hid, bench=BENCH, sessions=sess) for hid, sess in sessions.items()
    }
    items: list[Item] = []
    for row in related:
        if row["instance_id"] in experience_ids:
            raise ValueError(f"{row['instance_id']} is in both the experience and related sets")
        hid = repo_history_id(row["repo"])
        if hid not in histories:
            continue  # no experience for this repository, nothing for memory to supply
        items.append(
            Item(
                item_id=row["instance_id"],
                bench=BENCH,
                history_id=hid,
                category=row["repo"],
                question=row["problem_statement"],
                gold=row["patch"],
                kind="coding",
                meta={
                    "swebench_instance": {k: row[k] for k in INSTANCE_FIELDS},
                    "oracle_experience_ids": links.get(row["instance_id"], []),
                },
            )
        )
    return histories, items


def load_swectx(data_dir: Path, lite: bool = False) -> tuple[dict[str, History], list[Item]]:
    suffix = "Lite_" if lite else ""
    data_dir = Path(data_dir)
    experience = _rows(data_dir / f"SWEContextBench_{suffix}Experience.parquet")
    related_name = "Related_Lite" if lite else "Related"
    related = _rows(data_dir / f"SWEContextBench_{related_name}.parquet")
    relationship = _rows(data_dir / "SWEContextBench_Relationship.parquet")
    return build_swectx(experience, related, relationship)
