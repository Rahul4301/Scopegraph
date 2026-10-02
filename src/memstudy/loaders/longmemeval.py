"""LongMemEval-S loader: 500 questions, one haystack (history) per question."""

from __future__ import annotations

import json
from pathlib import Path

from memstudy.schema import History, Item, Session, Turn

BENCH = "longmemeval"


def load_longmemeval(path: Path) -> tuple[dict[str, History], list[Item]]:
    raw = json.loads(Path(path).read_text())
    histories: dict[str, History] = {}
    items: list[Item] = []
    for entry in raw:
        qid = str(entry["question_id"])
        sessions = [
            Session(
                session_id=str(sid),
                timestamp=date,
                turns=[Turn(speaker=m["role"], text=m["content"]) for m in turns],
            )
            for sid, date, turns in zip(
                entry["haystack_session_ids"],
                entry["haystack_dates"],
                entry["haystack_sessions"],
                strict=True,
            )
        ]
        histories[qid] = History(history_id=qid, bench=BENCH, sessions=sessions)
        items.append(
            Item(
                item_id=qid,
                bench=BENCH,
                history_id=qid,
                category=str(entry["question_type"]),
                question=entry["question"],
                gold=str(entry["answer"]),
                question_date=entry.get("question_date"),
                meta={
                    "abstention": qid.endswith("_abs"),
                    "evidence_session_ids": [str(x) for x in entry.get("answer_session_ids", [])],
                },
            )
        )
    return histories, items
