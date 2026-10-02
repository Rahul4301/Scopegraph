"""LoCoMo loader.

Categories 1 to 4 (1,540 questions) are primary. Category 5 (446 adversarial questions) is
loaded with primary=False and reported separately. Adversarial questions have no gold answer;
the correct behaviour is to say the information is not in the conversation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from memstudy.schema import History, Item, Session, Turn

BENCH = "locomo"
ADVERSARIAL_CATEGORY = 5
ABSTAIN_GOLD = "Not mentioned in the conversation."


def _session_keys(conversation: dict[str, object]) -> list[int]:
    nums = [
        int(m.group(1))
        for key in conversation
        if (m := re.fullmatch(r"session_(\d+)", key))
    ]
    return sorted(nums)


def load_locomo(path: Path) -> tuple[dict[str, History], list[Item]]:
    raw = json.loads(Path(path).read_text())
    histories: dict[str, History] = {}
    items: list[Item] = []
    for conv in raw:
        history_id = str(conv["sample_id"])
        c = conv["conversation"]
        sessions = [
            Session(
                session_id=str(n),
                timestamp=c.get(f"session_{n}_date_time"),
                turns=[Turn(speaker=t["speaker"], text=t["text"]) for t in c[f"session_{n}"]],
            )
            for n in _session_keys(c)
        ]
        histories[history_id] = History(history_id=history_id, bench=BENCH, sessions=sessions)
        for idx, qa in enumerate(conv["qa"]):
            category = int(qa["category"])
            adversarial = category == ADVERSARIAL_CATEGORY
            items.append(
                Item(
                    item_id=f"{history_id}-{idx:04d}",
                    bench=BENCH,
                    history_id=history_id,
                    category=f"cat{category}",
                    question=qa["question"],
                    gold=ABSTAIN_GOLD if adversarial else str(qa["answer"]),
                    primary=not adversarial,
                    meta={
                        "evidence": qa.get("evidence", []),
                        "abstention": adversarial,
                        "adversarial_answer": qa.get("adversarial_answer"),
                    },
                )
            )
    return histories, items
