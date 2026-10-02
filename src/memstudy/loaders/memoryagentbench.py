"""MemoryAgentBench loader (ai-hyz/MemoryAgentBench, pinned in configs/data_manifest.yaml).

Each parquet row is one long context shared by many questions, so a row is one History holding
the context verbatim (History.document) and one Item per question. Nothing is cut or reordered.

Items carry every accepted answer in meta["answers"]; Item.gold is the first of them. Accurate
retrieval and conflict resolution (selective forgetting) are the confirmatory competencies in the
proposal and load as primary; test-time learning and long-range understanding are exploratory.
Native-metric scoring is not part of the loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from memstudy.schema import History, Item, safe_name

BENCH = "memoryagentbench"
COMPETENCIES = (
    "Accurate_Retrieval",
    "Conflict_Resolution",
    "Long_Range_Understanding",
    "Test_Time_Learning",
)
CONFIRMATORY = {"Accurate_Retrieval", "Conflict_Resolution"}
# Metadata lists aligned with the questions, copied per item when the row has them.
_PER_QUESTION = {
    "qa_pair_ids": "qa_pair_id",
    "previous_events": "previous_events",
    "question_ids": "question_id",
    "question_types": "question_type",
}
# Per-row metadata copied to every item of the row (the in-context demo and gold key points).
_PER_ROW = ("demo", "keypoints")


def _at(values: Any, index: int) -> Any:
    return values[index] if values is not None else None


def load_memoryagentbench(data_dir: Path) -> tuple[dict[str, History], list[Item]]:
    histories: dict[str, History] = {}
    items: list[Item] = []
    for competency in COMPETENCIES:
        files = sorted(Path(data_dir).glob(f"{competency}-*.parquet"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one {competency} parquet file in {data_dir}, found {len(files)}")
        for row_no, row in enumerate(pq.read_table(files[0]).to_pylist()):
            meta: dict[str, Any] = row["metadata"] or {}
            source = str(meta["source"])
            history_id = safe_name(f"{competency.lower()}-{source}-r{row_no:02d}")
            histories[history_id] = History(history_id=history_id, bench=BENCH, document=row["context"])
            questions, answers = row["questions"], row["answers"]
            if len(questions) != len(answers):
                raise ValueError(f"{history_id}: {len(questions)} questions but {len(answers)} answers")
            for i, (question, accepted) in enumerate(zip(questions, answers, strict=True)):
                accepted_list = [str(a) for a in accepted]
                item_meta: dict[str, Any] = {
                    "competency": competency,
                    "source": source,
                    "row": row_no,
                    "answers": accepted_list,
                    "overlaps_longmemeval": source.startswith("longmemeval_s"),
                }
                for key, name in _PER_QUESTION.items():
                    value = _at(meta.get(key), i)
                    if value is not None:
                        item_meta[name] = value
                for key in _PER_ROW:
                    if meta.get(key) is not None:
                        item_meta[key] = meta[key]
                if str(item_meta.get("question_id", "")).endswith("_abs"):
                    item_meta["abstention"] = True
                items.append(
                    Item(
                        item_id=f"{history_id}-q{i:03d}",
                        bench=BENCH,
                        history_id=history_id,
                        category=competency,
                        question=str(question),
                        gold=accepted_list[0],
                        question_date=_at(meta.get("question_dates"), i),
                        primary=competency in CONFIRMATORY,
                        meta=item_meta,
                    )
                )
    return histories, items
