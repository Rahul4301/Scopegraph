"""As-of querying on LoCoMo (PREREG.md Section 10, exploratory).

Each question is asked after the session that holds its last evidence turn (the as-of
checkpoint) and again at the end of the conversation. A checkpoint history is the conversation
cut after that session. All checkpoints of one conversation share one memory store
(History.store_id), so memory arms ingest sessions incrementally, and the transcript sent by
arm A at checkpoint k is a prefix of the one at checkpoint k+1 so cached prefixes grow.

Questions are ordered by conversation, then checkpoint, then original order, so each store only
ever moves forward in time.
"""

from __future__ import annotations

import re

from memstudy.schema import History, Item

EVIDENCE = re.compile(r"D(\d+):\d+")


def last_evidence_session(evidence: list[str], session_numbers: list[int]) -> int:
    """Number of the latest session cited as evidence that exists in the conversation. A question
    with no readable evidence is asked only at the end, so it maps to the last session."""
    present = set(session_numbers)
    cited = [int(n) for ref in evidence for n in EVIDENCE.findall(ref) if int(n) in present]
    return max(cited) if cited else max(session_numbers)


def checkpoint_id(history_id: str, session: int) -> str:
    return f"{history_id}@s{session:02d}"


def expand_asof(
    histories: dict[str, History], items: list[Item]
) -> tuple[dict[str, History], list[Item]]:
    """Checkpoint histories and as-of items for the given LoCoMo items. Each question yields an
    as-of copy (item_id suffix .asof) unless its checkpoint is the final session, and always a
    final copy (suffix .final) on the full conversation."""
    out_histories: dict[str, History] = {}
    out_items: list[Item] = []
    by_conv: dict[str, list[Item]] = {}
    for item in items:
        by_conv.setdefault(item.history_id, []).append(item)
    for hid, group in by_conv.items():
        full = histories[hid]
        numbers = sorted(int(s.session_id) for s in full.sessions)
        last = numbers[-1]
        staged: list[tuple[int, int, Item]] = []
        for order, item in enumerate(group):
            k = last_evidence_session(list(item.meta.get("evidence", [])), numbers)
            if k < last:
                cut = checkpoint_id(hid, k)
                if cut not in out_histories:
                    out_histories[cut] = History(
                        history_id=cut,
                        bench=full.bench,
                        sessions=[s for s in full.sessions if int(s.session_id) <= k],
                        store_id=hid,
                    )
                staged.append(
                    (k, order, item.model_copy(update={
                        "item_id": f"{item.item_id}.asof",
                        "history_id": cut,
                        "meta": {**item.meta, "checkpoint": "asof", "asof_session": k},
                    }))
                )
            staged.append(
                (last, order, item.model_copy(update={
                    "item_id": f"{item.item_id}.final",
                    "meta": {**item.meta, "checkpoint": "final", "asof_session": last},
                }))
            )
        out_histories[hid] = full
        out_items.extend(item for _, _, item in sorted(staged, key=lambda t: (t[0], t[1])))
    return out_histories, out_items
