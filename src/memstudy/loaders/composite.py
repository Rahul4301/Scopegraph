"""Composite long histories for the history-length axis (PREREG.md Section 10, exploratory).

A composite concatenates the evidence sessions of a fixed random subset of LongMemEval-S
questions with distractor sessions drawn from other questions' haystacks, built to a target
length. Every question whose evidence sessions all lie in the composite is asked of it.

Everything that could change a result is fixed in the study config and the seed:
  * the question subset is one seeded draw, shared by every target length, so accuracy at
    different lengths is measured on the same questions;
  * distractors come from one seeded shuffle of the sessions that are evidence for no question
    in LongMemEval-S, so a distractor can never be another question's gold evidence;
  * conflict screen: a distractor is also rejected when it is topically close to any asked
    question, meaning it contains at least max(topical_min_hits, ceil(topical_threshold * k))
    of that question's k content words. Rejections are counted in the build report;
  * sessions are ordered by their timestamps, ties in original order.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memstudy.schema import History, Item, Session, Turn, render_session
from memstudy.tokens import count_tokens

BENCH = "composite"
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
MIN_WORD = 4
STOPWORDS = frozenset(
    """about after again also been before being between both could did does doing during each from
    have having here how into just like many more most much only other over same should some such
    than that their them then there these they this those through very want was were what when
    where which while who whom why will with would your you""".split()
)
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CompositeSpec:
    seed: int
    targets: tuple[int, ...]
    questions: int
    topical_threshold: float
    topical_min_hits: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CompositeSpec:
        c = cfg["composite"]
        return cls(
            seed=int(cfg["seed"]),
            targets=tuple(int(t) for t in c["targets"]),
            questions=int(c["questions"]),
            topical_threshold=float(c["topical_threshold"]),
            topical_min_hits=int(c["topical_min_hits"]),
        )


def keywords(text: str) -> frozenset[str]:
    """Content words: lowercase alphanumeric tokens of at least MIN_WORD characters, no stopwords."""
    return frozenset(w for w in _WORD.findall(text.lower()) if len(w) >= MIN_WORD and w not in STOPWORDS)


def is_topical(question_words: frozenset[str], session_words: frozenset[str], threshold: float, min_hits: int) -> bool:
    """The conflict screen: does a session share enough content words with a question?"""
    if not question_words:
        return False
    need = max(min_hits, math.ceil(threshold * len(question_words)))
    return len(question_words & session_words) >= need


def _session_date(session: Session) -> datetime:
    try:
        return datetime.strptime(session.timestamp or "", DATE_FORMAT)
    except ValueError:
        return datetime.max


def label(target: int) -> str:
    return f"{target // 1000}k"


class _Pool:
    """Sessions of a LongMemEval-S file by id, with lazily counted tokens and keywords."""

    def __init__(self, raw: list[dict[str, Any]]) -> None:
        self.sessions: dict[str, Session] = {}
        for entry in raw:
            for sid, date, turns in zip(
                entry["haystack_session_ids"], entry["haystack_dates"], entry["haystack_sessions"], strict=True
            ):
                if sid not in self.sessions:
                    self.sessions[str(sid)] = Session(
                        session_id=str(sid),
                        timestamp=date,
                        turns=[Turn(speaker=m["role"], text=m["content"]) for m in turns],
                    )
        self._tokens: dict[str, int] = {}
        self._words: dict[str, frozenset[str]] = {}

    def tokens(self, sid: str) -> int:
        if sid not in self._tokens:
            self._tokens[sid] = count_tokens(render_session(self.sessions[sid]))
        return self._tokens[sid]

    def words(self, sid: str) -> frozenset[str]:
        if sid not in self._words:
            self._words[sid] = keywords(" ".join(t.text for t in self.sessions[sid].turns))
        return self._words[sid]


def choose_questions(raw: list[dict[str, Any]], spec: CompositeSpec, exclude: frozenset[str]) -> list[dict[str, Any]]:
    """The fixed random subset: eligible questions (not abstention, not in the development set)
    sorted by id, shuffled with the registered seed, first spec.questions kept."""
    eligible = sorted(
        (e for e in raw if not str(e["question_id"]).endswith("_abs") and str(e["question_id"]) not in exclude),
        key=lambda e: str(e["question_id"]),
    )
    random.Random(spec.seed).shuffle(eligible)
    return eligible[: spec.questions]


def build_composite(
    pool: _Pool,
    raw: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    distractor_order: list[str],
    spec: CompositeSpec,
    target: int,
    exclude: frozenset[str],
) -> tuple[History, list[Item], dict[str, Any]]:
    evidence = [str(s) for e in chosen for s in e["answer_session_ids"]]
    ids = list(dict.fromkeys(evidence))
    total = sum(pool.tokens(s) for s in ids)
    if total > target:
        raise ValueError(
            f"the evidence sessions of {len(chosen)} questions need {total} tokens, over the "
            f"{target} target; lower composite.questions or raise the target"
        )
    question_words = [keywords(str(e["question"])) for e in chosen]
    rejected = {"topical": 0}
    for sid in distractor_order:
        t = pool.tokens(sid)
        if total + t > target:
            if target - total < 200:
                break
            continue
        words = pool.words(sid)
        if any(is_topical(q, words, spec.topical_threshold, spec.topical_min_hits) for q in question_words):
            rejected["topical"] += 1
            continue
        ids.append(sid)
        total += t
    ordered = sorted(ids, key=lambda s: _session_date(pool.sessions[s]))  # stable: ties keep order
    history_id = f"composite-{label(target)}"
    history = History(history_id=history_id, bench=BENCH, sessions=[pool.sessions[s] for s in ordered])
    have = set(ordered)
    by_id = {str(e["question_id"]): e for e in raw}
    asked = [
        e
        for e in sorted(by_id.values(), key=lambda e: str(e["question_id"]))
        if str(e["question_id"]) not in exclude
        and not str(e["question_id"]).endswith("_abs")
        and e["answer_session_ids"]
        and {str(s) for s in e["answer_session_ids"]} <= have
    ]
    cumulative = [0]
    for sid in ordered:
        cumulative.append(cumulative[-1] + pool.tokens(sid))
    items: list[Item] = []
    for e in asked:
        qid = str(e["question_id"])
        positions = [round(cumulative[ordered.index(str(s))] / max(1, total), 4) for s in e["answer_session_ids"]]
        items.append(
            Item(
                item_id=f"{qid}.c{label(target)}",
                bench=BENCH,
                history_id=history_id,
                category=str(e["question_type"]),
                question=e["question"],
                gold=str(e["answer"]),
                question_date=e.get("question_date"),
                meta={
                    "abstention": False,
                    "source_question_id": qid,
                    "target_tokens": target,
                    "evidence_session_ids": [str(s) for s in e["answer_session_ids"]],
                    "evidence_positions": positions,
                },
            )
        )
    report = {
        "history_id": history_id,
        "target_tokens": target,
        "history_tokens": total,
        "sessions": len(ordered),
        "evidence_sessions": len(set(evidence)),
        "distractor_sessions": len(ordered) - len(set(evidence)),
        "questions": len(items),
        "screened_out_topical": rejected["topical"],
    }
    return history, items, report


def load_composites(
    raw: list[dict[str, Any]], cfg: dict[str, Any], exclude: frozenset[str] = frozenset()
) -> tuple[dict[str, History], list[Item], list[dict[str, Any]]]:
    """Composite histories at every configured target length, their items, and a build report
    per history. exclude holds the development-set question ids that must not be used."""
    spec = CompositeSpec.from_config(cfg)
    pool = _Pool(raw)
    evidence_anywhere = {str(s) for e in raw for s in e["answer_session_ids"]}
    candidates = sorted(s for s in pool.sessions if s not in evidence_anywhere)
    random.Random(spec.seed).shuffle(candidates)
    chosen = choose_questions(raw, spec, exclude)
    histories: dict[str, History] = {}
    items: list[Item] = []
    reports: list[dict[str, Any]] = []
    for target in spec.targets:
        history, its, report = build_composite(pool, raw, chosen, candidates, spec, target, exclude)
        histories[history.history_id] = history
        items.extend(its)
        reports.append(report)
    return histories, items, reports


def development_question_ids(raw: list[dict[str, Any]], cfg: dict[str, Any], locomo_path: Path) -> frozenset[str]:
    """LongMemEval-S questions of the pilot sample (the development set, PREREG.md Section 6)."""
    from memstudy.loaders.locomo import load_locomo
    from memstudy.pilot import select_pilot

    light = [
        Item(
            item_id=str(e["question_id"]),
            bench="longmemeval",
            history_id=str(e["question_id"]),
            category=str(e["question_type"]),
            question=e["question"],
            gold=str(e["answer"]),
        )
        for e in raw
    ]
    pilot = select_pilot({"locomo": load_locomo(locomo_path)[1], "longmemeval": light}, cfg)
    return frozenset(i.item_id for i in pilot["longmemeval"])


def load_composite_bench(
    path: Path, locomo_path: Path, cfg: dict[str, Any]
) -> tuple[dict[str, History], list[Item]]:
    """The composite benchmark as the runner loads it: development-set questions excluded."""
    raw = json.loads(Path(path).read_text())
    histories, items, _ = load_composites(raw, cfg, development_question_ids(raw, cfg, locomo_path))
    return histories, items
