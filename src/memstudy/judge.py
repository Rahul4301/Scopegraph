"""Judge: GPT-5 nano, the same graders for every arm.

The grader depends on the item, never on the arm (prompts.judge_scheme): LongMemEval questions use
LongMemEval's own per-type prompts, InfBench summaries use MemoryAgentBench's summary grader, and
everything else uses the study's single rule. LoCoMo answers also get a secondary grade with
Mem0's prompt, reported beside the primary one for comparison with published numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from memstudy.llm import CallResult, ModelCaller
from memstudy.prompts import (
    JUDGE_INSTRUCTIONS,
    judge_prompt,
    judge_scheme,
    longmemeval_judge_prompt,
    mem0_judge_prompt,
    summary_judge_prompts,
)
from memstudy.schema import Item, Usage
from memstudy.scoring import normalize

RULE_MODEL = "rule:not_mentioned"

_VERDICT = re.compile(r'"verdict"\s*:\s*"(CORRECT|INCORRECT)"', re.IGNORECASE)
_LABEL = re.compile(r'"label"\s*:\s*"(CORRECT|WRONG)"', re.IGNORECASE)
_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_verdict(text: str) -> bool | None:
    """True for CORRECT, False for INCORRECT, None when the output is unparseable."""
    match = _VERDICT.search(text)
    if match is None:
        return None
    return match.group(1).upper() == "CORRECT"


def parse_yes_no(text: str) -> bool | None:
    """LongMemEval's rule: correct when the reply contains "yes". None when the reply has neither
    a yes nor a no, so an empty or cut-off reply is counted as unparseable, not as wrong."""
    if _YES_NO.search(text) is None:
        return None
    return "yes" in text.lower()


def parse_mem0_label(text: str) -> bool | None:
    match = _LABEL.search(text)
    if match is None:
        return None
    return match.group(1).upper() == "CORRECT"


def parse_last_json(text: str) -> dict[str, Any] | None:
    """MemoryAgentBench's parse_json: the last {...} object in the reply."""
    matches = _JSON_OBJECT.findall(text)
    if not matches:
        return None
    try:
        value = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def summary_scores(fluency: str, recall: str, precision: str, keypoints: int) -> dict[str, Any] | None:
    """MemoryAgentBench's scoring: f1 = fluency * 2PR / (P + R), recall over the item's keypoints,
    precision over the summary's sentences. None when any of the three replies is unparseable."""
    f, r, p = parse_last_json(fluency), parse_last_json(recall), parse_last_json(precision)
    try:
        assert f is not None and r is not None and p is not None
        flu = float(f["fluency"])
        found, sentences, supported = float(r["recall"]), float(p["sentence_count"]), float(p["precision"])
    except (AssertionError, KeyError, TypeError, ValueError):
        return None
    rec = found / keypoints if keypoints > 0 else 0.0
    prec = supported / sentences if sentences > 0 else 0.0
    f1 = flu * 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    return {"fluency": flu, "recall": rec, "precision": prec, "f1": f1}


@dataclass
class JudgeResult:
    correct: bool | None
    call: CallResult
    scheme: str = "default"
    scores: dict[str, Any] | None = None


def _rule_call(text: str) -> CallResult:
    return CallResult(
        text=text, usage=Usage(), latency_s=0.0, model_returned=RULE_MODEL, response_id="", cost_usd=0.0
    )


def _combined(calls: list[CallResult]) -> CallResult:
    usage = Usage()
    for c in calls:
        usage = usage + c.usage
    return CallResult(
        text=json.dumps([c.text for c in calls]),
        usage=usage,
        latency_s=sum(c.latency_s for c in calls),
        model_returned=calls[0].model_returned,
        response_id=",".join(c.response_id for c in calls),
        cost_usd=sum(c.cost_usd for c in calls),
        warnings=[w for c in calls for w in c.warnings],
    )


def judge_input(item: Item, model_answer: str) -> list[dict[str, Any]]:
    return [
        {"role": "developer", "content": [{"type": "input_text", "text": JUDGE_INSTRUCTIONS}]},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": judge_prompt(item, model_answer)}],
        },
    ]


def user_only(prompt: str) -> list[dict[str, Any]]:
    """The official graders send their prompt as one user message with no system text."""
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]


class Judge:
    def __init__(self, caller: ModelCaller, cfg: dict[str, Any]) -> None:
        self.caller = caller
        self.cfg = cfg["judge"]

    def _call(self, stage: str, tag: dict[str, Any], items: list[dict[str, Any]], max_out: int) -> CallResult:
        return self.caller.call(
            stage=stage,
            tag={**tag, "purpose": "judge"},
            input_items=items,
            max_output_tokens=max_out,
            reasoning_effort=self.cfg["reasoning_effort"],
        )

    def grade(
        self, *, stage: str, tag: dict[str, Any], item: Item, model_answer: str
    ) -> JudgeResult:
        scheme = judge_scheme(item)
        # The reader's refusal string can never be right for a question that has a gold answer,
        # so it is graded INCORRECT without a model call (and without the model's occasional slip).
        if not item.meta.get("abstention") and normalize(model_answer) == "not mentioned":
            return JudgeResult(correct=False, call=_rule_call('{"verdict": "INCORRECT"}'), scheme=scheme)
        max_out = self.cfg["max_output_tokens"]
        if scheme == "longmemeval":
            call = self._call(stage, tag, user_only(longmemeval_judge_prompt(item, model_answer)), max_out)
            return JudgeResult(correct=parse_yes_no(call.text), call=call, scheme=scheme)
        if scheme == "summary":
            calls = [
                self._call(stage, tag, user_only(p), self.cfg["summary_max_output_tokens"])
                for p in summary_judge_prompts(item, model_answer)
            ]
            keypoints = len(item.meta.get("keypoints") or [])
            scores = summary_scores(calls[0].text, calls[1].text, calls[2].text, keypoints)
            # A summary has no right or wrong verdict; its score is the f1 above.
            return JudgeResult(correct=None, call=_combined(calls), scheme=scheme, scores=scores)
        call = self._call(stage, tag, judge_input(item, model_answer), max_out)
        return JudgeResult(correct=parse_verdict(call.text), call=call, scheme=scheme)

    def grade_mem0(
        self, *, stage: str, tag: dict[str, Any], item: Item, model_answer: str
    ) -> JudgeResult:
        """Secondary LoCoMo grade with Mem0's own prompt (prompts.uses_mem0_judge)."""
        if normalize(model_answer) == "not mentioned":
            return JudgeResult(correct=False, call=_rule_call('{"label": "WRONG"}'), scheme="mem0")
        call = self._call(
            stage, {**tag, "grader": "mem0"}, user_only(mem0_judge_prompt(item, model_answer)),
            self.cfg["max_output_tokens"],
        )
        return JudgeResult(correct=parse_mem0_label(call.text), call=call, scheme="mem0")
