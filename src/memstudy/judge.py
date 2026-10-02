"""Judge: GPT-5 nano at the lowest reasoning effort, one fixed prompt for every arm."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memstudy.llm import CallResult, ModelCaller
from memstudy.prompts import JUDGE_INSTRUCTIONS, judge_prompt
from memstudy.schema import Item, Usage
from memstudy.scoring import normalize

RULE_MODEL = "rule:not_mentioned"

_VERDICT = re.compile(r'"verdict"\s*:\s*"(CORRECT|INCORRECT)"', re.IGNORECASE)


def parse_verdict(text: str) -> bool | None:
    """True for CORRECT, False for INCORRECT, None when the output is unparseable."""
    match = _VERDICT.search(text)
    if match is None:
        return None
    return match.group(1).upper() == "CORRECT"


@dataclass
class JudgeResult:
    correct: bool | None
    call: CallResult


def judge_input(item: Item, model_answer: str) -> list[dict[str, Any]]:
    return [
        {"role": "developer", "content": [{"type": "input_text", "text": JUDGE_INSTRUCTIONS}]},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": judge_prompt(item, model_answer)}],
        },
    ]


class Judge:
    def __init__(self, caller: ModelCaller, cfg: dict[str, Any]) -> None:
        self.caller = caller
        self.cfg = cfg["judge"]

    def grade(
        self, *, stage: str, tag: dict[str, Any], item: Item, model_answer: str
    ) -> JudgeResult:
        # The reader's refusal string can never be right for a question that has a gold answer,
        # so it is graded INCORRECT without a model call (and without the model's occasional slip).
        if not item.meta.get("abstention") and normalize(model_answer) == "not mentioned":
            verdict = CallResult(
                text='{"verdict": "INCORRECT"}',
                usage=Usage(),
                latency_s=0.0,
                model_returned=RULE_MODEL,
                response_id="",
                cost_usd=0.0,
            )
            return JudgeResult(correct=False, call=verdict)
        call = self.caller.call(
            stage=stage,
            tag={**tag, "purpose": "judge"},
            input_items=judge_input(item, model_answer),
            max_output_tokens=self.cfg["max_output_tokens"],
            reasoning_effort=self.cfg["reasoning_effort"],
        )
        return JudgeResult(correct=parse_verdict(call.text), call=call)
