"""Reader: GPT-6 Luna, one prompt template for every arm.

MemoryAgentBench items add one task-format line for their source (prompts.TASK_INSTRUCTIONS),
the same for every arm; a source may also have a larger output allowance in the config."""

from __future__ import annotations

from typing import Any

from memstudy.llm import CallResult, ModelCaller
from memstudy.prompts import READER_INSTRUCTIONS, TASK_INSTRUCTIONS, context_block, question_block


def build_input(
    context: str,
    question: str,
    question_date: str | None,
    breakpoint_after_context: bool,
    task: str | None = None,
) -> list[dict[str, Any]]:
    """History or retrieved context first, question last.

    With a breakpoint, the developer message and the context form a cacheable prefix that every
    later question on the same history reuses.
    """
    ctx: dict[str, Any] = {"type": "input_text", "text": context_block(context)}
    if breakpoint_after_context:
        ctx["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": READER_INSTRUCTIONS}],
        },
        {
            "role": "user",
            "content": [ctx, {"type": "input_text", "text": question_block(question, question_date, task)}],
        },
    ]


class Reader:
    def __init__(self, caller: ModelCaller, cfg: dict[str, Any]) -> None:
        self.caller = caller
        self.cfg = cfg["reader"]

    def answer(
        self,
        *,
        stage: str,
        tag: dict[str, Any],
        context: str,
        question: str,
        question_date: str | None,
        cache_prefix: bool,
        cache_key: str,
        task_key: str | None = None,
    ) -> CallResult:
        task = TASK_INSTRUCTIONS[task_key] if task_key else None
        max_output = self.cfg.get("max_output_tokens_by_task", {}).get(task_key, self.cfg["max_output_tokens"])
        return self.caller.call(
            stage=stage,
            tag={**tag, "purpose": "read"},
            input_items=build_input(context, question, question_date, cache_prefix, task),
            max_output_tokens=max_output,
            reasoning_effort=self.cfg["reasoning_effort"],
            temperature=self.cfg["temperature"],
            extra_body={
                "prompt_cache_key": cache_key,
                "prompt_cache_options": {"mode": "explicit", "ttl": self.cfg["cache_ttl"]},
            },
        )
