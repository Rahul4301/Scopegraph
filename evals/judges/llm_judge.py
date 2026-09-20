"""Optional structured LLM judge for semantic answer evaluation."""

from typing import Any

from scopegraph.llm.base import StructuredLLMProvider


class LLMJudge:
    def __init__(self, provider: StructuredLLMProvider) -> None:
        self.provider = provider

    async def score(self, answer: str | None, gold_answer: str) -> dict[str, Any]:
        result = await self.provider.complete_json(
            system_prompt=(
                "Judge whether the candidate answer is supported by and equivalent to "
                "the reference. Return a score from 0 to 1 and a brief reason."
            ),
            user_prompt=f"Reference: {gold_answer}\nCandidate: {answer or ''}",
            schema_name="answer_judgment",
            json_schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
                "additionalProperties": False,
            },
        )
        return result
