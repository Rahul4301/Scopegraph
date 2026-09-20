"""Common answer-judge protocol."""

from typing import Protocol


class AnswerJudge(Protocol):
    def score(self, answer: str | None, gold_answer: str) -> dict[str, float]: ...
