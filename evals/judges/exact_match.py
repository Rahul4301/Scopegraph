"""Credential-free deterministic answer judge."""

from evals.metrics.answer_accuracy import exact_match, token_f1


class ExactMatchJudge:
    def score(self, answer: str | None, gold_answer: str) -> dict[str, float]:
        return {"exact_match": exact_match(answer, gold_answer),
                "token_f1": token_f1(answer, gold_answer)}
