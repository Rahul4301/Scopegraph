"""Correction relapse metrics."""


def error_relapse_rate(predicted_answers: list[str], old_error: str) -> float:
    if not predicted_answers:
        return 0.0
    normalized_error = old_error.casefold().strip()
    relapse = sum(normalized_error in answer.casefold() for answer in predicted_answers)
    return relapse / len(predicted_answers)
