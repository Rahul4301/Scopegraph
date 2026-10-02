"""Fixed prompt templates. Identical for every arm; only the context body differs.

PROMPT_VERSION is hashed into every run's config hash. Changing any text below means a new
version and a new pre-registration.
"""

from __future__ import annotations

from memstudy.schema import Item

PROMPT_VERSION = "v2"

READER_INSTRUCTIONS = (
    "You answer a question about a conversation or document. The context is the full text or "
    "excerpts from it; sessions carry dates. Use only the context. "
    "Answer in one short phrase or sentence. "
    "For time questions, work out the actual date from the session dates instead of repeating "
    "relative words like \"last week\". "
    "If the context conflicts, use the most recent information. "
    "If the context does not contain the answer, reply exactly: Not mentioned."
)


def context_block(context: str) -> str:
    return f"<context>\n{context}\n</context>"


def question_block(question: str, question_date: str | None) -> str:
    date = f"Current date: {question_date}\n" if question_date else ""
    return f"{date}Question: {question}\nAnswer:"


JUDGE_INSTRUCTIONS = (
    "You grade a model answer against a gold answer. "
    "The answer is CORRECT if it has the core components of the gold answer and means the same "
    "thing. Ignore wording, format, and extra detail. "
    "Dates and numbers match if they are the same value in any format. "
    "A longer answer is fine if it contains the core and does not contradict the gold. "
    "If the gold says the information is not available, the answer is CORRECT only if it also "
    "says so. Otherwise it is INCORRECT. "
    "Reply with only this JSON: {\"verdict\": \"CORRECT\"} or {\"verdict\": \"INCORRECT\"}."
)


def accepted_answers(item: Item) -> list[str]:
    """Distinct accepted answers in order; just the gold when the item lists no others."""
    listed = [str(a) for a in item.meta.get("answers", [])]
    return list(dict.fromkeys(listed)) or [item.gold]


def judge_prompt(item: Item, model_answer: str) -> str:
    golds = accepted_answers(item)
    gold_line = (
        f"Gold answer: {golds[0]}"
        if len(golds) == 1
        else "Gold answers (matching any one is correct): " + " | ".join(golds)
    )
    return (
        f"Question: {item.question}\n"
        f"{gold_line}\n"
        f"Model answer: {model_answer}"
    )
