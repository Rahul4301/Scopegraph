"""Fixed prompt templates. Identical for every arm; only the context body differs.

PROMPT_VERSION is hashed into every run's config hash. Changing any text below means a new
version and a new pre-registration.
"""

from __future__ import annotations

from memstudy.schema import Item

PROMPT_VERSION = "v5"

READER_INSTRUCTIONS = (
    "You answer a question about a conversation or document. The context is the full text or "
    "excerpts from it; sessions carry dates. Use only the context, but when the answer is not "
    "stated outright, give your best inference from what the context says or implies. "
    "Answer in one short phrase or sentence. "
    "For time questions, work out the actual date from the session dates instead of repeating "
    "relative words like \"last week\". "
    "If the context conflicts, use the most recent information. "
    "Reply exactly \"Not mentioned\" only when the context gives no basis at all for an answer."
)


def context_block(context: str) -> str:
    return f"<context>\n{context}\n</context>"


def question_block(question: str, question_date: str | None) -> str:
    date = f"Current date: {question_date}\n" if question_date else ""
    return f"{date}Question: {question}\nAnswer:"


JUDGE_INSTRUCTIONS = (
    "You grade a model answer against a gold answer for a question about a conversation. "
    "The answer is CORRECT if it conveys the same core information as the gold answer. "
    "Rules: "
    "Ignore wording, format, and extra detail. Extra information in the answer is never a reason "
    "for INCORRECT: a longer answer is CORRECT if it contains the core of the gold and does not "
    "contradict it. "
    "If the gold lists several items, the answer must include all of them (extra items are fine). "
    "Dates, numbers, and durations match when they are the same value in any form "
    "(\"7 May 2023\" = \"May 7, 2023\"; \"three\" = 3). "
    "A relative gold date such as \"the Friday before 15 July 2023\" means one calendar date "
    "(Friday, 14 July 2023); an answer that gives that date is CORRECT, so work out the date "
    "before deciding. "
    "For a \"likely\" or open-ended gold, the answer is CORRECT if it reaches the same "
    "conclusion without contradicting the gold. "
    "An answer that says the information is missing or not mentioned is INCORRECT unless the gold "
    "itself says the information is not available. "
    "A different or contradicting answer is INCORRECT. "
    "Reply with only this JSON, the reason first: "
    "{\"reason\": \"<one short sentence comparing the answer to the gold>\", "
    "\"verdict\": \"CORRECT\"} or the same with \"INCORRECT\"."
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
