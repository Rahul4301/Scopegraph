"""Fixed prompt templates. Identical for every arm; only the context body differs.

PROMPT_VERSION is hashed into every run's config hash. Changing any text below means a new
version and a new pre-registration.
"""

from __future__ import annotations

from memstudy.schema import Item

PROMPT_VERSION = "v1"

READER_INSTRUCTIONS = (
    "You answer a question using only the context provided. "
    "Answer in one short phrase or sentence. "
    "If the context does not contain the answer, reply exactly: Not mentioned."
)


def context_block(context: str) -> str:
    return f"<context>\n{context}\n</context>"


def question_block(question: str, question_date: str | None) -> str:
    date = f"Current date: {question_date}\n" if question_date else ""
    return f"{date}Question: {question}\nAnswer:"


JUDGE_INSTRUCTIONS = (
    "You grade an answer to a question about a conversation history. "
    "Compare the model answer with the gold answer. "
    "Reply with only this JSON: {\"verdict\": \"CORRECT\"} or {\"verdict\": \"INCORRECT\"}."
)

_GUIDANCE_DEFAULT = (
    "The answer is CORRECT if it contains the key information of the gold answer. "
    "Extra detail is fine. A contradiction of the gold answer is INCORRECT."
)
_GUIDANCE_TEMPORAL = (
    "Dates and durations may be written in any format. For counts of days, weeks, or months, "
    "an off-by-one answer is CORRECT. Otherwise follow the default rule: "
    + _GUIDANCE_DEFAULT
)
_GUIDANCE_UPDATE = (
    "The gold answer is the most recent value. The answer is CORRECT only if it gives the most "
    "recent value as the current one. Mentioning an older value as past is fine."
)
_GUIDANCE_PREFERENCE = (
    "The gold text describes what a good personalized answer should take into account, not an "
    "exact string. The answer is CORRECT if it uses the user's stated preferences as described."
)
_GUIDANCE_ABSTAIN = (
    "The information is not in the conversation. The answer is CORRECT only if it says the "
    "information is not mentioned or not available. Any specific claim is INCORRECT."
)

TEMPORAL_CATEGORIES = {"cat2", "temporal-reasoning"}


def judge_guidance(item: Item) -> str:
    if item.meta.get("abstention"):
        return _GUIDANCE_ABSTAIN
    if item.category == "knowledge-update":
        return _GUIDANCE_UPDATE
    if item.category == "single-session-preference":
        return _GUIDANCE_PREFERENCE
    if item.category in TEMPORAL_CATEGORIES:
        return _GUIDANCE_TEMPORAL
    return _GUIDANCE_DEFAULT


def judge_prompt(item: Item, model_answer: str) -> str:
    return (
        f"{judge_guidance(item)}\n\n"
        f"Question: {item.question}\n"
        f"Gold answer: {item.gold}\n"
        f"Model answer: {model_answer}"
    )
