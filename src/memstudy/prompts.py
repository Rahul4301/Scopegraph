"""Fixed prompt templates. Identical for every arm; only the context body differs.

PROMPT_VERSION is hashed into every run's config hash. Changing any text below means a new
version and a new pre-registration.
"""

from __future__ import annotations

from memstudy import official_prompts as official
from memstudy.schema import Item

PROMPT_VERSION = "v7"

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


# MemoryAgentBench sources each want a different answer shape. One extra line per source, adapted
# from the benchmark's own query templates (utils/templates.py); it goes after the context so the
# cached prefix is unchanged. Keys are matched as prefixes of the item's source.
TASK_INSTRUCTIONS: dict[str, str] = {
    "ruler_qa": "Only give the answer and do not output any other words.",
    "eventqa": "Choose from the listed possible events the one that happens next, and reply with "
    "that event copied exactly as listed, without anything else.",
    "factconsolidation": "Each fact has a serial number, and a fact with a larger serial number is "
    "newer and replaces any older fact it conflicts with. Answer from the facts in the context, "
    "not from real-world knowledge, with a very concise answer.",
    "icl_": "The context maps example texts to numerical labels. Assign a numerical label to the "
    "text in the question using that mapping. Only output \"label: <number>\" and nothing else.",
    "detective_qa": "Answer the multiple-choice question in the output format shown in the "
    "question, giving the option letter and its text.",
    "infbench_sum": "This task overrides the short-answer rule: write the summary the question asks "
    "for, about 1,000 to 1,200 words on the plot and characters.",
}


def task_key(item: Item) -> str | None:
    source = str(item.meta.get("source", ""))
    return next((key for key in TASK_INSTRUCTIONS if source.startswith(key)), None)


def reader_task_key(item: Item) -> str | None:
    """The task line the reader adds. None when `run --task_prompts` already wrapped the question
    in the benchmark's own template, so the instructions are never given twice."""
    return None if item.meta.get("task_prompt") else task_key(item)


def task_instruction(item: Item) -> str | None:
    key = task_key(item)
    return TASK_INSTRUCTIONS[key] if key else None


def question_block(question: str, question_date: str | None, task: str | None = None) -> str:
    date = f"Current date: {question_date}\n" if question_date else ""
    note = f"Task format: {task}\n" if task else ""
    return f"{date}{note}Question: {question}\nAnswer:"


JUDGE_INSTRUCTIONS = (
    "You grade a model answer against a gold answer for a question about a conversation. "
    "The answer is CORRECT if it conveys the same core information as the gold answer. "
    "Rules: "
    "Ignore wording, format, and extra detail. Extra information in the answer is never a reason "
    "for INCORRECT: a longer answer is CORRECT if it contains the core of the gold and does not "
    "contradict it. "
    "If the gold lists several items and the question asks for all of them (a plural question such "
    "as \"What are her pets' names?\"), the answer must include all of them; extra items are fine. "
    "If the question asks for one thing or an example (such as \"What is a stress reliever for "
    "him?\") and the gold lists several acceptable ones, an answer that gives at least one of them "
    "without contradicting the rest is CORRECT. "
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


# Exploratory, not registered: MemoryAgentBench's own query templates for the two splits where
# the generic reader prompt does not state the task (fact consolidation, label classification).
# Copied verbatim from the benchmark's utils/templates.py (str.format with {question}); arm A
# gets the long-context template, arms B, C and D the RAG template. Opt-in with
# `run --task_prompts`, which enters the config hash.
TASK_PROMPTS: dict[str, dict[str, str]] = {
    'factconsolidation_': {
        'long_context_agent': (
            'Pretend you are a knowledge management system. Each fact in the knowledge pool is '
            'provided with a serial number at the beginning, and the newer fact has larger serial '
            'number. \n You need to solve the conflicts of facts in the knowledge pool by finding '
            'the newest fact with larger serial number. You need to answer a question based on '
            'this rule. You should give a very concise answer without saying other words for '
            'the question **only** from the knowledge pool you have memorized rather than the '
            'real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on '
            'the provided Knowledge Pool, what is the name of the current president of Russia? '
            '\nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge '
            'Pool, {question} \nAnswer:'
        ),
        'rag_agent': (
            'Pretend you are a knowledge management system. Each fact in the knowledge pool is '
            'provided with a serial number at the beginning, and the newer fact has larger serial '
            'number. \n You need to solve the conflicts of facts in the knowledge pool by finding '
            'the newest fact with larger serial number. You need to answer a question based on '
            'this rule. You should give a very concise answer without saying other words for '
            'the question **only** from the knowledge pool you have memorized rather than the '
            'real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on '
            'the provided Knowledge Pool, what is the name of the current president of Russia? '
            '\nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge '
            'Pool, {question} \nAnswer:'
        ),
    },
    'icl_': {
        'long_context_agent': (
            'Use the provided mapping from the context to numerical label to assign a numerical '
            'label to the context. Only output "label: {{label}}" and nothing else. \n\n{question} '
            '\n\n label:'
        ),
        'rag_agent': (
            'Use the provided mapping from the context to numerical label to assign a numerical '
            'label to the context. Only output "label: {{label}}" and nothing else. \n\nQuestion:{question} '
            '\n\n label:'
        ),
    },
}


# Which grader an item gets. LongMemEval questions (and the composite histories built from them)
# use LongMemEval's own per-type prompts; InfBench summaries use MemoryAgentBench's summary
# grader; everything else uses the JUDGE_INSTRUCTIONS rule above.
LONGMEMEVAL_TEMPLATES = {
    "single-session-user": official.LME_DEFAULT,
    "single-session-assistant": official.LME_DEFAULT,
    "multi-session": official.LME_DEFAULT,
    "temporal-reasoning": official.LME_TEMPORAL,
    "knowledge-update": official.LME_KNOWLEDGE_UPDATE,
    "single-session-preference": official.LME_PREFERENCE,
}


def judge_scheme(item: Item) -> str:
    if item.bench in ("longmemeval", "composite") and item.category in LONGMEMEVAL_TEMPLATES:
        return "longmemeval"
    if task_key(item) == "infbench_sum":
        return "summary"
    return "default"


def longmemeval_judge_prompt(item: Item, model_answer: str) -> str:
    template = (
        official.LME_ABSTENTION
        if item.meta.get("abstention")
        else LONGMEMEVAL_TEMPLATES[item.category]
    )
    return template.format(item.question, item.gold, model_answer)


def uses_mem0_judge(item: Item) -> bool:
    """Secondary LoCoMo grade with Mem0's prompt, for comparison with published numbers only.
    Mem0 and Zep never graded category 5, so neither does this."""
    return item.bench == "locomo" and not item.meta.get("abstention")


def mem0_judge_prompt(item: Item, model_answer: str) -> str:
    return official.MEM0_ACCURACY_PROMPT.format(
        question=item.question, gold_answer=item.gold, generated_answer=model_answer
    )


def summary_judge_prompts(item: Item, summary: str) -> tuple[str, str, str]:
    """Fluency, keypoint recall and sentence precision prompts, as MemoryAgentBench builds them."""
    keypoints = [str(k) for k in item.meta.get("keypoints") or []]
    text = summary.strip()
    return (
        official.FLUENCY_PROMPT_BOOK.format(text=text),
        official.RECALL_PROMPT_BOOK.format(
            keypoints="\n".join(f"{i + 1}. {kp}" for i, kp in enumerate(keypoints)), summary=text
        ),
        official.PRECISION_PROMPT_BOOK.format(expert_summary=item.gold, summary=text),
    )
