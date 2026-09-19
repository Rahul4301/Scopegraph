import re

from scopegraph.models.retrieval import RetrievedMemory

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def pack_to_token_budget(
    items: list[RetrievedMemory], token_budget: int
) -> tuple[list[RetrievedMemory], int]:
    if token_budget < 1:
        return [], 0
    packed: list[RetrievedMemory] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item.content)
        if used + cost <= token_budget:
            packed.append(item)
            used += cost
    return packed, used
