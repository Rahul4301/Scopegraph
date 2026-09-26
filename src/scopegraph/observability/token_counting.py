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
    included_source_ids: set[str] = set()
    for item in items:
        memory_cost = estimate_tokens(item.content)
        if used + memory_cost > token_budget:
            continue
        selected_sources = []
        source_cost = 0
        for source in item.source_messages:
            if source.id in included_source_ids:
                continue
            cost = estimate_tokens(source.content)
            if used + memory_cost + source_cost + cost > token_budget:
                continue
            selected_sources.append(source)
            source_cost += cost
            included_source_ids.add(source.id)
        packed.append(item.model_copy(update={"source_messages": selected_sources}))
        used += memory_cost + source_cost
    return packed, used
