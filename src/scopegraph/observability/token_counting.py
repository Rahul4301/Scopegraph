import re

from scopegraph.models.retrieval import RetrievedMemory

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def pack_to_token_budget(
    items: list[RetrievedMemory], token_budget: int, *, query: str | None = None
) -> tuple[list[RetrievedMemory], int]:
    if token_budget < 1:
        return [], 0
    packed: list[RetrievedMemory] = []
    used = 0
    included_source_ids: set[str] = set()
    included_texts: set[str] = set()
    for item in items:
        memory_cost = estimate_tokens(item.content)
        if used + memory_cost > token_budget:
            continue
        selected_sources = []
        source_cost = 0
        for source in item.source_messages:
            if source.id in included_source_ids:
                continue
            content = _relevant_span(source.content, query, max_tokens=220)
            normalized = " ".join(content.casefold().split())
            if normalized in included_texts:
                included_source_ids.add(source.id)
                continue
            cost = (
                0 if normalized == " ".join(item.content.casefold().split())
                else estimate_tokens(content)
            )
            if used + memory_cost + source_cost + cost > token_budget:
                continue
            selected_sources.append(source.model_copy(update={"content": content}))
            source_cost += cost
            included_source_ids.add(source.id)
            included_texts.add(normalized)
        packed.append(item.model_copy(update={"source_messages": selected_sources}))
        included_texts.add(" ".join(item.content.casefold().split()))
        used += memory_cost + source_cost
    return packed, used


def _relevant_span(text: str, query: str | None, *, max_tokens: int) -> str:
    tokens = TOKEN_PATTERN.findall(text)
    if len(tokens) <= max_tokens:
        return text
    query_terms = {
        term.casefold() for term in TOKEN_PATTERN.findall(query or "")
        if term.isalnum() and len(term) > 2
    }
    lowered = [token.casefold() for token in tokens]
    hits = [index for index, token in enumerate(lowered) if token in query_terms]
    center = hits[0] if hits else 0
    start = max(0, min(center - max_tokens // 3, len(tokens) - max_tokens))
    prefix = "… " if start else ""
    suffix = " …" if start + max_tokens < len(tokens) else ""
    return prefix + " ".join(tokens[start : start + max_tokens]) + suffix
