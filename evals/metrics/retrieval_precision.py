"""Precision@K."""


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    gold = set(gold_ids)
    return sum(memory_id in gold for memory_id in retrieved) / len(retrieved)
