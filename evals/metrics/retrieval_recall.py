"""Recall@K."""


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 1.0
    return len(set(retrieved_ids[:k]) & gold) / len(gold)
