import re
import unicodedata

from scopegraph.models.memory import MemoryCandidate


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.rstrip(".?!")


def normalize_candidate(candidate: MemoryCandidate) -> MemoryCandidate:
    return candidate.model_copy(
        update={
            "content": candidate.content.strip(),
            "subject": normalize_text(candidate.subject) if candidate.subject else None,
            "predicate": normalize_text(candidate.predicate) if candidate.predicate else None,
            "object": normalize_text(candidate.object) if candidate.object else None,
            "source_message_ids": list(dict.fromkeys(candidate.source_message_ids)),
        }
    )


def candidate_key(candidate: MemoryCandidate) -> str:
    if candidate.subject and candidate.predicate and candidate.object:
        return "|".join((candidate.subject, candidate.predicate, candidate.object))
    return normalize_text(candidate.content)


def conflict_key(candidate: MemoryCandidate) -> str | None:
    if candidate.subject and candidate.predicate:
        return f"{candidate.subject}|{candidate.predicate}"
    return None
