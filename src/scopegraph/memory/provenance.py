from scopegraph.models.memory import MemoryCandidate


def validate_provenance(candidate: MemoryCandidate, session_message_ids: set[str]) -> None:
    unknown = set(candidate.source_message_ids) - session_message_ids
    if unknown:
        raise ValueError(f"Candidate provenance is outside the ingested session: {sorted(unknown)}")
