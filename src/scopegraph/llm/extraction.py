import json
from typing import Protocol

from scopegraph.llm.base import StructuredLLMProvider
from scopegraph.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from scopegraph.models.memory import MemoryCandidate, MemoryCandidateBatch
from scopegraph.models.scope import ScopeRef
from scopegraph.models.source import SourceMessage


class CandidateExtractor(Protocol):
    async def extract(
        self,
        messages: list[SourceMessage],
        *,
        current_scope: ScopeRef | None,
        existing_memories: list[str] | None = None,
    ) -> list[MemoryCandidate]: ...


class LLMMemoryExtractor:
    def __init__(self, provider: StructuredLLMProvider) -> None:
        self.provider = provider

    async def extract(
        self,
        messages: list[SourceMessage],
        *,
        current_scope: ScopeRef | None,
        existing_memories: list[str] | None = None,
    ) -> list[MemoryCandidate]:
        prompt_payload = {
            "current_scope": current_scope.model_dump(mode="json") if current_scope else None,
            "messages": [message.model_dump(mode="json") for message in messages],
            "existing_memories": existing_memories or [],
        }
        raw = await self.provider.complete_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=json.dumps(prompt_payload, default=str),
            schema_name="memory_candidates",
            json_schema=MemoryCandidateBatch.model_json_schema(),
        )
        batch = MemoryCandidateBatch.model_validate(raw)
        valid_source_ids = {message.id for message in messages}
        for candidate in batch.candidates:
            unknown = set(candidate.source_message_ids) - valid_source_ids
            if unknown:
                raise ValueError(
                    f"Extractor returned unknown source message IDs: {sorted(unknown)}"
                )
        return batch.candidates


class StaticMemoryExtractor:
    """Deterministic fake used by tests and offline demonstrations."""

    def __init__(self, candidates: list[MemoryCandidate]) -> None:
        self.candidates = candidates

    async def extract(
        self,
        messages: list[SourceMessage],
        *,
        current_scope: ScopeRef | None,
        existing_memories: list[str] | None = None,
    ) -> list[MemoryCandidate]:
        del current_scope, existing_memories
        valid_source_ids = {message.id for message in messages}
        for candidate in self.candidates:
            if not set(candidate.source_message_ids) <= valid_source_ids:
                raise ValueError("Static extractor candidate references an unknown source message")
        return list(self.candidates)
