import json
import logging
from typing import Protocol

from scopegraph.llm.base import StructuredLLMProvider
from scopegraph.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from scopegraph.models.memory import MemoryCandidate, MemoryCandidateBatch, MemoryType
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
    def __init__(
        self,
        provider: StructuredLLMProvider,
        *,
        skip_invalid_source_ids: bool = False,
        preserve_unextracted_messages: bool = False,
    ) -> None:
        self.provider = provider
        self.skip_invalid_source_ids = skip_invalid_source_ids
        self.preserve_unextracted_messages = preserve_unextracted_messages

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
        # JSON Schema cannot express the cross-field invariant enforced by the
        # Pydantic model. Clamp model-produced inferred confidence at the
        # documented ceiling before strict validation.
        candidates = raw.get("candidates")
        if isinstance(candidates, list):
            for item in candidates:
                if (
                    isinstance(item, dict)
                    and item.get("inferred") is True
                    and isinstance(item.get("confidence"), int | float)
                ):
                    item["confidence"] = min(float(item["confidence"]), 0.8)
        batch = MemoryCandidateBatch.model_validate(raw)
        valid_source_ids = {message.id for message in messages}
        accepted: list[MemoryCandidate] = []
        for candidate in batch.candidates:
            unknown = set(candidate.source_message_ids) - valid_source_ids
            if unknown:
                if self.skip_invalid_source_ids:
                    logging.warning(
                        "Discarding memory candidate with unknown source message IDs: %s",
                        sorted(unknown),
                    )
                    continue
                raise ValueError(
                    f"Extractor returned unknown source message IDs: {sorted(unknown)}"
                )
            accepted.append(candidate)
        if self.preserve_unextracted_messages:
            cited_source_ids = {
                source_id
                for candidate in accepted
                for source_id in candidate.source_message_ids
            }
            accepted.extend(
                MemoryCandidate(
                    content=message.content,
                    memory_type=MemoryType.SUMMARY,
                    proposed_scope_level="scope",
                    confidence=0.5,
                    durability=0.8,
                    source_message_ids=[message.id],
                )
                for message in messages
                if message.id not in cited_source_ids
            )
        return accepted


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
