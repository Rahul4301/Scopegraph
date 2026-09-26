from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.llm.answering import OpenAICompatibleAnswerer
from scopegraph.llm.extraction import LLMMemoryExtractor, StaticMemoryExtractor
from scopegraph.llm.scope_classification import resolve_candidate_scope
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.models.memory import (
    MemoryCandidate,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from scopegraph.models.retrieval import RetrievedMemory
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessage, SourceMessageCreate


class FakeStructuredProvider:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    async def complete_json(self, **_: object) -> dict[str, object]:
        return self.response


@pytest.mark.asyncio
async def test_answerer_uses_configured_abstention_phrase_and_source_evidence(monkeypatch) -> None:
    answerer = OpenAICompatibleAnswerer(
        base_url="https://example.invalid",
        api_key="test",
        model="test-model",
        unknown_response="No information available",
    )
    captured: dict[str, object] = {}

    async def fake_post(url, *, payload, api_key):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "No information available"}}]}

    monkeypatch.setattr(answerer.transport, "post", fake_post)
    try:
        result = await answerer.generate(
            question="When?",
            context=[RetrievedMemory(
                memory_id="memory",
                content="The launch was rescheduled.",
                score=1.0,
                scope_id="alpha",
                scope_level=ScopeLevel.SCOPE,
                status=MemoryStatus.ACTIVE,
                confidence=1.0,
                source_messages=[SourceMessage(
                    id="source",
                    session_id="session",
                    role=MessageRole.USER,
                    content="The launch moved to Friday at 3 PM.",
                    timestamp=datetime(2023, 5, 8, 13, 0, tzinfo=UTC),
                    turn_index=0,
                )],
            )],
        )
    finally:
        await answerer.aclose()
    assert result == "No information available"
    payload = captured["payload"]
    assert "return No information available" in payload["messages"][0]["content"]
    assert "Verbatim source messages are primary evidence" in payload["messages"][0]["content"]
    assert "The launch moved to Friday at 3 PM." in payload["messages"][1]["content"]


def candidate(
    *,
    source_id: str,
    content: str = "Alpha uses Neo4j",
    object_value: str = "Neo4j",
    level: str = "scope",
    durability: float = 0.9,
    explicit_global: bool = False,
) -> MemoryCandidate:
    return MemoryCandidate(
        content=content,
        memory_type=MemoryType.DECISION,
        subject="Alpha",
        predicate="uses_database",
        object=object_value,
        proposed_scope_level=level,
        confidence=0.95,
        durability=durability,
        source_message_ids=[source_id],
        explicit_global_signal=explicit_global,
    )


def session(session_id: str, scope_id: str, message_id: str, content: str) -> SessionInput:
    return SessionInput(
        id=session_id,
        scope_id=scope_id,
        messages=[
            SourceMessageCreate(
                id=message_id,
                session_id=session_id,
                role=MessageRole.USER,
                content=content,
                turn_index=0,
            )
        ],
    )


@pytest.mark.asyncio
async def test_extractor_rejects_unknown_provenance() -> None:
    provider = FakeStructuredProvider(
        {"candidates": [candidate(source_id="unknown").model_dump(mode="json")]}
    )
    extractor = LLMMemoryExtractor(provider)
    message = SourceMessageCreate(
        id="known",
        session_id="s1",
        role=MessageRole.USER,
        content="Alpha uses Neo4j",
        turn_index=0,
    )
    with pytest.raises(ValueError, match="unknown source"):
        await extractor.extract(
            [message], current_scope=ScopeRef(id="alpha"), existing_memories=[]
        )


@pytest.mark.asyncio
async def test_eval_extractor_discards_only_unknown_provenance(caplog) -> None:
    provider = FakeStructuredProvider({"candidates": [
        candidate(source_id="known").model_dump(mode="json"),
        candidate(source_id="unknown").model_dump(mode="json"),
    ]})
    extractor = LLMMemoryExtractor(provider, skip_invalid_source_ids=True)
    message = SourceMessageCreate(
        id="known",
        session_id="s1",
        role=MessageRole.USER,
        content="Alpha uses Neo4j",
        turn_index=0,
    )
    extracted = await extractor.extract([message], current_scope=ScopeRef(id="alpha"))
    assert [item.source_message_ids for item in extracted] == [["known"]]
    assert "Discarding memory candidate" in caplog.text


def test_inferred_candidate_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError, match="inferred"):
        MemoryCandidate(
            **candidate(source_id="m1").model_dump(exclude={"confidence", "inferred"}),
            confidence=0.95,
            inferred=True,
        )


@pytest.mark.asyncio
async def test_live_extractor_clamps_inferred_confidence() -> None:
    raw = candidate(source_id="known").model_dump(mode="json")
    raw.update(inferred=True, confidence=0.97)
    extractor = LLMMemoryExtractor(FakeStructuredProvider({"candidates": [raw]}))
    message = SourceMessageCreate(
        id="known",
        session_id="s1",
        role=MessageRole.USER,
        content="Alpha probably uses Neo4j",
        turn_index=0,
    )
    extracted = await extractor.extract([message], current_scope=ScopeRef(id="alpha"))
    assert extracted[0].confidence == 0.8
    assert extracted[0].inferred is True


@pytest.mark.asyncio
async def test_live_extractor_preserves_messages_without_extracted_candidates() -> None:
    extracted = candidate(source_id="known").model_dump(mode="json")
    extractor = LLMMemoryExtractor(
        FakeStructuredProvider({"candidates": [extracted]}),
        preserve_unextracted_messages=True,
    )
    messages = [
        SourceMessageCreate(
            id="known",
            session_id="s1",
            role=MessageRole.USER,
            content="Alpha uses Neo4j",
            turn_index=0,
        ),
        SourceMessageCreate(
            id="uncited",
            session_id="s1",
            role=MessageRole.USER,
            content="Alpha moved the meeting to Friday",
            turn_index=1,
        ),
    ]
    result = await extractor.extract(messages, current_scope=ScopeRef(id="alpha"))
    fallback = next(item for item in result if item.source_message_ids == ["uncited"])
    assert fallback.content == "Alpha moved the meeting to Friday"
    assert fallback.memory_type is MemoryType.SUMMARY
    assert fallback.confidence == 0.5


def test_explicit_scope_wins_and_unapproved_global_is_downscoped() -> None:
    decision = resolve_candidate_scope(
        candidate(source_id="m1", level="global", explicit_global=False),
        current_scope=ScopeRef(id="alpha", scope_type=ScopeType.PROJECT),
        global_scope_id="global",
    )
    assert decision.scope_id == "alpha"
    assert decision.scope_level == "scope"


@pytest.mark.asyncio
async def test_complete_ingestion_preserves_provenance_and_deduplicates() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(
            id="alpha",
            name="Alpha",
            scope_type=ScopeType.PROJECT,
            parent_scope_id="global",
        )
    )
    first = ScopeGraphMemorySystem(
        repository, StaticMemoryExtractor([candidate(source_id="m1")])
    )
    result = await first.ingest_session(
        session("s1", "alpha", "m1", "Alpha uses Neo4j"),
        current_scope=ScopeRef(id="alpha", scope_type=ScopeType.PROJECT),
    )
    memory = await repository.get_memory(result.memory_ids[0])
    assert memory is not None
    assert memory.source_ids == ["m1"]
    assert memory.scope_level is ScopeLevel.SCOPE

    second = ScopeGraphMemorySystem(
        repository, StaticMemoryExtractor([candidate(source_id="m2")])
    )
    duplicate = await second.ingest_session(
        session("s2", "alpha", "m2", "Alpha uses Neo4j"),
        current_scope=ScopeRef(id="alpha", scope_type=ScopeType.PROJECT),
    )
    assert duplicate.duplicate_count == 1
    assert len(await repository.list_memories(scope_id="alpha")) == 1


@pytest.mark.asyncio
async def test_historical_chat_timestamps_survive_ingestion_and_close_session() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    started_at = datetime(2023, 5, 8, 13, 0, tzinfo=UTC)
    last_message_at = datetime(2023, 5, 8, 13, 5, tzinfo=UTC)
    system = ScopeGraphMemorySystem(repository, StaticMemoryExtractor([]))

    await system.ingest_session(
        SessionInput(
            id="historical",
            scope_id="alpha",
            started_at=started_at,
            messages=[SourceMessageCreate(
                id="historical-message",
                session_id="historical",
                role=MessageRole.USER,
                content="Alpha uses Neo4j",
                timestamp=last_message_at,
                turn_index=0,
            )],
        ),
        current_scope=ScopeRef(id="alpha"),
    )

    stored_messages = await repository.list_source_messages("historical")
    stored_session = await repository.get_session("historical")
    assert stored_messages[0].timestamp == last_message_at
    assert stored_session is not None
    assert stored_session.started_at == started_at
    assert stored_session.ended_at == last_message_at


@pytest.mark.asyncio
async def test_low_durability_memory_stays_at_session_level() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(id="beta", name="Beta", scope_type=ScopeType.PROJECT)
    )
    system = ScopeGraphMemorySystem(
        repository,
        StaticMemoryExtractor(
            [candidate(source_id="m1", content="Testing SQLite", durability=0.2)]
        ),
    )
    result = await system.ingest_session(
        session("s1", "beta", "m1", "Temporarily test SQLite"),
        current_scope=ScopeRef(id="beta", scope_type=ScopeType.PROJECT),
    )
    memory = await repository.get_memory(result.memory_ids[0])
    assert memory is not None
    assert memory.scope_level is ScopeLevel.SESSION


@pytest.mark.asyncio
async def test_conflict_supersedes_old_memory_without_deleting_history() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    first = ScopeGraphMemorySystem(
        repository,
        StaticMemoryExtractor(
            [candidate(source_id="m1", content="Alpha uses MongoDB", object_value="MongoDB")]
        ),
    )
    first_result = await first.ingest_session(
        session("s1", "alpha", "m1", "Alpha uses MongoDB"),
        current_scope=ScopeRef(id="alpha"),
    )
    second = ScopeGraphMemorySystem(
        repository,
        StaticMemoryExtractor(
            [candidate(source_id="m2", content="Alpha uses PostgreSQL", object_value="PostgreSQL")]
        ),
    )
    second_result = await second.ingest_session(
        session("s2", "alpha", "m2", "Alpha switched to PostgreSQL"),
        current_scope=ScopeRef(id="alpha"),
    )
    old = await repository.get_memory(first_result.memory_ids[0])
    new = await repository.get_memory(second_result.memory_ids[0])
    assert old is not None and new is not None
    assert old.status is MemoryStatus.SUPERSEDED
    assert old.valid_to is not None
    assert (new.id, old.id) in repository.supersedes
    assert second_result.conflict_count == 1


@pytest.mark.asyncio
async def test_session_override_does_not_supersede_scope_memory() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    durable = ScopeGraphMemorySystem(
        repository,
        StaticMemoryExtractor(
            [candidate(source_id="m1", content="Alpha uses MongoDB", object_value="MongoDB")]
        ),
    )
    await durable.ingest_session(
        session("s1", "alpha", "m1", "Alpha uses MongoDB"),
        current_scope=ScopeRef(id="alpha"),
    )
    temporary = ScopeGraphMemorySystem(
        repository,
        StaticMemoryExtractor(
            [candidate(
                source_id="m2", content="Temporarily use SQLite", object_value="SQLite",
                level="session", durability=0.1,
            )]
        ),
    )
    result = await temporary.ingest_session(
        session("s2", "alpha", "m2", "Temporarily use SQLite"),
        current_scope=ScopeRef(id="alpha", session_id="s2"),
    )

    memories = await repository.list_memories(scope_id="alpha", include_inactive=True)
    durable_memory = next(
        memory for memory in memories if memory.metadata.get("object") == "mongodb"
    )
    temporary_memory = next(
        memory for memory in memories if memory.metadata.get("object") == "sqlite"
    )
    assert result.conflict_count == 0
    assert durable_memory.status is MemoryStatus.ACTIVE
    assert temporary_memory.scope_level is ScopeLevel.SESSION


@pytest.mark.asyncio
async def test_memory_validity_starts_at_source_message_timestamp() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    timestamp = datetime(2026, 6, 10, 10, 0, tzinfo=UTC)
    message = SourceMessageCreate(
        id="m1", session_id="s1", role=MessageRole.USER,
        content="Alpha uses MongoDB", turn_index=0, timestamp=timestamp,
    )
    system = ScopeGraphMemorySystem(
        repository,
        StaticMemoryExtractor([candidate(source_id="m1", object_value="MongoDB")]),
    )
    result = await system.ingest_session(
        SessionInput(id="s1", scope_id="alpha", messages=[message]),
        current_scope=ScopeRef(id="alpha"),
    )
    memory = await repository.get_memory(result.memory_ids[0])
    assert memory is not None
    assert memory.valid_from == timestamp


@pytest.mark.asyncio
async def test_global_promotion_requires_three_distinct_scopes() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    for index in range(3):
        scope_id = f"project-{index}"
        message_id = f"m-{index}"
        await repository.create_scope(
            ScopeCreate(id=scope_id, name=scope_id, scope_type=ScopeType.PROJECT)
        )
        memory_candidate = candidate(
            source_id=message_id,
            content="User prefers concise answers",
            object_value="concise answers",
        ).model_copy(update={"subject": "user", "predicate": "prefers"})
        system = ScopeGraphMemorySystem(
            repository,
            StaticMemoryExtractor([memory_candidate]),
            promotion_policy=PromotionPolicy(minimum_distinct_scopes=3),
        )
        result = await system.ingest_session(
            session(f"s-{index}", scope_id, message_id, "Please keep answers concise"),
            current_scope=ScopeRef(id=scope_id),
        )
        assert result.promoted_count == (1 if index == 2 else 0)

    global_memories = await repository.list_memories(scope_id="global")
    assert len(global_memories) == 1
    assert global_memories[0].scope_level is ScopeLevel.GLOBAL
    assert len(global_memories[0].source_ids) == 3


@pytest.mark.asyncio
async def test_explicit_global_statement_is_stored_globally() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL)
    )
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    explicit = candidate(
        source_id="m1",
        content="User generally prefers PostgreSQL",
        object_value="PostgreSQL",
        level="global",
        explicit_global=True,
    ).model_copy(update={"subject": "user", "predicate": "prefers"})
    system = ScopeGraphMemorySystem(repository, StaticMemoryExtractor([explicit]))
    result = await system.ingest_session(
        session("s1", "alpha", "m1", "I generally prefer PostgreSQL"),
        current_scope=ScopeRef(id="alpha"),
    )
    memory = await repository.get_memory(result.memory_ids[0])
    assert memory is not None
    assert memory.scope_id == "global"
    assert memory.scope_level is ScopeLevel.GLOBAL


@pytest.mark.asyncio
async def test_structured_entity_bridge_creates_traversable_relation() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="alpha", name="Alpha", scope_type=ScopeType.PROJECT)
    )
    depends = candidate(
        source_id="m1", content="Alpha depends on worker-42", object_value="worker-42"
    ).model_copy(update={"subject": "alpha", "predicate": "depends_on"})
    uses = candidate(
        source_id="m2", content="worker-42 uses Neo4j", object_value="Neo4j"
    ).model_copy(update={"subject": "worker-42", "predicate": "uses_database"})
    first = ScopeGraphMemorySystem(repository, StaticMemoryExtractor([depends]))
    first_result = await first.ingest_session(
        session("s1", "alpha", "m1", depends.content),
        current_scope=ScopeRef(id="alpha"),
    )
    second = ScopeGraphMemorySystem(repository, StaticMemoryExtractor([uses]))
    second_result = await second.ingest_session(
        session("s2", "alpha", "m2", uses.content),
        current_scope=ScopeRef(id="alpha"),
    )
    neighbors = await repository.get_memory_neighbors(first_result.memory_ids)
    assert any(
        neighbor.memory.id in second_result.memory_ids
        and neighbor.relation == "RELATES_TO:DEPENDS_ON"
        for neighbor in neighbors
    )


@pytest.mark.asyncio
async def test_retrospective_event_ending_before_message_is_ingested() -> None:
    repository = InMemoryMemoryRepository()
    await repository.create_scope(
        ScopeCreate(id="beta", name="Beta", scope_type=ScopeType.PROJECT)
    )
    ended = datetime(2023, 6, 26, 23, 59, tzinfo=UTC)
    event = candidate(source_id="m1", content="Went camping last week").model_copy(
        update={"memory_type": MemoryType.EVENT, "valid_to": ended}
    )
    system = ScopeGraphMemorySystem(repository, StaticMemoryExtractor([event]))
    result = await system.ingest_session(
        session("s1", "beta", "m1", "We went camping last week"),
        current_scope=ScopeRef(id="beta", scope_type=ScopeType.PROJECT),
    )
    memory = await repository.get_memory(result.memory_ids[0])
    assert memory is not None
    assert memory.valid_from is None
    assert memory.valid_to == ended
