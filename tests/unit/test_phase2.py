import pytest
from pydantic import ValidationError

from scopegraph.backends.scopegraph import ScopeGraphMemorySystem
from scopegraph.graph.in_memory import InMemoryMemoryRepository
from scopegraph.llm.extraction import LLMMemoryExtractor, StaticMemoryExtractor
from scopegraph.llm.scope_classification import resolve_candidate_scope
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.models.memory import (
    MemoryCandidate,
    MemoryStatus,
    MemoryType,
    ScopeLevel,
)
from scopegraph.models.scope import ScopeCreate, ScopeRef, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


class FakeStructuredProvider:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    async def complete_json(self, **_: object) -> dict[str, object]:
        return self.response


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


def test_inferred_candidate_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError, match="inferred"):
        MemoryCandidate(
            **candidate(source_id="m1").model_dump(exclude={"confidence", "inferred"}),
            confidence=0.95,
            inferred=True,
        )


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
