"""Deterministic CrossScopeMem scenario generation."""

import random
from datetime import UTC, datetime, timedelta

from evals.schemas import BenchmarkExample, CrossScopeScenario
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate

_PROJECTS = (
    ("alpha", "Alpha", "Rust", "Neo4j", "React"),
    ("beta", "Beta", "Python", "MongoDB", "Vue"),
    ("gamma", "Gamma", "TypeScript", "PostgreSQL", "Svelte"),
    ("delta", "Delta", "Go", "Redis", "Solid"),
    ("epsilon", "Epsilon", "Java", "MySQL", "Angular"),
    ("zeta", "Zeta", "Kotlin", "SQLite", "Compose"),
    ("eta", "Eta", "Clojure", "Datomic", "Reagent"),
    ("theta", "Theta", "Swift", "CockroachDB", "SwiftUI"),
    ("iota", "Iota", "Elixir", "PostgreSQL", "LiveView"),
)


def _candidate(message: SourceMessageCreate, *, subject: str, predicate: str, value: str,
               level: str, memory_type: MemoryType = MemoryType.FACT,
               durability: float = 1.0) -> MemoryCandidate:
    return MemoryCandidate(
        content=message.content,
        memory_type=memory_type,
        subject=subject,
        predicate=predicate,
        object=value,
        proposed_scope_level=level,  # type: ignore[arg-type]
        confidence=0.98,
        durability=durability,
        source_message_ids=[message.id],
    )


def generate_cross_scope_mem(*, seed: int = 42, difficulty: int = 2,
                             scenario_id: str | None = None) -> CrossScopeScenario:
    """Generate one reproducible world and its gold retrieval questions."""
    if difficulty not in {1, 2, 3, 4}:
        raise ValueError("difficulty must be between 1 and 4")
    rng = random.Random(seed)
    project_count = {1: 2, 2: 3, 3: 5, 4: 9}[difficulty]
    projects = list(_PROJECTS[:project_count])
    rng.shuffle(projects)
    projects.sort(key=lambda item: item[0])
    base = datetime(2026, 6, 10, 10, 0, tzinfo=UTC)
    scopes = [
        ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL, created_at=base)
    ]
    scopes.extend(
        ScopeCreate(
            id=f"scope_{slug}", name=name, scope_type=ScopeType.PROJECT,
            parent_scope_id="global", created_at=base,
        )
        for slug, name, *_ in projects
    )
    sessions: list[SessionInput] = []

    def add_session(session_id: str, scope_id: str, turn: int, content: str) -> None:
        message = SourceMessageCreate(
            id=f"msg_{session_id}", session_id=session_id,
            role=MessageRole.USER, content=content, turn_index=turn,
            timestamp=base + timedelta(minutes=len(sessions)),
        )
        sessions.append(SessionInput(id=session_id, scope_id=scope_id, started_at=message.timestamp,
                                     messages=[message]))

    add_session("session_global_language", "global", 0, "I generally prefer Python.")
    add_session("session_global_database", "global", 0, "I generally prefer PostgreSQL.")
    for slug, name, _language, database, _frontend in projects:
        add_session(f"session_{slug}_facts", f"scope_{slug}", 0,
                    f"{name} uses {database}.")
        add_session(f"session_{slug}_language", f"scope_{slug}", 0,
                    f"{name} uses {_language} for its backend language.")
        add_session(f"session_{slug}_frontend", f"scope_{slug}", 0,
                    f"{name} uses {_frontend} for its frontend.")

    beta = next((item for item in projects if item[0] == "beta"), None)
    if beta is not None:
        add_session("session_beta_override", "scope_beta", 0,
                    "For today's migration test, temporarily use SQLite.")
    if difficulty >= 3 and beta is not None:
        add_session("session_beta_update", "scope_beta", 0,
                    "Beta is switching its database to PostgreSQL.")

    examples: list[BenchmarkExample] = []
    if beta is not None:
        examples.extend([
            BenchmarkExample(
                question_id="q_beta_normal", question_type="scope_specific_recall",
                question="What database does Beta use normally?", gold_answer="MongoDB",
                gold_scope_ids=["scope_beta"], gold_memory_ids=["m_beta_database"],
                gold_memory_contents=["Beta uses MongoDB."], current_scope_id="scope_beta",
                current_session_id="session_beta_override", timestamp=base,
            ),
            BenchmarkExample(
                question_id="q_beta_override", question_type="session_override",
                question="What database are we testing right now?", gold_answer="SQLite",
                gold_scope_ids=["scope_beta"], gold_memory_ids=["m_beta_override"],
                gold_memory_contents=["For today's migration test, temporarily use SQLite."],
                current_scope_id="scope_beta", current_session_id="session_beta_override",
                timestamp=base,
            ),
        ])
    examples.extend([
        BenchmarkExample(
            question_id="q_global_database", question_type="global_recall",
            question="What database do I generally prefer?", gold_answer="PostgreSQL",
            gold_scope_ids=["global"], gold_memory_ids=["m_global_database"],
            gold_memory_contents=["I generally prefer PostgreSQL."], current_scope_id="global",
            timestamp=base,
        ),
    ])
    alpha = next((item for item in projects if item[0] == "alpha"), None)
    if alpha is not None and beta is not None:
        examples.append(BenchmarkExample(
            question_id="q_named_beta", question_type="cross_scope_comparison",
            question="What database does Beta use?", gold_answer="MongoDB",
            gold_scope_ids=["scope_beta"], gold_memory_ids=["m_beta_database"],
            gold_memory_contents=["Beta uses MongoDB."], current_scope_id="scope_alpha",
            timestamp=base,
        ))
        examples.append(BenchmarkExample(
            question_id="q_alpha_database", question_type="scope_specific_recall",
            question="What database does Alpha use?", gold_answer="Neo4j",
            gold_scope_ids=["scope_alpha"], gold_memory_ids=["m_alpha_database"],
            gold_memory_contents=["Alpha uses Neo4j."], current_scope_id="scope_alpha",
            timestamp=base,
        ))

    scenario = CrossScopeScenario(
        scenario_id=scenario_id or f"cross_scope_mem_{seed}_{difficulty}",
        difficulty=difficulty, seed=seed, scopes=scopes, sessions=sessions, examples=examples,
    )
    return _with_canonical_content(scenario)


def _with_canonical_content(scenario: CrossScopeScenario) -> CrossScopeScenario:
    """Keep logical gold IDs stable while messages contain natural benchmark text."""
    for example in scenario.examples:
        if example.question_id == "q_beta_normal":
            example.gold_memory_contents[:] = ["Beta uses MongoDB."]
        elif example.question_id == "q_alpha_database":
            example.gold_memory_contents[:] = ["Alpha uses Neo4j."]
    return scenario


def candidates_by_message(scenario: CrossScopeScenario) -> dict[str, list[MemoryCandidate]]:
    """Return deterministic extraction candidates keyed by source-message ID."""
    candidates: dict[str, list[MemoryCandidate]] = {}
    for session in scenario.sessions:
        message = session.messages[0]
        content = message.content
        if session.scope_id == "global":
            if "Python" in content:
                candidate = _candidate(
                    message, subject="user", predicate="preferred_language", value="Python",
                    level="global", memory_type=MemoryType.PREFERENCE,
                )
            else:
                candidate = _candidate(
                    message, subject="user", predicate="preferred_database", value="PostgreSQL",
                    level="global", memory_type=MemoryType.PREFERENCE,
                )
        elif "temporarily" in content:
            candidate = _candidate(
                message, subject="beta", predicate="uses_database", value="SQLite",
                level="session", memory_type=MemoryType.TASK_STATE, durability=0.1,
            )
        elif "switching" in content:
            candidate = _candidate(message, subject="beta", predicate="uses_database",
                                   value="PostgreSQL", level="scope")
        else:
            slug = session.scope_id.removeprefix("scope_")
            project = next(item for item in _PROJECTS if item[0] == slug)
            if "backend language" in content:
                predicate, value = "uses_language", project[2]
            elif "frontend" in content:
                predicate, value = "uses_frontend", project[4]
            else:
                predicate, value = "uses_database", project[3]
            candidate = _candidate(message, subject=slug, predicate=predicate,
                                   value=value, level="scope")
        candidates[message.id] = [candidate]
    return candidates
