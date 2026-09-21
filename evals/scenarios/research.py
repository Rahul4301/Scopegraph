"""Structured, varied histories with evidence ground truth and explicit snapshots.

These are synthetic research workloads, not substitutes for external datasets.
Seeds change facts and distractors, rather than merely changing insertion order.
"""

import random
from datetime import UTC, datetime, timedelta

from evals.schemas import BenchmarkExample, CrossScopeScenario
from scopegraph.models.memory import MemoryCandidate, MemoryType
from scopegraph.models.scope import ScopeCreate, ScopeType
from scopegraph.models.session import SessionInput
from scopegraph.models.source import MessageRole, SourceMessageCreate


def generate_research_scenario(
    *, seed: int, difficulty: int, scenario_id: str | None = None
) -> CrossScopeScenario:
    if difficulty not in {1, 2, 3, 4}:
        raise ValueError("difficulty must be between 1 and 4")
    rng = random.Random(seed)
    count, session_target = {1: (3, 10), 2: (5, 50), 3: (10, 100), 4: (10, 200)}[difficulty]
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"][
        :count
    ]
    databases = ["Neo4j", "MongoDB", "PostgreSQL", "Redis", "SQLite", "MySQL"]
    languages = ["Python", "Rust", "Go", "TypeScript", "Java", "Kotlin"]
    frameworks = ["React", "Vue", "Svelte", "Angular", "Solid"]
    base = datetime(2026, 6, 10, tzinfo=UTC)
    scopes = [ScopeCreate(id="global", name="Global", scope_type=ScopeType.GLOBAL, created_at=base)]
    scopes += [
        ScopeCreate(
            id=f"scope_{name.lower()}",
            name=name,
            scope_type=ScopeType.PROJECT,
            parent_scope_id="global",
            created_at=base,
        )
        for name in names
    ]
    sessions: list[SessionInput] = []
    candidates: dict[str, list[MemoryCandidate]] = {}
    evidence: dict[tuple[str, str], tuple[str, str]] = {}

    def add(
        key: str,
        scope: str,
        content: str,
        predicate: str,
        value: str,
        *,
        subject: str | None = None,
        level: str = "scope",
    ) -> str:
        date = base + timedelta(hours=len(sessions))
        session_id, message_id = f"session_{key}", f"msg_{key}"
        sessions.append(
            SessionInput(
                id=session_id,
                scope_id=scope,
                started_at=date,
                messages=[
                    SourceMessageCreate(
                        id=message_id,
                        session_id=session_id,
                        content=content,
                        role=MessageRole.USER,
                        timestamp=date,
                        turn_index=0,
                    )
                ],
            )
        )
        candidates[message_id] = [
            MemoryCandidate(
                content=content,
                subject=subject or scope,
                predicate=predicate,
                object=value,
                proposed_scope_level=level,
                confidence=1,
                durability=0.1 if level == "session" else 1,
                source_message_ids=[message_id],
                memory_type=MemoryType.FACT,
                explicit_global_signal=level == "global",
                valid_from=date,
            )
        ]
        return message_id

    global_db, global_language = rng.choice(databases), rng.choice(languages)
    global_db_source = add(
        "global_db",
        "global",
        f"I generally prefer {global_db} as a database.",
        "preferred_database",
        global_db,
        level="global",
    )
    global_language_source = add(
        "global_language",
        "global",
        f"I generally prefer {global_language} as a programming language.",
        "preferred_language",
        global_language,
        level="global",
    )
    for name in names:
        attributes = [("database", rng.choice(databases))]
        if difficulty >= 2:
            attributes += [
                ("language", rng.choice(languages)),
                ("frontend", rng.choice(frameworks)),
            ]
        for attribute, value in attributes:
            text = rng.choice(
                [
                    f"{name} uses {value} for its {attribute}.",
                    f"The {attribute} chosen for {name} is {value}.",
                    f"For {name}, our {attribute} is {value}.",
                ]
            )
            source = add(
                f"{name}_{attribute}", f"scope_{name.lower()}", text, f"uses_{attribute}", value
            )
            evidence[(name, attribute)] = (value, source)
    beta_old, beta_old_source = evidence[("Beta", "database")]
    beta_new = rng.choice([db for db in databases if db != beta_old])
    if difficulty >= 3:
        source = add(
            "beta_update",
            "scope_beta",
            f"The migration is complete. Beta now uses {beta_new} as its database "
            f"instead of {beta_old}.",
            "uses_database",
            beta_new,
        )
        evidence[("Beta", "database")] = (beta_new, source)
    # A two-fact dependency chain inside one scope.
    worker_source = db_source = None
    worker_db = rng.choice(databases)
    if difficulty >= 2:
        worker = f"worker-{rng.randrange(1000, 9999)}"
        worker_source = add(
            "worker",
            "scope_alpha",
            f"Alpha depends on the service {worker}.",
            "depends_on",
            worker,
            subject="alpha",
        )
        db_source = add(
            "worker_db",
            "scope_alpha",
            f"Service {worker} uses {worker_db} as its database.",
            "uses_database",
            worker_db,
            subject=worker,
        )
    # Leave one session for the temporary override. Filler is meaningful, durable,
    # unrelated project state and creates genuine interference under tight budgets.
    while len(sessions) < session_target - 1:
        name = rng.choice(names)
        number = len(sessions)
        value = f"milestone-{rng.randrange(10000)}"
        add(
            f"distractor_{number}",
            f"scope_{name.lower()}",
            f"{name} completed {value} for release {number}.",
            f"release_{number}",
            value,
        )
    override = rng.choice([db for db in databases if db != evidence[("Beta", "database")][0]])
    override_source = add(
        "beta_override",
        "scope_beta",
        f"For this session only, test {override}; the normal database is unchanged.",
        "uses_database",
        override,
        level="session",
    )
    final_time = sessions[-1].started_at + timedelta(minutes=1)
    examples: list[BenchmarkExample] = []

    def question(
        key,
        kind,
        text,
        answer,
        sources,
        current,
        *,
        session=None,
        allowed=None,
        timestamp=None,
    ):
        source_to_scope = {
            message.id: item.scope_id for item in sessions for message in item.messages
        }
        gold_scopes = sorted({source_to_scope[source] for source in sources})
        examples.append(
            BenchmarkExample(
                question_id=key,
                question_type=kind,
                question=text,
                gold_answer=answer,
                gold_source_ids=sources,
                gold_scope_ids=gold_scopes,
                allowed_scope_ids=allowed or sorted({"global", current, *gold_scopes}),
                current_scope_id=current,
                current_session_id=session,
                timestamp=timestamp or final_time,
                gold_memory_contents=[candidates[source][0].content for source in sources],
            )
        )

    question(
        "q_global_database",
        "global_recall",
        "What database do I generally prefer?",
        global_db,
        [global_db_source],
        "global",
    )
    question(
        "q_global_language",
        "global_recall",
        "What programming language do I generally prefer?",
        global_language,
        [global_language_source],
        "global",
    )
    for name in names:
        value, source = evidence[(name, "database")]
        question(
            f"q_{name.lower()}_database",
            "scope_specific_recall",
            "What database does this project use normally?",
            value,
            [source],
            f"scope_{name.lower()}",
        )
    value, source = evidence[("Beta", "database")]
    question(
        "q_beta_normal",
        "local_override",
        "What database does Beta use normally?",
        value,
        [source],
        "scope_beta",
    )
    question(
        "q_beta_override",
        "session_override",
        "What database should I use for this test?",
        override,
        [override_source],
        "scope_beta",
        session="session_beta_override",
    )
    question(
        "q_named_beta",
        "named_scope",
        "What database does Beta use now?",
        value,
        [source],
        "scope_alpha",
    )
    alpha_db, alpha_source = evidence[("Alpha", "database")]
    question(
        "q_compare",
        "cross_scope_comparison",
        "Give Alpha's and Beta's current databases, in that order, separated by a semicolon.",
        f"{alpha_db}; {value}",
        [alpha_source, source],
        "scope_alpha",
    )
    question(
        "q_abstain",
        "abstention",
        "What is the shipping address for this project?",
        "UNKNOWN",
        [],
        "scope_alpha",
    )
    if difficulty >= 3:
        # Historical retrieval is an explicit as-of snapshot. Without this timestamp,
        # the old value is correctly inactive at ``final_time`` and the benchmark would
        # be asking the retriever to violate its temporal-validity contract.
        beta_update_time = next(
            item.started_at for item in sessions if item.id == "session_beta_update"
        )
        question(
            "q_beta_history",
            "temporal_historical",
            "At this historical snapshot, what database does Beta use?",
            beta_old,
            [beta_old_source],
            "scope_beta",
            timestamp=beta_update_time - timedelta(microseconds=1),
        )
    if worker_source and db_source:
        question(
            "q_dependency",
            "multi_hop",
            "What database is used by the service Alpha depends on?",
            worker_db,
            [worker_source, db_source],
            "scope_alpha",
        )
    return CrossScopeScenario(
        profile="research",
        scenario_id=scenario_id or f"research_{seed}_{difficulty}",
        difficulty=difficulty,
        seed=seed,
        scopes=scopes,
        sessions=sessions,
        examples=examples,
        oracle_candidates=candidates,
    )
