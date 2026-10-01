"""Regression: the answer model must receive the current scope and the query timestamp.

Seven of 190 Neo4j-live answers were UNKNOWN with the gold evidence at rank 1 because
"this project" had no referent: the runner never passed the scope or as-of time.
"""

import pytest

from evals.adapters.cross_scope_mem import KeywordEmbeddingProvider, ScenarioExtractor
from evals.runners.providers import EvaluationProviders, PreparedEmbedder
from evals.runners.run_eval import run_scenario
from evals.scenarios.generate_cross_scope import candidates_by_message
from evals.scenarios.research import generate_research_scenario
from scopegraph.llm.answering import OpenAICompatibleAnswerer, describe_scope
from scopegraph.models.retrieval import RetrievedMemory
from scopegraph.models.scope import ScopeRef, ScopeType


class RecordingAnswerer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, *, question, context, instruction=None,
                       max_output_tokens=None, as_of=None, current_scope=None):
        self.calls.append(
            {"question": question, "as_of": as_of, "current_scope": current_scope}
        )
        return "UNKNOWN"


@pytest.mark.asyncio
async def test_runner_passes_current_scope_and_timestamp_to_answer_model():
    scenario = generate_research_scenario(seed=42, difficulty=1)
    providers = EvaluationProviders(
        ScenarioExtractor(candidates_by_message(scenario)),
        PreparedEmbedder(KeywordEmbeddingProvider()),
    )
    answerer = RecordingAnswerer()
    rows = await run_scenario(
        scenario, system_name="scopegraph", run_id="r", config={}, config_hash="h",
        providers=providers, answer_model=answerer,
    )
    assert len(answerer.calls) == len(rows) == len(scenario.examples)
    scoped = [e for e in scenario.examples if e.current_scope_id]
    assert scoped
    for example in scoped:
        matches = [
            c for c in answerer.calls
            if c["question"] == example.question and c["current_scope"] is not None
            and c["current_scope"].id == example.current_scope_id
        ]
        assert matches, f"{example.question_id}: answer model never saw its current scope"
        assert matches[0]["as_of"] == example.timestamp
        assert matches[0]["current_scope"].name


@pytest.mark.asyncio
async def test_answer_prompt_names_current_scope_and_asof():
    answerer = OpenAICompatibleAnswerer(base_url="http://x", api_key="k", model="m")
    sent: dict = {}

    async def fake_post(url, *, payload, api_key):
        sent.update(payload)
        return {"choices": [{"message": {"content": "SQLite"}}]}

    answerer.transport.post = fake_post  # type: ignore[method-assign]
    from datetime import UTC, datetime

    item = RetrievedMemory(
        memory_id="m1", content="Beta uses SQLite", score=1.0, scope_id="scope_beta",
        scope_level="scope", status="active", confidence=1.0,
    )
    stamp = datetime(2026, 7, 1, tzinfo=UTC)
    scope = ScopeRef(id="scope_beta", name="Beta", scope_type=ScopeType.PROJECT)
    await answerer.generate(
        question="What database does this project use normally?",
        context=[item], current_scope=scope, as_of=stamp,
    )
    user = sent["messages"][1]["content"]
    assert "Current scope: Beta (id=scope_beta; type=project)" in user
    assert stamp.isoformat() in user
    assert "this project" in sent["messages"][0]["content"]


def test_describe_scope_handles_missing_metadata():
    assert describe_scope(None) == "not provided"
    assert describe_scope(ScopeRef(id="x")) == "(id=x; type=unknown type)"
