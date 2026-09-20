"""CrossScopeMem dataset adapter and deterministic offline providers."""

import hashlib
import re

from evals.scenarios.generate_cross_scope import candidates_by_message, generate_cross_scope_mem
from evals.schemas import BenchmarkExample, CrossScopeScenario
from scopegraph.llm.extraction import CandidateExtractor
from scopegraph.models.memory import MemoryCandidate
from scopegraph.models.scope import ScopeRef
from scopegraph.models.source import SourceMessage


class ScenarioExtractor(CandidateExtractor):
    def __init__(self, candidates: dict[str, list[MemoryCandidate]]) -> None:
        self.candidates = candidates

    async def extract(self, messages: list[SourceMessage], *, current_scope: ScopeRef | None,
                      existing_memories: list[str] | None = None) -> list[MemoryCandidate]:
        del current_scope, existing_memories
        return [
            candidate.model_copy(deep=True)
            for message in messages
            for candidate in self.candidates.get(message.id, [])
        ]


class KeywordEmbeddingProvider:
    """Stable hash-bucket embeddings; no API key is needed for benchmark smoke runs."""

    model_name = "cross-scope-mem-hash-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 64
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
                vector[bucket % len(vector)] += 1.0
            vectors.append(vector)
        return vectors


class CrossScopeMemAdapter:
    name = "cross_scope_mem"

    def __init__(self, *, seed: int = 42, difficulty: int = 2, scenario_count: int = 1) -> None:
        self._scenarios = [
            generate_cross_scope_mem(seed=seed + index, difficulty=difficulty,
                                     scenario_id=f"cross_scope_mem_{seed + index}_{difficulty}")
            for index in range(scenario_count)
        ]

    def scenarios(self) -> list[CrossScopeScenario]:
        return list(self._scenarios)

    def examples(self) -> list[BenchmarkExample]:
        return [example for scenario in self._scenarios for example in scenario.examples]

    @staticmethod
    def extractor(scenario: CrossScopeScenario) -> ScenarioExtractor:
        return ScenarioExtractor(candidates_by_message(scenario))
