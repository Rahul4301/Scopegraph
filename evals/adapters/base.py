"""Common dataset adapter protocol."""

from typing import Protocol

from evals.schemas import BenchmarkExample, CrossScopeScenario


class DatasetAdapter(Protocol):
    name: str

    def scenarios(self) -> list[CrossScopeScenario]: ...

    def examples(self) -> list[BenchmarkExample]: ...
