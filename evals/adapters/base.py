"""Common dataset adapter protocol."""

from pathlib import Path
from typing import Protocol

from evals.schemas import (
    AdapterValidation,
    BenchmarkExample,
    CrossScopeScenario,
    ExternalBenchmarkExample,
)


class DatasetAdapter(Protocol):
    name: str

    def scenarios(self) -> list[CrossScopeScenario]: ...

    def examples(self) -> list[BenchmarkExample]: ...


class ExternalDatasetAdapter(Protocol):
    name: str

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]: ...

    def validate(self, path: str | Path) -> AdapterValidation: ...
