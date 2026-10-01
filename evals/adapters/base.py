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

    def scenarios(self) -> list[CrossScopeScenario]:
        """Return the generated account scenarios."""

    def examples(self) -> list[BenchmarkExample]:
        """Return every question across all scenarios."""


class ExternalDatasetAdapter(Protocol):
    name: str

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]:
        """Load the dataset's questions with their histories."""

    def validate(self, path: str | Path) -> AdapterValidation:
        """Check that the dataset loads and report its size."""
