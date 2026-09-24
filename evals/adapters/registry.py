"""Single source of truth for external benchmark adapters."""

from evals.adapters.base import ExternalDatasetAdapter
from evals.adapters.locomo import LoCoMoAdapter
from evals.adapters.longmemeval import LongMemEvalAdapter
from evals.adapters.memoryagentbench import MemoryAgentBenchAdapter


def external_adapters() -> dict[str, ExternalDatasetAdapter]:
    adapters: list[ExternalDatasetAdapter] = [
        LongMemEvalAdapter(),
        LoCoMoAdapter(),
        MemoryAgentBenchAdapter(),
    ]
    return {adapter.name: adapter for adapter in adapters}


EXTERNAL_DATASETS = tuple(external_adapters())
