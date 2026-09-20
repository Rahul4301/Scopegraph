"""Single source of truth for external benchmark adapters."""

from evals.adapters.base import ExternalDatasetAdapter
from evals.adapters.extended import (
    LongMemEvalV2Adapter,
    Mem2ActBenchAdapter,
    MemBenchAdapter,
    MemoryAgentBenchAdapter,
    RHELMAdapter,
    TIMEAdapter,
)
from evals.adapters.locomo import LoCoMoAdapter
from evals.adapters.longmemeval import LongMemEvalAdapter
from evals.adapters.memconflict import MemConflictAdapter


def external_adapters() -> dict[str, ExternalDatasetAdapter]:
    adapters: list[ExternalDatasetAdapter] = [
        LongMemEvalAdapter(),
        LongMemEvalV2Adapter(),
        LoCoMoAdapter(),
        MemConflictAdapter(),
        MemoryAgentBenchAdapter(),
        RHELMAdapter(),
        MemBenchAdapter(),
        Mem2ActBenchAdapter(),
        TIMEAdapter(),
    ]
    return {adapter.name: adapter for adapter in adapters}


EXTERNAL_DATASETS = tuple(external_adapters())
