"""Benchmark loaders. Each returns (histories, items) in the common schema."""

from __future__ import annotations

from pathlib import Path

from memstudy.loaders.locomo import load_locomo
from memstudy.loaders.longmemeval import load_longmemeval
from memstudy.loaders.memoryagentbench import load_memoryagentbench
from memstudy.schema import History, Item

DEFAULT_PATHS = {
    "locomo": Path("data/locomo/locomo10.json"),
    "longmemeval": Path("data/longmemeval/longmemeval_s_cleaned.json"),
    "memoryagentbench": Path("data/memoryagentbench/data"),
}


def load_bench(bench: str, path: Path | None = None) -> tuple[dict[str, History], list[Item]]:
    if bench == "locomo":
        return load_locomo(path or DEFAULT_PATHS["locomo"])
    if bench == "longmemeval":
        return load_longmemeval(path or DEFAULT_PATHS["longmemeval"])
    if bench == "memoryagentbench":
        return load_memoryagentbench(path or DEFAULT_PATHS["memoryagentbench"])
    raise ValueError(f"unknown chat benchmark: {bench}")
