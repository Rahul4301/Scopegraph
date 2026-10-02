"""Prices, cost computation, and the spend guard.

The guard refuses any call whose worst-case cost would push a stage or the study past its cap.
Spend is an append-only JSONL ledger, so a restarted process resumes from the true total.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from memstudy.schema import Usage

TOTAL_CAP_USD = 350.0
STAGE_CAPS_USD = {"pilot": 15.0, "chat": 90.0}


class BudgetExceeded(RuntimeError):
    pass


class UnverifiedPrice(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPrice:
    name: str
    verified: bool
    input: float
    cached_input: float
    cache_write: float
    output: float
    long_context_threshold: int | None
    long_context_multiplier: dict[str, float]
    context_window: int | None

    def multipliers(self, input_tokens: int) -> dict[str, float]:
        if self.long_context_threshold and input_tokens > self.long_context_threshold:
            return self.long_context_multiplier
        return {"input": 1.0, "cached_input": 1.0, "cache_write": 1.0, "output": 1.0}


def load_prices(path: Path) -> dict[str, ModelPrice]:
    raw = yaml.safe_load(Path(path).read_text())["models"]
    prices: dict[str, ModelPrice] = {}
    for name, m in raw.items():
        std = m["standard"]
        prices[name] = ModelPrice(
            name=name,
            verified=bool(m["verified"]),
            input=float(std["input"]),
            cached_input=float(std["cached_input"]),
            cache_write=float(std["cache_write"]),
            output=float(std["output"]),
            long_context_threshold=m.get("long_context_threshold"),
            long_context_multiplier=m.get("long_context_multiplier")
            or {"input": 1.0, "cached_input": 1.0, "cache_write": 1.0, "output": 1.0},
            context_window=m.get("context_window"),
        )
    return prices


@dataclass(frozen=True)
class SupermemoryPrice:
    """Published rate card of the hosted service (an estimate: usage is not metered by us)."""

    usd_per_1m_memory_tokens: float
    usd_per_1m_queries: float
    usd_per_1m_operations: float

    def cost(self, memory_tokens: int, queries: int, operations: int) -> float:
        return (
            memory_tokens * self.usd_per_1m_memory_tokens
            + queries * self.usd_per_1m_queries
            + operations * self.usd_per_1m_operations
        ) / 1_000_000


def load_supermemory_price(path: Path) -> SupermemoryPrice:
    raw = yaml.safe_load(Path(path).read_text())["services"]["supermemory"]
    if not raw["verified"]:
        raise UnverifiedPrice("supermemory price is not verified in configs/prices.yaml")
    return SupermemoryPrice(
        usd_per_1m_memory_tokens=float(raw["usd_per_1m_memory_tokens"]),
        usd_per_1m_queries=float(raw["usd_per_1m_queries"]),
        usd_per_1m_operations=float(raw["usd_per_1m_operations"]),
    )


def cost_usd(price: ModelPrice, usage: Usage) -> float:
    """Dollar cost of one call.

    Input splits into uncached, cached-read, and cache-write tokens, each at its own rate (the
    cache write is not an additive fee). Output tokens already include reasoning tokens. Prompts
    over the long-context threshold are priced at the multipliers for the whole request.
    """
    if not price.verified:
        raise UnverifiedPrice(f"price for {price.name} is not verified in configs/prices.yaml")
    mult = price.multipliers(usage.input_tokens)
    uncached = usage.input_tokens - usage.cached_tokens - usage.cache_write_tokens
    total = (
        uncached * price.input * mult["input"]
        + usage.cached_tokens * price.cached_input * mult["cached_input"]
        + usage.cache_write_tokens * price.cache_write * mult["cache_write"]
        + usage.output_tokens * price.output * mult["output"]
    )
    return total / 1_000_000


def worst_case_usd(price: ModelPrice, input_tokens: int, max_output_tokens: int) -> float:
    """Upper bound used by the guard before a call: every input token at the dearer of the
    uncached and cache-write rates, and the full output allowance spent."""
    if not price.verified:
        raise UnverifiedPrice(f"price for {price.name} is not verified in configs/prices.yaml")
    mult = price.multipliers(input_tokens)
    in_rate = max(price.input * mult["input"], price.cache_write * mult["cache_write"])
    return (input_tokens * in_rate + max_output_tokens * price.output * mult["output"]) / 1e6


class Budget:
    """Append-only ledger with stage and total caps."""

    def __init__(
        self,
        ledger_path: Path,
        total_cap: float = TOTAL_CAP_USD,
        stage_caps: dict[str, float] | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.total_cap = total_cap
        self.stage_caps = dict(STAGE_CAPS_USD if stage_caps is None else stage_caps)
        self._lock = threading.Lock()
        self._spent: dict[str, float] = {}
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._spent[row["stage"]] = self._spent.get(row["stage"], 0.0) + row["usd"]

    @property
    def spent_total(self) -> float:
        return sum(self._spent.values())

    def spent(self, stage: str) -> float:
        return self._spent.get(stage, 0.0)

    def remaining(self, stage: str) -> float:
        return min(
            self.stage_caps[stage] - self.spent(stage), self.total_cap - self.spent_total
        )

    def guard(self, stage: str, estimate_usd: float) -> None:
        """Raise before a call if its worst case would exceed the stage or total cap."""
        if stage not in self.stage_caps:
            raise BudgetExceeded(f"unknown stage {stage!r}")
        with self._lock:
            if self.spent(stage) + estimate_usd > self.stage_caps[stage]:
                raise BudgetExceeded(
                    f"stage {stage}: spent {self.spent(stage):.4f} + call {estimate_usd:.4f} "
                    f"exceeds cap {self.stage_caps[stage]:.2f}"
                )
            if self.spent_total + estimate_usd > self.total_cap:
                raise BudgetExceeded(
                    f"total: spent {self.spent_total:.4f} + call {estimate_usd:.4f} "
                    f"exceeds cap {self.total_cap:.2f}"
                )

    def charge(self, stage: str, usd: float, **meta: Any) -> None:
        row = {"ts": time.time(), "stage": stage, "usd": usd, **meta}
        with self._lock:
            with self.ledger_path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            self._spent[stage] = self._spent.get(stage, 0.0) + usd
