"""Meters OpenAI client calls made outside the reader and judge (Mem0 internals, embeddings).

Mem0 builds its own OpenAI clients for fact extraction and embeddings. Wrapping those clients
puts every ingestion and retrieval call under the same budget guard and the same ledger as the
scored calls, so no spend is invisible.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from memstudy.budget import Budget, ModelPrice, cost_usd, worst_case_usd
from memstudy.schema import Usage
from memstudy.tokens import count_tokens

DEFAULT_MAX_OUTPUT = 4096


@dataclass
class CostSink:
    """Accumulates cost, tokens, calls, and call seconds for one purpose (ingest, retrieve)."""

    usd: float = 0.0
    calls: int = 0
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)

    def snapshot(self) -> CostSink:
        return CostSink(self.usd, self.calls, self.seconds, self.usage.model_copy())

    def since(self, earlier: CostSink) -> CostSink:
        return CostSink(
            usd=self.usd - earlier.usd,
            calls=self.calls - earlier.calls,
            seconds=self.seconds - earlier.seconds,
            usage=Usage(
                input_tokens=self.usage.input_tokens - earlier.usage.input_tokens,
                cached_tokens=self.usage.cached_tokens - earlier.usage.cached_tokens,
                cache_write_tokens=self.usage.cache_write_tokens
                - earlier.usage.cache_write_tokens,
                output_tokens=self.usage.output_tokens - earlier.usage.output_tokens,
                reasoning_tokens=self.usage.reasoning_tokens - earlier.usage.reasoning_tokens,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "usd": self.usd,
            "calls": self.calls,
            "seconds": self.seconds,
            "usage": self.usage.model_dump(),
        }


def _chat_usage(raw: Any) -> Usage:
    details_in = getattr(raw, "prompt_tokens_details", None)
    details_out = getattr(raw, "completion_tokens_details", None)
    return Usage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        cached_tokens=int(getattr(details_in, "cached_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        reasoning_tokens=int(getattr(details_out, "reasoning_tokens", 0) or 0),
    )


def _embedding_usage(raw: Any) -> Usage:
    return Usage(input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0))


class Meter:
    """Wraps chat.completions.create and embeddings.create on an OpenAI client in place."""

    def __init__(
        self,
        budget: Budget,
        prices: dict[str, ModelPrice],
        stage: str,
        sink: CostSink,
        tag: dict[str, Any],
    ) -> None:
        self.budget = budget
        self.prices = prices
        self.stage = stage
        self.sink = sink
        self.tag = tag

    def _price(self, model: str) -> ModelPrice:
        if model not in self.prices:
            raise KeyError(f"no price entry for model {model!r} in configs/prices.yaml")
        return self.prices[model]

    def _run(
        self, model: str, est_in: int, est_out: int, fn: Any, usage_fn: Any, purpose: str
    ) -> Any:
        price = self._price(model)
        self.budget.guard(self.stage, worst_case_usd(price, est_in, est_out))
        start = time.perf_counter()
        resp = fn()
        seconds = time.perf_counter() - start
        usage = usage_fn(resp.usage)
        usd = cost_usd(price, usage)
        self.budget.charge(
            self.stage,
            usd,
            model=model,
            input_tokens=usage.input_tokens,
            cached_tokens=usage.cached_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            purpose=purpose,
            **self.tag,
        )
        self.sink.usd += usd
        self.sink.calls += 1
        self.sink.seconds += seconds
        self.sink.usage = self.sink.usage + usage
        return resp

    @contextmanager
    def use_sink(self, sink: CostSink) -> Iterator[None]:
        """Route spend to a different sink (ingest vs query) for the duration of a block."""
        previous = self.sink
        self.sink = sink
        try:
            yield
        finally:
            self.sink = previous

    def wrap(self, client: Any) -> Any:
        chat_create = client.chat.completions.create
        embed_create = client.embeddings.create

        def chat(*args: Any, **kwargs: Any) -> Any:
            est_in = count_tokens(json.dumps(kwargs.get("messages", []))) + 16
            est_out = int(
                kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or DEFAULT_MAX_OUTPUT
            )
            return self._run(
                kwargs["model"],
                est_in,
                est_out,
                lambda: chat_create(*args, **kwargs),
                _chat_usage,
                "mem0-extract",
            )

        def embed(*args: Any, **kwargs: Any) -> Any:
            texts = kwargs["input"]
            texts = [texts] if isinstance(texts, str) else texts
            est_in = sum(count_tokens(t) for t in texts) + 8
            return self._run(
                kwargs["model"],
                est_in,
                0,
                lambda: embed_create(*args, **kwargs),
                _embedding_usage,
                "embed",
            )

        client.chat.completions.create = chat
        client.embeddings.create = embed
        return client
