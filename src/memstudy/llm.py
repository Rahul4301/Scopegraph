"""One guarded, metered path for every scored model call (reader and judge)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import openai

from memstudy.budget import Budget, ModelPrice, cost_usd, worst_case_usd
from memstudy.schema import Usage
from memstudy.tokens import count_tokens

TRANSIENT = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
MAX_ATTEMPTS = 4


class IncompleteResponse(RuntimeError):
    """The API returned a non-completed response (for example max_output_tokens reached)."""


@dataclass
class CallResult:
    text: str
    usage: Usage
    latency_s: float
    model_returned: str
    response_id: str
    cost_usd: float
    warnings: list[str] = field(default_factory=list)


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def usage_from_response(raw: Any) -> tuple[Usage, list[str]]:
    """Map a Responses API usage object to Usage and list any fields the API did not return."""
    missing: list[str] = []
    details_in = _get(raw, "input_tokens_details")
    details_out = _get(raw, "output_tokens_details")
    cached = _get(details_in, "cached_tokens")
    written = _get(details_in, "cache_write_tokens")
    reasoning = _get(details_out, "reasoning_tokens")
    for name, value in (
        ("cached_tokens", cached),
        ("cache_write_tokens", written),
        ("reasoning_tokens", reasoning),
    ):
        if value is None:
            missing.append(name)
    usage = Usage(
        input_tokens=int(_get(raw, "input_tokens") or 0),
        cached_tokens=int(cached or 0),
        cache_write_tokens=int(written or 0),
        output_tokens=int(_get(raw, "output_tokens") or 0),
        reasoning_tokens=int(reasoning or 0),
    )
    return usage, missing


def estimate_input_tokens(input_items: list[dict[str, Any]]) -> int:
    total = 0
    for message in input_items:
        for block in message["content"]:
            total += count_tokens(block["text"]) + 8
    return total + 16


class ModelCaller:
    def __init__(
        self,
        client: Any,
        price: ModelPrice,
        budget: Budget,
        sleep: Any = time.sleep,
    ) -> None:
        self.client = client
        self.price = price
        self.budget = budget
        self._sleep = sleep

    def call(
        self,
        *,
        stage: str,
        tag: dict[str, Any],
        input_items: list[dict[str, Any]],
        max_output_tokens: int,
        reasoning_effort: str,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CallResult:
        estimate = worst_case_usd(
            self.price, estimate_input_tokens(input_items), max_output_tokens
        )
        kwargs: dict[str, Any] = {
            "model": self.price.name,
            "input": input_items,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if extra_body:
            kwargs["extra_body"] = extra_body

        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self.budget.guard(stage, estimate)
            start = time.perf_counter()
            try:
                resp = self.client.responses.create(**kwargs)
            except TRANSIENT as err:
                last_error = err
                self._sleep(2.0**attempt)
                continue
            latency = time.perf_counter() - start
            usage, missing = usage_from_response(resp.usage)
            cost = cost_usd(self.price, usage)
            self.budget.charge(
                stage,
                cost,
                model=self.price.name,
                input_tokens=usage.input_tokens,
                cached_tokens=usage.cached_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                **tag,
            )
            if resp.status != "completed":
                raise IncompleteResponse(
                    f"{self.price.name} status={resp.status} id={resp.id} (cost was charged)"
                )
            return CallResult(
                text=resp.output_text,
                usage=usage,
                latency_s=latency,
                model_returned=str(resp.model),
                response_id=str(resp.id),
                cost_usd=cost,
                warnings=[f"usage field missing: {m}" for m in missing],
            )
        assert last_error is not None
        raise last_error
