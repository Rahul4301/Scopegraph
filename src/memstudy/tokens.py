"""Token counting. Nothing in the study truncates; items that do not fit are reported.

GPT-6 Luna has no tokenizer mapping in tiktoken 0.14.0, so o200k_base is used as the closest
available proxy. Counts are therefore estimates; the API's own usage.input_tokens is
authoritative and is what cost is computed from. A safety margin absorbs the proxy error.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

ENCODING_NAME = "o200k_base"


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


class DoesNotFit(RuntimeError):
    """Raised instead of truncating. The runner records the item as not fitting."""

    def __init__(self, label: str, tokens: int, limit: int) -> None:
        super().__init__(f"{label}: {tokens} tokens exceeds usable window of {limit}")
        self.label = label
        self.tokens = tokens
        self.limit = limit


@dataclass(frozen=True)
class FitReport:
    tokens: int
    window: int
    margin: float
    fits: bool
    over_long_context_threshold: bool


def usable_window(window: int, margin: float) -> int:
    return int(window * (1.0 - margin))


def check_fit(
    tokens: int, window: int, margin: float, long_context_threshold: int | None
) -> FitReport:
    return FitReport(
        tokens=tokens,
        window=window,
        margin=margin,
        fits=tokens <= usable_window(window, margin),
        over_long_context_threshold=bool(
            long_context_threshold and tokens > long_context_threshold
        ),
    )


def require_fit(label: str, tokens: int, window: int, margin: float) -> None:
    limit = usable_window(window, margin)
    if tokens > limit:
        raise DoesNotFit(label, tokens, limit)
