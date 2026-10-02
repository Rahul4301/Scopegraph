"""Arm A: the whole history first, question last, no memory layer.

Cache policy: the cache breakpoint is placed after the history only when the history is queried
at least MIN_QUERIES_FOR_CACHE times in the run. A cache write is billed at 1.25x the input rate
and only pays back on a later read, so for a single query it is pure surcharge.
"""

from __future__ import annotations

from memstudy.arms.base import ArmContext, IngestStats
from memstudy.schema import History, Item, render_transcript
from memstudy.tokens import count_tokens, require_fit

MIN_QUERIES_FOR_CACHE = 2


class FullContextArm:
    name = "A"

    def __init__(
        self, window: int, margin: float, queries_per_history: dict[str, int] | None = None
    ) -> None:
        self.window = window
        self.margin = margin
        self.queries_per_history = queries_per_history or {}
        self._cache: dict[str, tuple[str, int]] = {}

    def prepare(self, history: History) -> IngestStats | None:
        self._transcript(history)
        return None

    def _transcript(self, history: History) -> tuple[str, int]:
        if history.history_id not in self._cache:
            text = render_transcript(history)
            tokens = count_tokens(text)
            require_fit(history.history_id, tokens, self.window, self.margin)
            self._cache = {history.history_id: (text, tokens)}
        return self._cache[history.history_id]

    def context(self, item: Item, history: History) -> ArmContext:
        text, tokens = self._transcript(history)
        use_cache = self.queries_per_history.get(history.history_id, 0) >= MIN_QUERIES_FOR_CACHE
        return ArmContext(text=text, context_tokens=tokens, cache_prefix=use_cache)
