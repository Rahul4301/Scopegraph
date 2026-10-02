"""Phase 0 token census: tokens per history, queries per history, window and threshold checks."""

from __future__ import annotations

import statistics
from typing import Any

from memstudy.budget import ModelPrice
from memstudy.schema import History, Item, render_transcript
from memstudy.tokens import check_fit, count_tokens


def census(
    histories: dict[str, History],
    items: list[Item],
    price: ModelPrice,
    margin: float,
    overhead_tokens: int = 600,
) -> dict[str, Any]:
    """Count every history with the proxy tokenizer. overhead_tokens covers the instructions,
    question, and answer so a history that fits only by a hair is still flagged."""
    queries: dict[str, int] = {}
    for item in items:
        queries[item.history_id] = queries.get(item.history_id, 0) + 1
    assert price.context_window is not None
    rows: list[dict[str, Any]] = []
    for hid, hist in histories.items():
        tokens = count_tokens(render_transcript(hist))
        report = check_fit(
            tokens + overhead_tokens, price.context_window, margin, price.long_context_threshold
        )
        rows.append(
            {
                "history_id": hid,
                "tokens": tokens,
                "queries": queries.get(hid, 0),
                "fits": report.fits,
                "over_long_context_threshold": report.over_long_context_threshold,
            }
        )
    counts: list[int] = [r["tokens"] for r in rows]
    return {
        "n_histories": len(rows),
        "n_items": len(items),
        "window": price.context_window,
        "long_context_threshold": price.long_context_threshold,
        "safety_margin": margin,
        "tokens": {
            "min": min(counts),
            "mean": statistics.fmean(counts),
            "median": statistics.median(counts),
            "max": max(counts),
            "total": sum(counts),
        },
        "n_exceed_window": sum(1 for r in rows if not r["fits"]),
        "n_over_long_context_threshold": sum(1 for r in rows if r["over_long_context_threshold"]),
        "histories": rows,
    }
