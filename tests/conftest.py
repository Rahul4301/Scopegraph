from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from memstudy.budget import Budget, load_prices
from memstudy.config import load_config
from memstudy.schema import History, Item, Session, Turn

ROOT = Path(__file__).resolve().parents[1]


def make_response(
    text: str = "ok",
    input_tokens: int = 1000,
    cached: int = 0,
    written: int = 0,
    output: int = 10,
    reasoning: int = 0,
    status: str = "completed",
    model: str = "gpt-6-luna-test-snapshot",
) -> Any:
    return NS(
        id="resp_test",
        model=model,
        status=status,
        output_text=text,
        usage=NS(
            input_tokens=input_tokens,
            input_tokens_details=NS(cached_tokens=cached, cache_write_tokens=written),
            output_tokens=output,
            output_tokens_details=NS(reasoning_tokens=reasoning),
        ),
    )


class FakeResponses:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.handler(kwargs)


class FakeClient:
    def __init__(self, handler: Any) -> None:
        self.responses = FakeResponses(handler)


@pytest.fixture
def prices() -> Any:
    return load_prices(ROOT / "configs" / "prices.yaml")


@pytest.fixture
def cfg() -> dict[str, Any]:
    return load_config(ROOT / "configs" / "study.yaml")


@pytest.fixture
def budget(tmp_path: Path) -> Budget:
    return Budget(tmp_path / "ledger.jsonl")


def make_history(history_id: str = "h1", bench: str = "locomo", marker: str = "alpha") -> History:
    return History(
        history_id=history_id,
        bench=bench,
        sessions=[
            Session(
                session_id="1",
                timestamp="1:00 pm on 1 May, 2023",
                turns=[
                    Turn(speaker="Ann", text=f"I adopted a dog named {marker}."),
                    Turn(speaker="Bob", text="That is wonderful news."),
                ],
            ),
            Session(
                session_id="2",
                timestamp="2:00 pm on 9 May, 2023",
                turns=[Turn(speaker="Ann", text=f"{marker} learned to swim today.")],
            ),
        ],
    )


def make_item(history_id: str = "h1", n: int = 0, bench: str = "locomo") -> Item:
    return Item(
        item_id=f"{history_id}-{n:04d}",
        bench=bench,
        history_id=history_id,
        category="cat1",
        question="What is the dog called?",
        gold="alpha",
    )
