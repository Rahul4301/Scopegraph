"""Immutable result store. Raw files are created exclusively and never overwritten."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from memstudy.schema import safe_name


class ResultStore:
    """Layout under root:
    raw/<arm>/<bench>/<item_id>.json      one scored item
    ingest/<arm>/<bench>/<history_id>.json  ingestion cost and time of one history
    errors/<arm>/<bench>/<item_id>-<ts>.json  items that did not complete (retryable)
    runs/<run_id>/run.json                run summary
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _raw(self, arm: str, bench: str, item_id: str) -> Path:
        return self.root / "raw" / arm / bench / f"{safe_name(item_id)}.json"

    def _ingest(self, arm: str, bench: str, history_id: str) -> Path:
        return self.root / "ingest" / arm / bench / f"{safe_name(history_id)}.json"

    @staticmethod
    def _write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as fh:  # raises FileExistsError instead of overwriting
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def has_item(self, arm: str, bench: str, item_id: str) -> bool:
        return self._raw(arm, bench, item_id).exists()

    def write_item(self, arm: str, bench: str, item_id: str, record: dict[str, Any]) -> None:
        self._write_new(self._raw(arm, bench, item_id), record)

    def read_items(self, arm: str, bench: str) -> list[dict[str, Any]]:
        folder = self.root / "raw" / arm / bench
        if not folder.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]

    def has_ingest(self, arm: str, bench: str, history_id: str) -> bool:
        return self._ingest(arm, bench, history_id).exists()

    def write_ingest(
        self, arm: str, bench: str, history_id: str, record: dict[str, Any]
    ) -> None:
        self._write_new(self._ingest(arm, bench, history_id), record)

    def read_ingest(self, arm: str, bench: str) -> list[dict[str, Any]]:
        folder = self.root / "ingest" / arm / bench
        if not folder.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]

    def write_error(self, arm: str, bench: str, item_id: str, record: dict[str, Any]) -> None:
        stamp = f"{time.time_ns()}"
        path = self.root / "errors" / arm / bench / f"{safe_name(item_id)}-{stamp}.json"
        self._write_new(path, record)

    def read_errors(self, arm: str, bench: str) -> list[dict[str, Any]]:
        folder = self.root / "errors" / arm / bench
        if not folder.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]

    def write_run(self, run_id: str, record: dict[str, Any]) -> None:
        self._write_new(self.root / "runs" / run_id / "run.json", record)
