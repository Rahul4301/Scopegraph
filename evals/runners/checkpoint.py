"""Atomic manifests and append-only per-question checkpoints."""

import hashlib
import json
from pathlib import Path
from typing import Any

from evals.schemas import EvaluationRecord


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for root in (Path("src"), Path("evals"), Path("configs")):
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".yaml"}:
                continue
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class RecordCheckpoint:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.records: dict[tuple[str, str], EvaluationRecord] = {}
        if path.exists():
            if not resume:
                raise FileExistsError(f"{path} exists; use --resume or choose a new output")
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = EvaluationRecord.model_validate_json(line)
                key = (record.scenario_id, record.question_id)
                if key in self.records:
                    raise ValueError(f"Duplicate checkpoint question: {key}")
                self.records[key] = record
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EvaluationRecord) -> None:
        key = (record.scenario_id, record.question_id)
        if key in self.records:
            raise ValueError(f"Question already checkpointed: {key}")
        with self.path.open("a") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
        self.records[key] = record
