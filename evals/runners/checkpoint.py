"""Atomic manifests and append-only per-question checkpoints."""

import hashlib
import json
import platform
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.schemas import EvaluationRecord

# Bulk text repeated for every retrieved item. New runs move it to a sidecar file so the
# main JSONL stays small enough to open, diff and load; scoring never reads these fields.
BULK_FIELDS = ("retrieved_source_contents", "delivered_source_contents")
RUN_JSON_VERSION = 1


def save_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as sorted, indented JSON via an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_stamp() -> str:
    """Local-time file stem ``mm_dd__hh_mm`` used to name new runs."""
    return datetime.now().strftime("%m_%d__%H_%M")


def unused_name(directory: Path, stem: str, suffix: str = "") -> Path:
    """``directory/stem+suffix``, or ``stem_2``, ``stem_3``... if that name is taken."""
    candidate, number = directory / f"{stem}{suffix}", 2
    while candidate.exists():
        candidate = directory / f"{stem}_{number}{suffix}"
        number += 1
    return candidate


def try_write_report(jsonl_paths: Sequence[Path], destination: Path, **kwargs: Any) -> None:
    """Write the readable report after a run; a report problem never fails the run."""
    from evals.analysis.tables import write_readable_report

    try:
        print(f"report: {write_readable_report(jsonl_paths, destination, **kwargs)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - reporting is best-effort
        print(f"WARNING: could not write the readable report: {exc}", file=sys.stderr)


def git_state() -> dict[str, Any]:
    """Return the HEAD commit, whether the tree is dirty, and a hash of the uncommitted diff."""

    def run(*args: str) -> bytes | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    diff = run("diff", "HEAD") or b""
    status = run("status", "--porcelain", "--untracked-files=no") or b""
    return {
        "commit": commit.decode().strip() if commit else None,
        "dirty": bool(status.strip()),
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def reproduce_command(module: str) -> str:
    """Rebuild the command line that started this run from ``sys.argv``."""
    return shlex.join(["uv", "run", "python", "-m", module, *sys.argv[1:]])


def run_json_path(output: Path) -> Path:
    """``run.json`` for a batch directory, ``<stem>.run.json`` beside a single JSONL file."""
    return output / "run.json" if output.suffix == "" else output.with_suffix(".run.json")


def write_run_json(output: Path, **fields: Any) -> Path:
    """Create or update the run's provenance file; later calls merge into earlier ones."""
    path = run_json_path(output)
    current: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    current.setdefault("run_json_version", RUN_JSON_VERSION)
    current.setdefault("created_at", datetime.now(UTC).isoformat())
    current.setdefault("git", git_state())
    current.setdefault("python", platform.python_version())
    current.update(fields)
    save_json(path, current)
    return path


def source_fingerprint() -> str:
    """Hash every ``.py``/``.yaml`` file under ``src``, ``evals`` and ``configs``."""
    digest = hashlib.sha256()
    for root in (Path("src"), Path("evals"), Path("configs")):
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".yaml"}:
                continue
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class RecordCheckpoint:
    """Append-only per-question JSONL; bulk text goes to ``<name>.bulk.jsonl`` on new runs."""

    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.bulk_path = path.with_suffix(".bulk.jsonl")
        # Resuming a legacy file keeps its inline layout so one file never mixes formats.
        self.use_sidecar = not path.exists() or self.bulk_path.exists()
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
        """Append one record, moving bulk text to the sidecar on new runs."""
        key = (record.scenario_id, record.question_id)
        if key in self.records:
            raise ValueError(f"Question already checkpointed: {key}")
        stored = record
        if self.use_sidecar and any(getattr(record, name) for name in BULK_FIELDS):
            bulk = {name: getattr(record, name) for name in BULK_FIELDS}
            with self.bulk_path.open("a") as handle:
                handle.write(json.dumps({
                    "scenario_id": record.scenario_id, "question_id": record.question_id, **bulk,
                }) + "\n")
                handle.flush()
            stored = record.model_copy(update={name: [] for name in BULK_FIELDS})
        with self.path.open("a") as handle:
            handle.write(stored.model_dump_json() + "\n")
            handle.flush()
        self.records[key] = stored
