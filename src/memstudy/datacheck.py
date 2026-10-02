"""Verify the git-ignored benchmark files in data/ against the committed checksum manifest."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

MANIFEST = Path("configs/data_manifest.yaml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path = MANIFEST, root: Path = Path(".")) -> list[dict[str, Any]]:
    """One row per manifest file or pinned repository with status ok, missing, or mismatch.
    Never modifies anything."""
    entries = yaml.safe_load(manifest_path.read_text())["files"]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = root / entry["path"]
        if not path.exists():
            status = "missing"
        elif path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            status = "mismatch"
        else:
            status = "ok"
        rows.append({"path": entry["path"], "status": status})
    for repo in yaml.safe_load(manifest_path.read_text()).get("repos", []):
        path = root / repo["path"]
        if not (path / ".git").exists():
            status = "missing"
        else:
            head = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            status = "ok" if head == repo["commit"] else "mismatch"
        rows.append({"path": repo["path"], "status": status})
    return rows
