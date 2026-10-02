"""Hard gates for paid commands: pre-registration tag, approvals file, and key handling."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

PREREG_TAG = "prereg-v1"
APPROVALS = Path("configs/approvals.yaml")


class NotApproved(RuntimeError):
    pass


def prereg_committed(tag: str = PREREG_TAG) -> bool:
    """True when the tag exists and PREREG.md is tracked at that tag."""
    try:
        files = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", tag],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return False
    return "PREREG.md" in files


def require_gates(*gates: str, approvals_path: Path = APPROVALS) -> None:
    if not prereg_committed():
        raise NotApproved(f"tag {PREREG_TAG} with PREREG.md is not committed")
    approved = yaml.safe_load(approvals_path.read_text())
    for gate in gates:
        if approved.get(gate) is not True:
            raise NotApproved(f"gate {gate} is not approved in {approvals_path}")


def load_env_key(
    target: str = "OPENAI_API_KEY", env_file: Path = Path(".env"), key_var: str | None = None
) -> None:
    """Make the environment variable `target` available without ever printing it. Only the one
    named variable (key_var, default target) is read, from the environment or the env file;
    key_var lets the owner point at a differently named variable holding the key."""
    source = key_var or target
    if source == target and os.environ.get(target):
        return
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == source and value.strip():
                os.environ[target] = value.strip().strip("\"'")
                return
    raise NotApproved(f"{source} is not set in the environment or {env_file}")


def load_openai_key(env_file: Path = Path(".env"), key_var: str = "OPENAI_API_KEY") -> None:
    load_env_key("OPENAI_API_KEY", env_file, key_var)
