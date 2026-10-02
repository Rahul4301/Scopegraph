"""Study configuration (configs/study.yaml) and the config hash written to every run.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from memstudy.prompts import (
    JUDGE_INSTRUCTIONS,
    PROMPT_VERSION,
    READER_INSTRUCTIONS,
)

DEFAULT_CONFIG = Path("configs/study.yaml")
DEFAULT_PRICES = Path("configs/prices.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    assert isinstance(cfg, dict)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of the study config plus the exact prompt texts."""
    payload = {
        "cfg": cfg,
        "prompt_version": PROMPT_VERSION,
        "reader_instructions": READER_INSTRUCTIONS,
        "judge_instructions": JUDGE_INSTRUCTIONS,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
