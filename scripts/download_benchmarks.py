"""Download only the official artifacts used by the ScopeGraph evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    path: str
    url: str
    sha256: str
    requires_hf_token: bool = False


ARTIFACTS = (
    Artifact(
        "data/official_scorers/longmemeval_evaluate_qa.py",
        "https://api.github.com/repos/xiaowu0162/LongMemEval/contents/src/evaluation/"
        "evaluate_qa.py?ref=9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    ),
    Artifact(
        "data/official_scorers/locomo_evaluation.py",
        "https://api.github.com/repos/snap-research/locomo/contents/task_eval/evaluation.py"
        "?ref=3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
        "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd",
    ),
    Artifact(
        "data/official_scorers/memoryagentbench_summarization_evaluate.py",
        "https://api.github.com/repos/HUST-AI-HYZ/MemoryAgentBench/contents/llm_based_eval/"
        "summarization_evaluate.py?ref=fe1735de8cf8b9908e1e3d3b5612afc815698062",
        "6951d19644cdb1972bcf6cdb272f4757aa995ae07099bf74012925158c376dd4",
    ),
    Artifact(
        "data/longmemeval/longmemeval_s_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
        "longmemeval_s_cleaned.json",
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        True,
    ),
    Artifact(
        "data/locomo/locomo10.json",
        "https://api.github.com/repos/snap-research/locomo/contents/data/locomo10.json"
        "?ref=3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
        "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    ),
    Artifact(
        "data/memoryagentbench/Accurate_Retrieval.parquet",
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/"
        "Accurate_Retrieval-00000-of-00001.parquet",
        "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
        True,
    ),
    Artifact(
        "data/memoryagentbench/Conflict_Resolution.parquet",
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/"
        "Conflict_Resolution-00000-of-00001.parquet",
        "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
        True,
    ),
    Artifact(
        "data/memoryagentbench/Long_Range_Understanding.parquet",
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/"
        "Long_Range_Understanding-00000-of-00001.parquet",
        "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
        True,
    ),
    Artifact(
        "data/memoryagentbench/Test_Time_Learning.parquet",
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/"
        "Test_Time_Learning-00000-of-00001.parquet",
        "5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",
        True,
    ),
    Artifact(
        "data/memoryagentbench/entity2id.json",
        "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json",
        "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5",
        True,
    ),
)


def _load_hf_token(root: Path) -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token
    env_path = root / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "HF_TOKEN":
            return value.strip().strip("'\"") or None
    return None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: Path, artifact: Artifact, *, token: str | None, force: bool) -> str:
    destination = root / artifact.path
    if destination.is_file() and not force and _digest(destination) == artifact.sha256:
        return f"verified {artifact.path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "scopegraph-benchmark-fetcher/1"}
    if artifact.requires_hf_token and token:
        headers["Authorization"] = f"Bearer {token}"
    if "api.github.com" in artifact.url:
        headers["Accept"] = "application/vnd.github.raw+json"
    request = urllib.request.Request(artifact.url, headers=headers)
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        actual = _digest(temporary)
        if actual != artifact.sha256:
            raise ValueError(
                f"Checksum mismatch for {artifact.path}: expected {artifact.sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return f"downloaded {artifact.path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    token = _load_hf_token(args.root)
    for artifact in ARTIFACTS:
        print(download(args.root, artifact, token=token, force=args.force))


if __name__ == "__main__":
    main()
