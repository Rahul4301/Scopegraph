"""Adapter for the official MemoryAgentBench Parquet release."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import tiktoken

from evals.adapters.longmemeval import _validation
from evals.schemas import (
    AdapterValidation,
    ExternalBenchmarkExample,
    ExternalSession,
    ExternalTurn,
)


def _chunks(text: str, *, token_limit: int = 4096) -> list[str]:
    """Preserve all official context while matching the benchmark chunk size."""
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".scopegraph/tiktoken").resolve()))
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(text)
    return [
        encoding.decode(tokens[offset : offset + token_limit])
        for offset in range(0, len(tokens), token_limit)
    ]


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


class MemoryAgentBenchAdapter:
    name = "memoryagentbench"

    def load(self, path: str | Path) -> list[ExternalBenchmarkExample]:
        source = Path(path)
        paths = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
        if not paths or not all(item.is_file() for item in paths):
            raise FileNotFoundError(f"MemoryAgentBench Parquet path does not exist: {source}")
        examples: list[ExternalBenchmarkExample] = []
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for parquet_path in paths:
            split = parquet_path.stem
            for row_index, row in enumerate(parquet.read_table(parquet_path).to_pylist()):
                context = row.get("context")
                questions = row.get("questions")
                answers = row.get("answers")
                metadata = row.get("metadata") or {}
                if not isinstance(context, str) or not isinstance(questions, list):
                    raise ValueError(f"{split} row {row_index} has invalid context/questions")
                if not isinstance(answers, list) or len(questions) != len(answers):
                    raise ValueError(f"{split} row {row_index} has mismatched questions/answers")
                corpus_id = f"{split.lower()}-{row_index}"
                sessions = [
                    ExternalSession(
                        session_id=f"chunk-{chunk_index}",
                        date=base + timedelta(seconds=chunk_index),
                        turns=[
                            ExternalTurn(
                                role="user",
                                content=chunk,
                                turn_id=f"chunk-{chunk_index}:0",
                            )
                        ],
                    )
                    for chunk_index, chunk in enumerate(_chunks(context))
                ]
                ids = metadata.get("qa_pair_ids") or metadata.get("question_ids") or []
                types = metadata.get("question_types") or []
                source_name = str(metadata.get("source") or split)
                for question_index, question in enumerate(questions):
                    gold_answers = _strings(answers[question_index])
                    question_id = (
                        str(ids[question_index])
                        if question_index < len(ids)
                        else f"{corpus_id}-q{question_index}"
                    )
                    examples.append(
                        ExternalBenchmarkExample(
                            dataset=self.name,
                            example_id=question_id,
                            question_id=question_id,
                            question_type=source_name,
                            question=str(question),
                            answer=gold_answers[0] if gold_answers else "",
                            sessions=sessions,
                            metadata={
                                "answers": gold_answers,
                                "benchmark_split": split,
                                "corpus_id": corpus_id,
                                "question_type": (
                                    str(types[question_index])
                                    if question_index < len(types)
                                    else "unknown"
                                ),
                                "source": source_name,
                                "keypoints": [
                                    str(item) for item in metadata.get("keypoints") or []
                                ],
                            },
                        )
                    )
        return examples

    def validate(self, path: str | Path) -> AdapterValidation:
        try:
            examples = self.load(path)
        except Exception as exc:
            return AdapterValidation(
                dataset=self.name,
                path=str(path),
                valid=False,
                errors=[str(exc)],
            )
        return _validation(self.name, path, examples)
