"""Pinned LLM judges matching the official benchmark implementations."""

from __future__ import annotations

import ast
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from evals.schemas import ExternalBenchmarkExample
from scopegraph.llm.transport import ModelTransport

LONGMEMEVAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
MEMORYAGENTBENCH_SUMMARY_JUDGE_MODEL = "gpt-4o-2024-05-13"


@dataclass(frozen=True)
class JudgeResult:
    metric: str
    score: float
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


@lru_cache(maxsize=2)
def _longmemeval_prompt_builder(path: Path) -> Callable[..., str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_anscheck_prompt"
        ),
        None,
    )
    if function is None:
        raise ValueError(f"Official LongMemEval prompt function missing from {path}")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return cast(Callable[..., str], namespace["get_anscheck_prompt"])


def longmemeval_prompt(
    example: ExternalBenchmarkExample,
    response: str,
    scorer_path: Path = Path("data/official_scorers/longmemeval_evaluate_qa.py"),
) -> str:
    task = str(example.metadata.get("question_type", example.question_type))
    return _longmemeval_prompt_builder(scorer_path)(
        task,
        example.question,
        example.answer,
        response,
        abstention="_abs" in example.question_id,
    )


def _source_constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {
        "fluency_prompt_book",
        "recall_prompt_book",
        "precision_prompt_book",
    }
    values: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    values[target.id] = value
    if set(values) != wanted:
        raise ValueError(f"Official MemoryAgentBench prompts missing from {path}")
    return values


def _last_json(text: str) -> dict[str, Any]:
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Official judge returned no JSON object")


class OfficialBenchmarkJudge:
    def __init__(self, *, base_url: str, api_key: str, scorer_root: Path = Path("data")) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.scorer_root = scorer_root
        self.transport = ModelTransport(timeout=120, retries=3)

    async def aclose(self) -> None:
        await self.transport.aclose()

    async def _complete(
        self, prompt: str, *, model: str, temperature: float, max_tokens: int
    ) -> tuple[str, dict[str, int]]:
        body = await self.transport.post(
            f"{self.base_url}/chat/completions",
            api_key=self.api_key,
            payload={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Judge response content must be text")
        return content.strip(), dict(self.transport.last_usage)

    async def judge(
        self, dataset: str, example: ExternalBenchmarkExample, response: str
    ) -> JudgeResult | None:
        if dataset == "longmemeval" or (
            dataset == "memoryagentbench" and "longmemeval" in example.question_type
        ):
            started = time.perf_counter()
            output, usage = await self._complete(
                longmemeval_prompt(
                    example,
                    response,
                    self.scorer_root / "official_scorers/longmemeval_evaluate_qa.py",
                ),
                model=LONGMEMEVAL_JUDGE_MODEL,
                temperature=0,
                max_tokens=10,
            )
            return JudgeResult(
                metric="llm_judge_accuracy",
                score=float("yes" in output.lower()),
                model=LONGMEMEVAL_JUDGE_MODEL,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        if dataset == "memoryagentbench" and "infbench" in example.question_type:
            return await self._judge_summary(example, response)
        return None

    async def _judge_summary(
        self, example: ExternalBenchmarkExample, response: str
    ) -> JudgeResult:
        script = self.scorer_root / "official_scorers/memoryagentbench_summarization_evaluate.py"
        prompts = _source_constants(script)
        keypoints = [str(item) for item in example.metadata.get("keypoints", [])]
        inputs = [
            prompts["fluency_prompt_book"].format(text=response.strip()),
            prompts["recall_prompt_book"].format(
                keypoints="\n".join(f"{index + 1}. {item}" for index, item in enumerate(keypoints)),
                summary=response.strip(),
            ),
            prompts["precision_prompt_book"].format(
                expert_summary=example.answer, summary=response.strip()
            ),
        ]
        started = time.perf_counter()
        outputs = []
        input_tokens = 0
        output_tokens = 0
        for prompt in inputs:
            output, usage = await self._complete(
                prompt,
                model=MEMORYAGENTBENCH_SUMMARY_JUDGE_MODEL,
                temperature=0.1,
                max_tokens=4096,
            )
            outputs.append(_last_json(output))
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
        fluency, recall_result, precision_result = outputs
        recall = float(recall_result["recall"]) / len(keypoints) if keypoints else 0.0
        sentence_count = float(precision_result["sentence_count"])
        precision = (
            float(precision_result["precision"]) / sentence_count if sentence_count else 0.0
        )
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        score = float(fluency["fluency"]) * f1
        return JudgeResult(
            metric="gpt-4-f1",
            score=score,
            model=MEMORYAGENTBENCH_SUMMARY_JUDGE_MODEL,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
