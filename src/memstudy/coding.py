"""Coding benchmark: a minimal bash agent on GPT-6 Luna, graded by the benchmark's own tests.

Arms: "off" (no memory) and "B" (Mem0 memory of the same repository). The agent protocol is one
fenced bash block per turn; the command `submit` ends the episode and the sandbox diff is the
patch. Grading applies the patch and runs the benchmark's tests, nothing else.

The Docker sandbox and the benchmark grader are written against the documented interfaces but are
not exercised offline; they first run in the Phase 2 pilot after approval. Tests use fakes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from memstudy.arms.base import Arm
from memstudy.budget import BudgetExceeded
from memstudy.llm import IncompleteResponse, ModelCaller
from memstudy.schema import History, Item, Usage
from memstudy.store import ResultStore
from memstudy.tokens import count_tokens, encoding

MAX_STEPS = 40
OBSERVATION_CAP_TOKENS = 4000
QUERY_CAP_TOKENS = 6000
_BASH_BLOCK = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)

AGENT_INSTRUCTIONS = (
    "You fix a bug or implement a change in a repository checked out at /testbed. "
    "Each turn, reply with your reasoning and exactly one command in a fenced block:\n"
    "```bash\n<command>\n```\n"
    "You see the command output next. Do not edit test files. "
    "When the fix is complete, reply with the single command `submit`."
)


class Sandbox(Protocol):
    def run(self, command: str, timeout: int) -> tuple[int, str]: ...

    def diff(self) -> str: ...

    def close(self) -> None: ...


class DockerSandbox:
    """One container per task. The repository is at /testbed. The benchmark images are amd64, so on
    Apple Silicon they run under emulation (slow but the same image the grader uses)."""

    def __init__(self, image: str, workdir: str = "/testbed", platform: str = "linux/amd64") -> None:
        self.workdir = workdir
        started = subprocess.run(
            ["docker", "run", "-d", "--rm", "--platform", platform, image, "sleep", "infinity"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.container = started.stdout.strip()

    def run(self, command: str, timeout: int = 120) -> tuple[int, str]:
        try:
            done = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.container, "bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 124, f"command timed out after {timeout}s"
        return done.returncode, done.stdout + done.stderr

    def diff(self) -> str:
        return self.run("git add -A && git diff --cached", 60)[1]

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)


class GradingError(RuntimeError):
    """The harness could not decide. Never recorded as unresolved: an infrastructure failure
    would otherwise bias the resolved rate downward."""


@dataclass(frozen=True)
class GradeResult:
    status: str  # "resolved" or "unresolved"; anything else raises GradingError

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


class Grader(Protocol):
    def grade(self, item: Item, patch: str) -> GradeResult: ...


PATCH_FAILED = "Patch failed"  # the model patch did not apply: a real, unresolved outcome


def parse_report(report: dict[str, Any], instance_id: str) -> GradeResult:
    """Map a harness run report to a result.

    The benchmark harness lists resolved_ids and unresolved_ids, and puts a per-instance record
    under the instance id. A record with an `error` is an infrastructure failure (image missing,
    no tests, image commit or verifier setup failed) and raises, except "Patch failed", which
    means the model's patch did not apply and counts as unresolved.
    """
    record = report.get(instance_id)
    if isinstance(record, dict) and record.get("error"):
        if record["error"] == PATCH_FAILED:
            return GradeResult("unresolved")
        raise GradingError(f"{instance_id}: {record['error']}")
    if instance_id in report.get("resolved_ids", []):
        return GradeResult("resolved")
    if instance_id in report.get("unresolved_ids", []):
        return GradeResult("unresolved")
    raise GradingError(f"{instance_id} is absent from the harness report")


def image_for_instance(instance_id: str) -> str:
    """Docker image the benchmark harness uses for a task (find_docker_image in its harness)."""
    return f"jiayuanz3/swecontextbench:{instance_id.replace('__', '.').lower()}"


class SweContextBenchGrader:
    """Grades with the benchmark's own harness, the way its evaluation.sh does.

    SWE Context Bench ships a fork of the SWE-bench harness (package `swebench_memory`, in
    github.com/jiayuanz3/SWEContextBench) with its own Docker images (jiayuanz3/swecontextbench).
    The stock `swebench` package (5.x) cannot grade this dataset: it needs `image`, `eval_script`
    and `log_parser` fields these rows do not have. Steps per task, mirroring evaluation.sh:

    1. write `<instance_id>_preds.json` in the format the README specifies,
    2. `python -m swebench_memory.harness.combine_instances` to build the dataset and predictions,
    3. `python -m swebench_memory.harness.run_evaluation`,
    4. read `<run_id>.json` and map the instance to resolved or unresolved.

    Run directories are kept under work_root for audit and are never overwritten; an existing
    report is read instead of re-running. Needs a pinned clone of the benchmark repository and its
    dependencies, plus the Docker images, all after the owner's approval. Not run offline: tests
    use an injected runner.
    """

    def __init__(
        self,
        repo_dir: Path,
        cases_dir: str,
        work_root: Path,
        python: str = sys.executable,
        runner: Any = subprocess.run,
        timeout: int = 3600,
    ) -> None:
        # Absolute, because the harness runs with its own working directory.
        self.repo_dir = Path(repo_dir).resolve()
        self.cases_dir = cases_dir
        self.work_root = Path(work_root).resolve()
        self.python = python
        self.runner = runner
        self.timeout = timeout

    def _run(self, args: list[str], cwd: Path) -> None:
        env = {**os.environ, "PYTHONPATH": f"{self.repo_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        done = self.runner(
            [self.python, "-m", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if done.returncode != 0:
            raise GradingError(f"{args[0]} failed with exit code {done.returncode}: {done.stderr[-500:]}")

    def grade(self, item: Item, patch: str) -> GradeResult:
        run_id = f"memstudy-{item.item_id}"
        work = self.work_root / item.item_id
        report_file = work / f"{run_id}.json"
        if not report_file.exists():
            preds = work / "predictions"
            preds.mkdir(parents=True, exist_ok=True)
            prediction = {
                item.item_id: {
                    "model_name_or_path": "memstudy",
                    "instance_id": item.item_id,
                    "model_patch": patch,
                }
            }
            (preds / f"{item.item_id}_preds.json").write_text(json.dumps(prediction))
            self._run(
                [
                    "swebench_memory.harness.combine_instances",
                    "--instances",
                    str(self.repo_dir / self.cases_dir),
                    "--predictions",
                    str(preds),
                    "--dataset-output",
                    "batch_dataset.json",
                    "--predictions-output",
                    "batch_predictions.json",
                ],
                work,
            )
            self._run(
                [
                    "swebench_memory.harness.run_evaluation",
                    "--dataset_name",
                    "batch_dataset.json",
                    "--predictions_path",
                    "batch_predictions.json",
                    "--run_id",
                    run_id,
                ],
                work,
            )
        if not report_file.exists():
            raise GradingError(f"harness wrote no report at {report_file}")
        return parse_report(json.loads(report_file.read_text()), item.item_id)


def parse_command(reply: str) -> str | None:
    match = _BASH_BLOCK.search(reply)
    return match.group(1).strip() if match else None


def cap_observation(text: str) -> str:
    """Observation cap with an explicit marker, so the agent knows output was cut."""
    ids = encoding().encode(text, disallowed_special=())
    if len(ids) <= OBSERVATION_CAP_TOKENS:
        return text
    kept = encoding().decode(ids[:OBSERVATION_CAP_TOKENS])
    return f"{kept}\n[output truncated: {len(ids) - OBSERVATION_CAP_TOKENS} tokens omitted]"


def cap_query(text: str) -> tuple[str, bool]:
    """The memory query is the issue text. Cap it for the embedder and record that it was cut."""
    ids = encoding().encode(text, disallowed_special=())
    if len(ids) <= QUERY_CAP_TOKENS:
        return text, False
    return encoding().decode(ids[:QUERY_CAP_TOKENS]), True


def _msg(role: str, kind: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": kind, "text": text}]}


def task_prompt(item: Item, memory: str) -> str:
    memory_block = f"<memory>\n{memory}\n</memory>\n\n" if memory else ""
    return f"{memory_block}<issue>\n{item.question}\n</issue>"


def run_agent(
    *,
    item: Item,
    memory: str,
    caller: ModelCaller,
    sandbox: Sandbox,
    stage: str,
    arm: str,
    cfg: dict[str, Any],
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    messages = [
        _msg("developer", "input_text", AGENT_INSTRUCTIONS),
        _msg("user", "input_text", task_prompt(item, memory)),
    ]
    usage = Usage()
    cost = 0.0
    start = time.perf_counter()
    steps = 0
    submitted = False
    for steps in range(1, max_steps + 1):
        result = caller.call(
            stage=stage,
            tag={"arm": arm, "item": item.item_id, "purpose": "agent", "step": steps},
            input_items=messages,
            max_output_tokens=cfg["reader"]["max_output_tokens"],
            reasoning_effort=cfg["reader"]["reasoning_effort"],
            temperature=cfg["reader"]["temperature"],
            extra_body={
                "prompt_cache_key": f"{arm}-{item.item_id}",
                "prompt_cache_options": {"mode": "implicit", "ttl": cfg["reader"]["cache_ttl"]},
            },
        )
        usage = usage + result.usage
        cost += result.cost_usd
        messages.append(_msg("assistant", "output_text", result.text))
        command = parse_command(result.text)
        if command == "submit":
            submitted = True
            break
        if command is None:
            messages.append(_msg("user", "input_text", "Reply with exactly one ```bash block."))
            continue
        code, output = sandbox.run(command, 120)
        messages.append(_msg("user", "input_text", f"exit code {code}\n{cap_observation(output)}"))
    return {
        "patch": sandbox.diff(),
        "steps": steps,
        "submitted": submitted,
        "usage": usage.model_dump(),
        "cost_usd": cost,
        "seconds": time.perf_counter() - start,
    }


def run_coding(
    *,
    items: list[Item],
    histories: dict[str, History],
    arm: Arm | None,
    arm_name: str,
    caller: ModelCaller,
    grader: Grader,
    make_sandbox: Any,
    store: ResultStore,
    stage: str,
    run_id: str,
    cfg: dict[str, Any],
) -> dict[str, int]:
    """Run each task once. arm=None is the memory-off condition."""
    counts = {"completed": 0, "skipped_existing": 0, "errors": 0}
    for item in items:
        if store.has_item(arm_name, item.bench, item.item_id):
            counts["skipped_existing"] += 1
            continue
        memory, memory_meta = "", {}
        if arm is not None:
            history = histories[item.history_id]
            arm.prepare(history)
            query, cut = cap_query(item.question)
            ctx = arm.context(item.model_copy(update={"question": query}), history)
            memory = ctx.text
            memory_meta = {
                "retrieved": ctx.retrieved,
                "memory_tokens": ctx.context_tokens,
                "query_truncated": cut,
                "retrieval_cost": ctx.retrieval_cost.to_dict(),
            }
        sandbox = make_sandbox(item)
        try:
            outcome = run_agent(
                item=item,
                memory=memory,
                caller=caller,
                sandbox=sandbox,
                stage=stage,
                arm=arm_name,
                cfg=cfg,
            )
            grade = (
                grader.grade(item, outcome["patch"])
                if outcome["patch"].strip()
                else GradeResult("unresolved")  # an empty patch cannot resolve the task
            )
        except BudgetExceeded:
            raise
        except (IncompleteResponse, GradingError) as err:
            store.write_error(arm_name, item.bench, item.item_id, {"error": str(err)})
            counts["errors"] += 1
            continue
        finally:
            sandbox.close()
        store.write_item(
            arm_name,
            item.bench,
            item.item_id,
            {
                "run_id": run_id,
                "arm": arm_name,
                "bench": item.bench,
                "item_id": item.item_id,
                "history_id": item.history_id,
                "category": item.category,
                "resolved": grade.resolved,
                "grade_status": grade.status,
                "memory": memory_meta,
                "memory_tokens_counted": count_tokens(memory),
                **outcome,
            },
        )
        counts["completed"] += 1
    return counts
