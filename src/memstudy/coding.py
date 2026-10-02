"""Coding benchmark: a minimal bash agent on GPT-6 Luna, graded by the benchmark's own tests.

Arms: "off" (no memory) and "B" (Mem0 memory of the same repository). The agent protocol is one
fenced bash block per turn; the command `submit` ends the episode and the sandbox diff is the
patch. Grading applies the patch and runs the benchmark's tests, nothing else.

The Docker sandbox and the SWE-bench grader are written against the documented interfaces but are
not exercised offline; they first run in the Phase 2 pilot after approval. Tests use fakes.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
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
    """One container per task. The repository is already at /testbed in SWE-bench images."""

    def __init__(self, image: str, workdir: str = "/testbed") -> None:
        self.workdir = workdir
        started = subprocess.run(
            ["docker", "run", "-d", "--rm", image, "sleep", "infinity"],
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


class Grader(Protocol):
    def grade(self, item: Item, patch: str) -> bool: ...


class SweBenchGrader:
    """Runs swebench.harness.run_evaluation on one prediction and reads the resolved list."""

    def __init__(self, namespace: str | None = None, timeout: int = 1800) -> None:
        self.namespace = namespace
        self.timeout = timeout

    def grade(self, item: Item, patch: str) -> bool:
        instance = item.meta["swebench_instance"]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "dataset.json").write_text(json.dumps([instance]))
            (work / "preds.json").write_text(
                json.dumps(
                    [
                        {
                            "instance_id": item.item_id,
                            "model_name_or_path": "memstudy",
                            "model_patch": patch,
                        }
                    ]
                )
            )
            cmd = [
                "python",
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                str(work / "dataset.json"),
                "--predictions_path",
                str(work / "preds.json"),
                "--instance_ids",
                item.item_id,
                "--run_id",
                "memstudy",
                "--max_workers",
                "1",
            ]
            if self.namespace:
                cmd += ["--namespace", self.namespace]
            subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=self.timeout)
            reports = list(work.glob("*.memstudy.json"))
            if not reports:
                return False
            return item.item_id in json.loads(reports[0].read_text()).get("resolved_ids", [])


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
            resolved = grader.grade(item, outcome["patch"]) if outcome["patch"].strip() else False
        except BudgetExceeded:
            raise
        except IncompleteResponse as err:
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
                "resolved": resolved,
                "memory": memory_meta,
                "memory_tokens_counted": count_tokens(memory),
                **outcome,
            },
        )
        counts["completed"] += 1
    return counts
