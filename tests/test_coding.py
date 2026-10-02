import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeClient, make_history, make_response

from memstudy.arms.base import ArmContext
from memstudy.budget import Budget
from memstudy.coding import (
    OBSERVATION_CAP_TOKENS,
    GradeResult,
    GradingError,
    SweContextBenchGrader,
    cap_observation,
    cap_query,
    image_for_instance,
    parse_command,
    parse_report,
    run_coding,
    task_prompt,
)
from memstudy.llm import ModelCaller
from memstudy.metering import CostSink
from memstudy.schema import Item
from memstudy.store import ResultStore


def _item(n=0):
    return Item(
        item_id=f"repo-{n}", bench="swectx", history_id="o__a", category="o/a",
        question="fix the bug", gold="diff", kind="coding",
        meta={"swebench_instance": {"instance_id": f"repo-{n}"}},
    )


def test_parse_command_takes_the_fenced_bash_block():
    assert parse_command("think\n```bash\nls -la\n```") == "ls -la"
    assert parse_command("no block here") is None
    assert parse_command("```bash\nsubmit\n```") == "submit"


def test_observation_cap_is_explicit_not_silent():
    short = "x " * 10
    assert cap_observation(short) == short
    long = "word " * (OBSERVATION_CAP_TOKENS * 2)
    capped = cap_observation(long)
    assert "[output truncated:" in capped


def test_query_cap_reports_whether_it_cut():
    text, cut = cap_query("short")
    assert (text, cut) == ("short", False)
    _, cut = cap_query("word " * 20000)
    assert cut is True


def test_memory_block_only_appears_when_there_is_memory():
    assert "<memory>" not in task_prompt(_item(), "")
    assert "<memory>\n- fact\n</memory>" in task_prompt(_item(), "- fact")


class FakeSandbox:
    def __init__(self):
        self.commands = []
        self.closed = False

    def run(self, command, timeout):
        self.commands.append(command)
        return 0, "ok"

    def diff(self):
        return "diff --git a b"

    def close(self):
        self.closed = True


class AlwaysResolves:
    def grade(self, item, patch):
        return GradeResult("resolved" if patch else "unresolved")


class Undecidable:
    def grade(self, item, patch):
        raise GradingError("container failed to start")


class FakeMemory:
    name = "B"

    def __init__(self):
        self.queries = []

    def prepare(self, history):
        return None

    def context(self, item, history):
        self.queries.append(item.question)
        return ArmContext(text="- remembered", context_tokens=2, cache_prefix=False,
                          retrieved=["m1"], retrieval_cost=CostSink())


def _caller(prices, tmp_path):
    script = iter(["```bash\ngrep -r bug .\n```", "no block", "```bash\nsubmit\n```"])
    client = FakeClient(lambda kw: make_response(text=next(script)))
    return client, ModelCaller(client, prices["gpt-6-luna"], Budget(tmp_path / "l.jsonl"))


def test_agent_loop_runs_commands_then_submits_and_is_graded(prices, cfg, tmp_path):
    client, caller = _caller(prices, tmp_path)
    sandboxes = []

    def make_sandbox(item):
        sandboxes.append(FakeSandbox())
        return sandboxes[0]

    store = ResultStore(tmp_path / "res")
    counts = run_coding(
        items=[_item()], histories={"o__a": make_history("o__a", "swectx")}, arm=None,
        arm_name="off", caller=caller, grader=AlwaysResolves(), make_sandbox=make_sandbox,
        store=store, stage="pilot", run_id="r1", cfg=cfg,
    )
    assert counts["completed"] == 1
    record = store.read_items("off", "swectx")[0]
    assert record["resolved"] is True and record["steps"] == 3 and record["submitted"] is True
    assert sandboxes[0].commands == ["grep -r bug ."] and sandboxes[0].closed
    first_prompt = client.responses.calls[0]["input"][1]["content"][0]["text"]
    assert "<memory>" not in first_prompt


def test_memory_on_arm_queries_its_own_repo_and_injects_memory(prices, cfg, tmp_path):
    client, caller = _caller(prices, tmp_path)
    arm = FakeMemory()
    store = ResultStore(tmp_path / "res")
    run_coding(
        items=[_item()], histories={"o__a": make_history("o__a", "swectx")}, arm=arm,
        arm_name="B", caller=caller, grader=AlwaysResolves(), make_sandbox=lambda i: FakeSandbox(),
        store=store, stage="pilot", run_id="r1", cfg=cfg,
    )
    assert arm.queries == ["fix the bug"]
    assert "- remembered" in client.responses.calls[0]["input"][1]["content"][0]["text"]
    assert store.read_items("B", "swectx")[0]["memory"]["retrieved"] == ["m1"]


def test_coding_resume_does_not_rerun_finished_tasks(prices, cfg, tmp_path):
    client, caller = _caller(prices, tmp_path)
    store = ResultStore(tmp_path / "res")
    kwargs = dict(
        items=[_item()], histories={"o__a": make_history("o__a", "swectx")}, arm=None,
        arm_name="off", caller=caller, grader=AlwaysResolves(),
        make_sandbox=lambda i: FakeSandbox(), store=store, stage="pilot", run_id="r1", cfg=cfg,
    )
    run_coding(**kwargs)
    calls = len(client.responses.calls)
    assert run_coding(**kwargs)["skipped_existing"] == 1
    assert len(client.responses.calls) == calls


def test_empty_patch_is_not_resolved(prices, cfg, tmp_path):
    client, caller = _caller(prices, tmp_path)

    class EmptyDiff(FakeSandbox):
        def diff(self):
            return "  "

    graded = []

    class Spy:
        def grade(self, item, patch):
            graded.append(patch)
            return GradeResult("resolved")

    store = ResultStore(tmp_path / "res")
    run_coding(
        items=[_item()], histories={"o__a": make_history("o__a", "swectx")}, arm=None,
        arm_name="off", caller=caller, grader=Spy(), make_sandbox=lambda i: EmptyDiff(),
        store=store, stage="pilot", run_id="r1", cfg=cfg,
    )
    assert graded == [] and store.read_items("off", "swectx")[0]["resolved"] is False


def test_grading_failure_is_an_error_not_an_unresolved_task(prices, cfg, tmp_path):
    client, caller = _caller(prices, tmp_path)
    store = ResultStore(tmp_path / "res")
    counts = run_coding(
        items=[_item()], histories={"o__a": make_history("o__a", "swectx")}, arm=None,
        arm_name="off", caller=caller, grader=Undecidable(), make_sandbox=lambda i: FakeSandbox(),
        store=store, stage="pilot", run_id="r1", cfg=cfg,
    )
    assert counts["errors"] == 1 and counts["completed"] == 0
    assert store.read_items("off", "swectx") == []
    assert "container failed" in store.read_errors("off", "swectx")[0]["error"]


def test_report_parsing_separates_resolved_unresolved_and_infrastructure():
    report = {
        "resolved_ids": ["a"],
        "unresolved_ids": ["b", "c", "d", "e"],
        "a": {"instance_id": "a", "resolved": True},
        "b": {"instance_id": "b", "resolved": False},
        "c": {"instance_id": "c", "resolved": False, "error": "Patch failed"},
        "d": {"instance_id": "d", "resolved": False, "error": "Hardened image not found for instance: d"},
        "e": {"instance_id": "e", "resolved": False, "error": "x", "failure_type": "verifier_setup_failed"},
    }
    assert parse_report(report, "a").resolved is True
    assert parse_report(report, "b").status == "unresolved"
    assert parse_report(report, "c").status == "unresolved"  # the model patch did not apply
    for bad in ("d", "e", "missing"):
        with pytest.raises(GradingError):
            parse_report(report, bad)


def test_image_name_matches_the_benchmark_harness():
    assert image_for_instance("django__django-11776") == "jiayuanz3/swecontextbench:django.django-11776"
    assert image_for_instance("Astropy__Astropy-15082").endswith(":astropy.astropy-15082")


class FakeHarness:
    """Stands in for subprocess.run; plays the two benchmark harness modules."""

    def __init__(self, resolved=True, rc=0):
        self.calls, self.resolved, self.rc = [], resolved, rc

    def __call__(self, cmd, cwd, env, **kwargs):
        self.calls.append((cmd, Path(cwd), env))
        module = cmd[2]
        if module.endswith("run_evaluation") and self.rc == 0:
            run_id = cmd[cmd.index("--run_id") + 1]
            key = "resolved_ids" if self.resolved else "unresolved_ids"
            (Path(cwd) / f"{run_id}.json").write_text(json.dumps({key: ["repo-0"]}))
        return SimpleNamespace(returncode=self.rc, stderr="boom", stdout="")


def _grader(tmp_path, runner):
    return SweContextBenchGrader(tmp_path / "bench", "cases/Lite", tmp_path / "grading", "py", runner)


def test_grader_follows_the_benchmarks_own_evaluation_steps(tmp_path):
    runner = FakeHarness()
    result = _grader(tmp_path, runner).grade(_item(), "diff --git a b")
    assert result.resolved
    combine, evaluate = runner.calls
    assert combine[0][2] == "swebench_memory.harness.combine_instances"
    assert str(tmp_path / "bench" / "cases/Lite") in combine[0]
    assert evaluate[0][2] == "swebench_memory.harness.run_evaluation"
    assert evaluate[0][evaluate[0].index("--dataset_name") + 1] == "batch_dataset.json"
    assert str(tmp_path / "bench") in evaluate[2]["PYTHONPATH"]
    pred = json.loads((tmp_path / "grading/repo-0/predictions/repo-0_preds.json").read_text())
    assert pred["repo-0"] == {
        "model_name_or_path": "memstudy", "instance_id": "repo-0", "model_patch": "diff --git a b"
    }


def test_grader_reads_an_existing_report_instead_of_rerunning(tmp_path):
    runner = FakeHarness()
    grader = _grader(tmp_path, runner)
    grader.grade(_item(), "p")
    calls = len(runner.calls)
    assert grader.grade(_item(), "p").resolved and len(runner.calls) == calls


def test_grader_reports_an_unresolved_task(tmp_path):
    assert _grader(tmp_path, FakeHarness(resolved=False)).grade(_item(), "p").status == "unresolved"


def test_grader_raises_when_the_harness_process_fails(tmp_path):
    with pytest.raises(GradingError, match="exit code 3"):
        _grader(tmp_path, FakeHarness(rc=3)).grade(_item(), "p")


def test_grader_resolves_relative_paths_before_changing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = FakeHarness()
    SweContextBenchGrader(Path("bench"), "cases/Lite", Path("grading"), "py", runner).grade(_item(), "p")
    cmd, cwd, env = runner.calls[0]
    assert Path(cwd).is_absolute() and env["PYTHONPATH"].startswith(str(tmp_path.resolve() / "bench"))
