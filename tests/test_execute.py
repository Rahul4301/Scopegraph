"""One named run end to end with fake clients: file naming, report content, immutability, resume."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from conftest import ROOT, FakeClient, make_history, make_item, make_response

import memstudy.preflight as pf
from memstudy.arms.full_context import FullContextArm
from memstudy.arms.mem0_arm import Mem0Arm
from memstudy.arms.rag import chunk_history
from memstudy.budget import Budget, BudgetExceeded
from memstudy.cli import main, stage_gate
from memstudy.config import config_hash
from memstudy.execute import (
    ResultExists,
    execute_run,
    resolve_system,
    result_name,
    select_cases,
)
from memstudy.judge import Judge
from memstudy.llm import ModelCaller
from memstudy.metering import CostSink, Meter
from memstudy.reader import Reader
from memstudy.schema import History


def handler(kw):
    if kw["model"] == "gpt-6-luna":
        return make_response(text="alpha", input_tokens=4000, cached=0, output=5)
    return make_response(text='{"verdict": "CORRECT"}', model="gpt-5-nano-test", input_tokens=200, output=40)


def rig(prices, budget, cfg, fn=handler):
    client = FakeClient(fn)
    reader = Reader(ModelCaller(client, prices["gpt-6-luna"], budget), cfg)
    judge = Judge(ModelCaller(client, prices["gpt-5-nano"], budget), cfg)
    return client, reader, judge


def run(prices, budget, cfg, tmp_path, n_items=2, num_cases=2, fn=handler):
    client, reader, judge = rig(prices, budget, cfg, fn)
    items = [make_item("h1", i) for i in range(n_items)]
    path = execute_run(
        system="no_memory", eval_name="locomo", num_cases=num_cases,
        arm=FullContextArm(100_000, 0.05), cases=items, histories={"h1": make_history("h1")},
        reader=reader, judge=judge, stage="smoke", budget_stage="pilot",
        results_dir=tmp_path, cfg=cfg, cfg_hash=config_hash(cfg),
    )
    return client, path


def test_result_names_follow_system_eval_numcases():
    assert result_name("mem0", "locomo", 1) == "mem0_locomo_1"
    assert result_name("no_memory", "longmemeval", None) == "no_memory_longmemeval"
    assert resolve_system("A") == ("no_memory", "A") and resolve_system("rag") == ("rag", "C")
    assert resolve_system("B")[0] == "mem0"
    with pytest.raises(ValueError):
        resolve_system("supermemory")


def test_select_cases_takes_the_first_primary_items():
    items = [make_item("h", i) for i in range(4)]
    items[1] = items[1].model_copy(update={"primary": False})
    assert [i.item_id for i in select_cases(items, 2, False)] == ["h-0000", "h-0002"]
    assert len(select_cases(items, None, False)) == 3 and len(select_cases(items, None, True)) == 4
    with pytest.raises(ValueError):
        select_cases(items, 0, False)


def test_smoke_run_writes_one_readable_file_named_by_system_eval_and_cases(prices, budget, cfg, tmp_path):
    _, path = run(prices, budget, cfg, tmp_path)
    assert path == tmp_path / "no_memory_locomo_2.json"
    report = json.loads(path.read_text())
    assert list(report) == ["run", "summary", "cases", "metric_notes"]
    assert report["run"]["memory_system"] == "no_memory" and report["run"]["num_cases"] == 2
    assert report["run"]["stage"] == "smoke"
    s = report["summary"]
    assert s["cases"]["selected"] == 2 and s["cases"]["completed"] == 2 and s["cases"]["errored"] == 0
    assert s["accuracy"]["judge"] == 1.0 and s["accuracy"]["exact_match"] == 1.0
    assert s["accuracy"]["gold_in_context"] == 1.0 and s["by_category"]["cat1"]["n"] == 2
    assert s["tokens"]["reader_total"]["input"] == 8000 and s["tokens"]["answer_mean"] == 5
    assert s["cost_usd"]["total"] > 0 and s["cost_usd"]["per_item"] == pytest.approx(s["cost_usd"]["total"] / 2)
    assert s["latency_ms"]["reader"]["p50"] is not None and s["ingest"]["histories"] == 0
    first = report["cases"][0]
    assert first["case"] == 1 and first["status"] == "ok" and first["answer"] == "alpha"
    assert first["gold"] == ["alpha"] and first["correct"] is True and first["cost_usd"]["total"] > 0


def test_full_run_is_named_without_a_case_count(prices, budget, cfg, tmp_path):
    _, path = run(prices, budget, cfg, tmp_path, num_cases=None)
    assert path.name == "no_memory_locomo.json"
    assert json.loads(path.read_text())["run"]["num_cases"] == "all"


def test_an_existing_result_is_never_overwritten_and_costs_nothing(prices, budget, cfg, tmp_path):
    client, path = run(prices, budget, cfg, tmp_path)
    before, calls = path.read_text(), len(client.responses.calls)
    second = FakeClient(handler)
    with pytest.raises(ResultExists):
        execute_run(
            system="no_memory", eval_name="locomo", num_cases=2, arm=FullContextArm(100_000, 0.05),
            cases=[make_item("h1", 0)], histories={"h1": make_history("h1")},
            reader=Reader(ModelCaller(second, prices["gpt-6-luna"], budget), cfg),
            judge=Judge(ModelCaller(second, prices["gpt-5-nano"], budget), cfg),
            stage="smoke", budget_stage="pilot", results_dir=tmp_path, cfg=cfg, cfg_hash="x",
        )
    assert path.read_text() == before and calls == 4 and not second.responses.calls


def test_a_stopped_run_keeps_raw_records_and_resumes_without_repeating_calls(prices, cfg, tmp_path):
    spent = Budget(tmp_path / "ledger.jsonl", stage_caps={"pilot": 0.0, "chat": 90.0})
    with pytest.raises(BudgetExceeded):
        run(prices, spent, cfg, tmp_path)
    assert not (tmp_path / "no_memory_locomo_2.json").exists()
    client, path = run(prices, Budget(tmp_path / "l2.jsonl"), cfg, tmp_path)
    assert len(client.responses.calls) == 4 and path.exists()


def test_failed_cases_stay_in_the_file_with_their_error(prices, budget, cfg, tmp_path):
    big = FullContextArm(10, 0.05)
    client, reader, judge = rig(prices, budget, cfg)
    path = execute_run(
        system="no_memory", eval_name="locomo", num_cases=1, arm=big, cases=[make_item("h1", 0)],
        histories={"h1": make_history("h1")}, reader=reader, judge=judge, stage="smoke",
        budget_stage="pilot", results_dir=tmp_path, cfg=cfg, cfg_hash="x",
    )
    report = json.loads(path.read_text())
    assert report["summary"]["cases"]["errored"] == 1 and report["summary"]["cases"]["does_not_fit"] == ["h1"]
    assert report["cases"][0]["status"] == "error" and report["cases"][0]["error"]["error"] == "does_not_fit"
    assert not client.responses.calls


def _document_history(text: str) -> History:
    return History(history_id="d1", bench="memoryagentbench", document=text)


def test_document_histories_reach_every_arm_verbatim(cfg, prices, tmp_path):
    from memstudy.schema import render_transcript

    doc = "Document 1:\n" + "\n".join(f"fact {i} is {i * 3}" for i in range(400))
    history = _document_history(doc)
    assert render_transcript(history) == doc
    assert len(chunk_history(history, 256)) > 1

    added = []
    memory = NS(add=lambda msgs, **kw: added.append((msgs, kw)), get_all=lambda **kw: {"results": []})
    meter = Meter(Budget(tmp_path / "l.jsonl"), prices, "pilot", CostSink(), {})
    small_chunks = {**cfg["arms"]["B"], "document_chunk_tokens": 200}
    arm = Mem0Arm(small_chunks, cfg["retrieval"], meter, memory, infer=False)
    arm.prepare(history)
    assert len(added) > 1 and all(m[0][0]["role"] == "user" for m in added)
    assert "\n".join(m[0][0]["content"] for m in added) == doc


def test_smoke_has_its_own_gate_and_the_cli_refuses_without_approval(monkeypatch):
    assert stage_gate("smoke", "locomo") == "stage_smoke"
    assert stage_gate("chat", "memoryagentbench") == "stage_chat_memoryagentbench"
    monkeypatch.setattr(pf, "prereg_committed", lambda tag=pf.PREREG_TAG: False)
    with pytest.raises(pf.NotApproved):
        main(["run", "--memory_system", "mem0", "--eval", "locomo", "--num_cases", "1"])


@pytest.mark.skipif(not (ROOT / "data/locomo/locomo10.json").exists(), reason="LoCoMo data not downloaded")
def test_dry_run_previews_without_gates_keys_or_calls(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    main(["run", "--memory_system", "no_memory", "--eval", "locomo", "--num_cases", "1", "--dry_run"])
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] and out["would_write"] == "results/no_memory_locomo_1.json" and out["cases"] == 1
    assert out["case_list"][0]["fits_window"] and out["reader_worst_case_usd_total"] > 0
    assert not Path("results/no_memory_locomo_1.json").exists()
