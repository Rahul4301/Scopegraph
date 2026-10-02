import json
from pathlib import Path

import pytest
from conftest import ROOT

from memstudy.loaders.locomo import ABSTAIN_GOLD, load_locomo
from memstudy.loaders.longmemeval import load_longmemeval
from memstudy.loaders.swectx import build_swectx, overlap_ids, repo_history_id


def test_locomo_loader_splits_primary_and_adversarial(tmp_path):
    raw = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "1 May",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}],
            },
            "qa": [
                {"question": "q1", "answer": 7, "evidence": ["D1:1"], "category": 2},
                {"question": "q2", "category": 5, "adversarial_answer": "x", "evidence": []},
            ],
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(raw))
    histories, items = load_locomo(path)
    assert histories["conv-1"].user_id == "locomo_conv-1"
    assert histories["conv-1"].sessions[0].timestamp == "1 May"
    first, adversarial = items
    assert first.primary and first.gold == "7" and first.category == "cat2"
    assert not adversarial.primary and adversarial.gold == ABSTAIN_GOLD
    assert adversarial.meta["abstention"] is True


def test_longmemeval_loader_one_history_per_question(tmp_path):
    raw = [
        {
            "question_id": "abc_abs",
            "question_type": "multi-session",
            "question": "q",
            "question_date": "2023/05/30 (Tue) 23:40",
            "answer": "a",
            "answer_session_ids": ["s1"],
            "haystack_dates": ["2023/05/20 (Sat) 10:00"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[{"role": "user", "content": "hello"}]],
        }
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(raw))
    histories, items = load_longmemeval(path)
    assert histories["abc_abs"].user_id == "longmemeval_abc_abs"
    assert items[0].meta["abstention"] is True and items[0].question_date.startswith("2023/05/30")


LOCOMO = ROOT / "data/locomo/locomo10.json"


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo data not downloaded")
def test_real_locomo_matches_the_preregistered_counts():
    histories, items = load_locomo(Path(LOCOMO))
    assert len(histories) == 10
    assert sum(i.primary for i in items) == 1540
    assert sum(not i.primary for i in items) == 446
    assert len({h.user_id for h in histories.values()}) == 10


LME = ROOT / "data/longmemeval/longmemeval_s_cleaned.json"


@pytest.mark.skipif(not LME.exists(), reason="LongMemEval-S data not downloaded")
def test_real_longmemeval_has_500_questions_each_with_its_own_history():
    histories, items = load_longmemeval(Path(LME))
    assert len(items) == 500 and len(histories) == 500
    assert len({h.user_id for h in histories.values()}) == 500


def _task(instance_id, repo, created="2020-01-01"):
    return {
        "instance_id": instance_id, "repo": repo, "created_at": created,
        "problem_statement": f"issue {instance_id}", "patch": f"diff {instance_id}",
        "base_commit": "c", "test_patch": "t", "hints_text": "", "version": "1",
        "FAIL_TO_PASS": "[]", "PASS_TO_PASS": "[]", "environment_setup_commit": "e",
    }


def test_swectx_histories_are_repo_scoped():
    experience = [_task("a-1", "o/a"), _task("a-2", "o/a", "2020-02-01"), _task("b-1", "o/b")]
    related = [_task("a-9", "o/a"), _task("c-9", "o/c")]
    links = [{"related_instance_id": "a-9", "experience_instance_id": "a-1"}]
    histories, items = build_swectx(experience, related, links)
    assert set(histories) == {"o__a", "o__b"}
    assert histories["o__a"].user_id == "swectx_o__a"
    assert [i.item_id for i in items] == ["a-9"]  # c-9 has no experience in its repo
    task_ids = {s.session_id for s in histories[items[0].history_id].sessions}
    assert task_ids == {"a-1", "a-2"} and "a-9" not in task_ids
    assert items[0].meta["oracle_experience_ids"] == ["a-1"]
    assert repo_history_id("o/a") == "o__a"


def test_swectx_excludes_a_related_task_that_is_in_the_experience_pool():
    experience = [_task("a-1", "o/a"), _task("a-2", "o/a")]
    related = [_task("a-1", "o/a"), _task("a-9", "o/a")]
    _, items = build_swectx(experience, related, [])
    assert [i.item_id for i in items] == ["a-9"]
    assert overlap_ids(experience, related) == ["a-1"]
