import json
from pathlib import Path

import pytest
from conftest import ROOT

from memstudy.loaders.locomo import ABSTAIN_GOLD, load_locomo
from memstudy.loaders.longmemeval import load_longmemeval


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
