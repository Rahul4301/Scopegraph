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


def _write_mab(tmp_path, competency, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), tmp_path / f"{competency}-00000-of-00001.parquet")


def test_memoryagentbench_loader_keeps_context_verbatim_and_all_answers(tmp_path):
    from memstudy.loaders.memoryagentbench import load_memoryagentbench
    from memstudy.schema import render_transcript

    context = "Document 1:\nline one\n\nDocument 2:\nline two  "
    meta = {"source": "ruler_qa1_197K", "qa_pair_ids": ["a0", "a1"], "demo": None}
    _write_mab(tmp_path, "Accurate_Retrieval", [
        {"context": context, "questions": ["q0", "q1"], "answers": [["France", "FR"], ["10th"]], "metadata": meta}
    ])
    _write_mab(tmp_path, "Conflict_Resolution", [
        {"context": "c", "questions": ["q"], "answers": [["x"]], "metadata": {"source": "factconsolidation_sh_6k"}},
        {"context": "c", "questions": ["q"], "answers": [["x"]], "metadata": {"source": "factconsolidation_sh_262k"}},
    ])
    _write_mab(tmp_path, "Long_Range_Understanding", [
        {"context": "l", "questions": ["q"], "answers": [["y"]],
         "metadata": {"source": "infbench_sum_eng_shots2", "demo": "D", "keypoints": ["k1", "k2"]}}
    ])
    _write_mab(tmp_path, "Test_Time_Learning", [
        {"context": "t", "questions": ["q"], "answers": [["7"]], "metadata": {"source": "icl_x"}}
    ])
    histories, items = load_memoryagentbench(tmp_path)
    first = histories["accurate_retrieval-ruler_qa1_197K-r00"]
    assert render_transcript(first) == context and first.user_id.startswith("memoryagentbench_")
    q0, q1 = items[0], items[1]
    assert q0.gold == "France" and q0.meta["answers"] == ["France", "FR"] and q0.meta["qa_pair_id"] == "a0"
    assert q1.item_id == "accurate_retrieval-ruler_qa1_197K-r00-q001"
    primary = {i.category: i.primary for i in items}
    assert primary == {
        "Accurate_Retrieval": True,
        "Conflict_Resolution": True,
        "Long_Range_Understanding": False,
        "Test_Time_Learning": False,
    }
    lru = next(i for i in items if i.category == "Long_Range_Understanding")
    assert lru.meta["demo"] == "D" and lru.meta["keypoints"] == ["k1", "k2"]
    # The 6k ablation row is dropped; the 262k row keeps its parquet row number in its id.
    assert "conflict_resolution-factconsolidation_sh_6k-r00" not in histories
    assert "conflict_resolution-factconsolidation_sh_262k-r01" in histories


def test_memoryagentbench_loader_flags_longmemeval_overlap_and_abstention(tmp_path):
    from memstudy.loaders.memoryagentbench import load_memoryagentbench

    meta = {
        "source": "longmemeval_s*", "question_ids": ["x_abs"], "question_dates": ["2023/05/30"],
        "question_types": ["multi-session"], "qa_pair_ids": ["p"],
    }
    def row(metadata):
        return [{"context": "c", "questions": ["q"], "answers": [["a"]], "metadata": metadata}]

    _write_mab(tmp_path, "Accurate_Retrieval", row(meta))
    for comp in ("Conflict_Resolution", "Long_Range_Understanding", "Test_Time_Learning"):
        _write_mab(tmp_path, comp, row({"source": "s"}))
    _, items = load_memoryagentbench(tmp_path)
    item = items[0]
    assert "*" not in item.item_id and item.question_date == "2023/05/30"
    assert item.meta["overlaps_longmemeval"] is True and item.meta["abstention"] is True


MAB = ROOT / "data/memoryagentbench/data"


@pytest.mark.skipif(not MAB.exists(), reason="MemoryAgentBench data not downloaded")
def test_real_memoryagentbench_matches_the_paper():
    """Hu et al. (ICLR 2026) Table 1: 2,071 questions; Table 6: contexts : questions per dataset."""
    from collections import Counter

    from memstudy.loaders.memoryagentbench import load_memoryagentbench

    histories, items = load_memoryagentbench(MAB)
    assert len(histories) == 130 and len(items) == 2071
    primary = [i for i in items if i.primary]
    assert len(primary) == 900 and len({i.history_id for i in primary}) == 9
    by_comp = Counter(i.category for i in items)
    assert by_comp == {
        "Accurate_Retrieval": 1000,
        "Conflict_Resolution": 200,
        "Long_Range_Understanding": 171,
        "Test_Time_Learning": 700,
    }
    sources = Counter(i.meta["source"] for i in items)
    contexts = Counter(h.split("-")[1] for h in histories)
    table6 = {  # source: (contexts, questions)
        "ruler_qa1_197K": (1, 100), "ruler_qa2_421K": (1, 100), "longmemeval_s*": (5, 300),
        "eventqa_full": (5, 500), "factconsolidation_sh_262k": (1, 100), "factconsolidation_mh_262k": (1, 100),
        "recsys_redial_full": (1, 200), "infbench_sum_eng_shots2": (100, 100), "detective_qa": (10, 71),
        **{f"icl_{t}": (1, 100) for t in ("banking77_5900shot_balance", "clinic150_7050shot_balance",
                                         "nlu_8296shot_balance", "trec_coarse_6600shot_balance",
                                         "trec_fine_6400shot_balance")},
    }
    assert dict(sources) == {s: q for s, (_, q) in table6.items()}
    assert dict(contexts) == {s.replace("*", ""): c for s, (c, _) in table6.items()}
    assert len({i.item_id for i in items}) == len(items)
    assert all(i.history_id in histories for i in items)
