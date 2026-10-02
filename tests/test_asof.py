import pytest
from conftest import make_item

from memstudy.asof import expand_asof, last_evidence_session
from memstudy.schema import History, Session, Turn


def conv(n=4) -> History:
    return History(
        history_id="c1", bench="locomo",
        sessions=[
            Session(session_id=str(i), timestamp=f"day {i}", turns=[Turn(speaker="A", text=f"fact {i}")])
            for i in range(1, n + 1)
        ],
    )


def question(n: int, evidence: list[str]):
    return make_item("c1", n).model_copy(update={"meta": {"evidence": evidence}})


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (["D1:3"], 1),
        (["D2:1", "D3:4"], 3),
        (["D8:6; D9:17"], 4),  # D8 and D9 do not exist in a four-session conversation
        (["D1:3 D4:4 D2:6"], 4),
        (["D"], 4),
        ([], 4),
    ],
)
def test_last_evidence_session_is_the_latest_cited_session_that_exists(evidence, expected):
    assert last_evidence_session(evidence, [1, 2, 3, 4]) == expected


def test_each_question_is_asked_at_its_checkpoint_and_at_the_end():
    qs = [question(0, ["D1:1"]), question(1, ["D3:2"]), question(2, ["D4:1"])]
    histories, items = expand_asof({"c1": conv()}, qs)
    assert [i.item_id for i in items] == [
        "c1-0000.asof", "c1-0001.asof", "c1-0000.final", "c1-0001.final", "c1-0002.final",
    ]
    assert [i.history_id for i in items] == ["c1@s01", "c1@s03", "c1", "c1", "c1"]
    assert set(histories) == {"c1@s01", "c1@s03", "c1"}


def test_a_checkpoint_history_is_a_prefix_that_shares_the_conversations_store():
    histories, _ = expand_asof({"c1": conv()}, [question(0, ["D2:1"])])
    cut = histories["c1@s02"]
    assert [s.session_id for s in cut.sessions] == ["1", "2"]
    assert cut.store_id == "c1" and cut.store_key == histories["c1"].store_key == "locomo_c1"
    assert cut.user_id != histories["c1"].user_id  # distinct history, same store


def test_questions_are_ordered_by_checkpoint_so_every_store_only_moves_forward():
    qs = [question(0, ["D3:1"]), question(1, ["D1:1"]), question(2, ["D2:1"])]
    _, items = expand_asof({"c1": conv()}, qs)
    asof = [i.meta["asof_session"] for i in items if i.meta["checkpoint"] == "asof"]
    assert asof == sorted(asof) == [1, 2, 3]
    assert all(i.meta["checkpoint"] == "final" for i in items[3:])


def test_item_ids_stay_filename_safe_and_unique():
    _, items = expand_asof({"c1": conv()}, [question(i, ["D1:1"]) for i in range(5)])
    assert len({i.item_id for i in items}) == len(items) == 10


def test_arm_a_transcript_at_a_checkpoint_is_a_prefix_of_the_next():
    from memstudy.schema import render_transcript

    histories, _ = expand_asof({"c1": conv()}, [question(0, ["D1:1"]), question(1, ["D3:1"])])
    a, b, full = (render_transcript(histories[k]) for k in ("c1@s01", "c1@s03", "c1"))
    assert b.startswith(a) and full.startswith(b)
