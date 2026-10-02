import pytest

from memstudy.chunking import chunk_document
from memstudy.prompts import judge_prompt
from memstudy.schema import Item
from memstudy.scoring import answer_metrics, exact_match, normalize, substring_match, token_f1


def test_normalize_drops_case_punctuation_and_articles():
    assert normalize("The  Dog, named Rex!") == "dog named rex"


def test_normalize_treats_curly_quotes_like_straight_ones():
    assert normalize("\u201cBecoming Nicole\u201d by Amy") == normalize('"Becoming Nicole" by Amy')
    assert token_f1("\u201cBecoming Nicole\u201d", ['"Becoming Nicole"']) == 1.0


def test_best_of_several_accepted_answers():
    golds = ["10th and 11th centuries", "in the 10th and 11th centuries"]
    assert exact_match("In the 10th and 11th centuries.", golds)
    assert substring_match("It was in the 10th and 11th centuries, roughly", golds)
    assert not exact_match("the 12th century", golds)
    assert token_f1("10th century", ["10th and 11th centuries"]) == pytest.approx(1 / 3)
    assert token_f1("the dog named Rex", ["dog Rex"]) == pytest.approx(0.8)


def test_answer_metrics_report_whether_the_gold_reached_the_reader():
    seen = answer_metrics("France", "Normandy is in France.", ["France"])
    assert seen["exact_match"] and seen["gold_in_context"] and seen["token_f1"] == 1.0
    missed = answer_metrics("Not mentioned", "Nothing relevant here.", ["France"])
    assert not missed["substring_match"] and not missed["gold_in_context"]


def _item(**meta):
    return Item(item_id="i", bench="b", history_id="h", category="c", question="q", gold="France", meta=meta)


def test_judge_prompt_is_unchanged_for_a_single_gold_and_lists_all_accepted_answers():
    assert "Gold answer: France\n" in judge_prompt(_item(), "x")
    many = judge_prompt(_item(answers=["France", "France", "FR"]), "x")
    assert "Gold answers (matching any one is correct): France | FR" in many


def test_chunk_document_is_lossless_and_respects_the_limit():
    text = "\n".join(f"line {i} " + "word " * (i % 7) for i in range(300)) + "\n" + "tok " * 500
    chunks = chunk_document(text, 50)
    assert max(len(c.split()) for c in chunks) <= 60
    assert "".join("".join(c.split()) for c in chunks) == "".join(text.split())
    assert chunk_document("one line", 50) == ["one line"]
