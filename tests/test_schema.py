import pytest
from conftest import make_history, make_item
from pydantic import ValidationError

from memstudy.schema import Item, Usage, make_user_id, render_transcript


def test_user_id_is_bench_underscore_history():
    assert make_user_id("locomo", "conv-26") == "locomo_conv-26"
    assert make_history("conv-26", "locomo").user_id == "locomo_conv-26"


def test_user_ids_differ_across_histories():
    assert make_history("a").user_id != make_history("b").user_id
    assert make_history("a", "locomo").user_id != make_history("a", "longmemeval").user_id


def test_item_id_must_be_filename_safe():
    with pytest.raises(ValidationError):
        Item(item_id="a/b", bench="x", history_id="h", category="c", question="q", gold="g")
    assert make_item().item_id == "h1-0000"


def test_usage_rejects_inconsistent_counts():
    with pytest.raises(ValidationError):
        Usage(input_tokens=100, cached_tokens=80, cache_write_tokens=30)
    with pytest.raises(ValidationError):
        Usage(output_tokens=10, reasoning_tokens=11)


def test_usage_adds_all_fields():
    a = Usage(input_tokens=10, cached_tokens=4, cache_write_tokens=2, output_tokens=5, reasoning_tokens=1)
    total = a + a
    assert total.input_tokens == 20 and total.cached_tokens == 8
    assert total.cache_write_tokens == 4 and total.reasoning_tokens == 2


def test_transcript_is_deterministic_and_complete():
    h = make_history()
    text = render_transcript(h)
    assert text == render_transcript(h)
    assert "[Session 1 | 1:00 pm on 1 May, 2023]" in text
    assert "Ann: I adopted a dog named alpha." in text
    assert "Ann: alpha learned to swim today." in text
