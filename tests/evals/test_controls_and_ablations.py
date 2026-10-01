"""The vector_scope_filter control and the documented ablations must run and bite."""

from pathlib import Path

import pytest

from evals.analysis.aggregate import aggregate_records, load_jsonl, paired_ablation_comparisons
from evals.runners.run_all import ABLATIONS
from evals.runners.run_eval import run_evaluation


async def _run(tmp_path: Path, ablation: str, **kwargs):
    path = await run_evaluation(
        system_name="scopegraph", seed=42, difficulty=kwargs.pop("difficulty", 3),
        scenario_count=kwargs.pop("scenario_count", 3), ablation=ablation,
        output=str(tmp_path / f"{ablation}.jsonl"), storage="memory", **kwargs,
    )
    return load_jsonl([path])


def test_batch_lists_every_documented_condition():
    assert set(ABLATIONS) >= {
        "full", "vector_only_control", "vector_scope_filter", "flat_graph_control",
        "two_level_control", "no_graph_traversal", "no_temporal_status",
    }


@pytest.mark.asyncio
async def test_vector_scope_filter_searches_only_allowed_scopes(tmp_path):
    filtered = await _run(tmp_path, "vector_scope_filter")
    unfiltered = await _run(tmp_path, "vector_only_control")
    for row in filtered:
        allowed = set(row.allowed_scope_ids)
        assert all(set(origin) <= allowed for origin in row.retrieved_origin_scope_ids)
        assert set(row.retrieved_scope_ids) <= allowed
    assert any(
        set(origin) - set(row.allowed_scope_ids)
        for row in unfiltered for origin in row.retrieved_origin_scope_ids
    ), "vector_only_control must still expose competing scopes, or the pair isolates nothing"
    # Same ranker as vector_only_control: the only difference is the filter.
    assert {row.question_id for row in filtered} == {row.question_id for row in unfiltered}


@pytest.mark.asyncio
async def test_no_graph_and_no_temporal_ablations_change_results(tmp_path):
    full = await _run(tmp_path, "full")
    no_graph = await _run(tmp_path, "no_graph_traversal")
    no_temporal = await _run(tmp_path, "no_temporal_status")
    assert any(row.trace for row in full)
    graph_steps = lambda rows: sum(  # noqa: E731
        step["relation"].startswith(("RELATES_TO", "SUPPORTS", "SUPERSEDES"))
        for row in rows for step in row.trace
    )
    assert graph_steps(no_graph) < graph_steps(full)
    stale = lambda rows: sum(s != "active" for row in rows for s in row.retrieved_statuses)  # noqa: E731
    assert stale(no_temporal) > stale(full)
    summary = aggregate_records([*full, *no_graph, *no_temporal])
    assert {"scopegraph", "scopegraph/no_graph_traversal",
            "scopegraph/no_temporal_status"} <= set(summary)
    paired = paired_ablation_comparisons([*full, *no_graph, *no_temporal])
    assert "cross_scope_mem/scopegraph/full-minus-no_graph_traversal" in paired
    assert "cross_scope_mem/scopegraph/full-minus-no_temporal_status" in paired
