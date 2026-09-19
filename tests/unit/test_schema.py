from scopegraph.graph.schema import CONSTRAINTS, INDEXES


def test_schema_covers_every_phase_one_node_identifier() -> None:
    schema = "\n".join(CONSTRAINTS)
    labels = ("Scope", "Session", "SourceMessage", "Memory", "CorrectionEvent", "ConsolidationRun")
    for label in labels:
        assert f"(n:{label})" in schema


def test_indexes_support_scope_status_and_content_queries() -> None:
    schema = "\n".join(INDEXES)
    assert "memory_scope_idx" in schema
    assert "memory_status_idx" in schema
    assert "memory_content_fulltext" in schema
