from evals.metrics.official import locomo_score, memoryagentbench_score


def test_locomo_official_category_rules() -> None:
    assert locomo_score("Paris", "Paris; France", "3") == 1.0
    assert locomo_score("There is no information available.", "irrelevant", "5") == 1.0
    assert locomo_score("Neo4j, Python", "Neo4j, Python", "1") == 1.0


def test_memoryagentbench_deterministic_rules() -> None:
    assert memoryagentbench_score("Answer: 28", ["28"], "icl_banking77") == (
        "exact_match",
        1.0,
    )
    assert memoryagentbench_score("It was Paris.", ["Paris"], "factconsolidation_sh") == (
        "substring_exact_match",
        1.0,
    )
    event_score = memoryagentbench_score(
        "event one and event two", ["event one", "event two"], "eventqa"
    )
    assert event_score == (
        "substring_exact_match",
        1.0,
    )
    assert memoryagentbench_score("anything", ["x"], "longmemeval_s") is None
