"""Evaluate extraction-time memory placement independently from retrieval."""

import re
from dataclasses import dataclass
from pathlib import Path

from evals.scenarios.generate_cross_scope import candidates_by_message
from evals.schemas import (
    CrossScopeScenario,
    ScopeClassificationEvaluation,
    ScopeClassificationPair,
)
from scopegraph.llm.scope_classification import resolve_candidate_scope
from scopegraph.memory.promoter import PromotionPolicy
from scopegraph.models.memory import MemoryCandidate
from scopegraph.models.scope import ScopeRef, ScopeType


@dataclass(frozen=True)
class Placement:
    level: str
    scope_id: str | None
    session_id: str | None

    @property
    def target(self) -> tuple[str, str | None, str | None]:
        return self.level, self.scope_id, self.session_id


def _placement(
    candidate: MemoryCandidate,
    *,
    current_scope: ScopeRef,
    global_scope_id: str | None,
    policy: PromotionPolicy,
) -> Placement:
    decision = resolve_candidate_scope(
        candidate,
        current_scope=current_scope,
        global_scope_id=global_scope_id,
    )
    level = decision.scope_level
    if level == "scope" and not policy.should_promote_session_to_scope(candidate):
        level = "session"
    return Placement(
        level=level,
        scope_id=decision.scope_id,
        session_id=current_scope.session_id if level == "session" else None,
    )


def _tokens(candidate: MemoryCandidate) -> set[str]:
    text = " ".join(
        value or ""
        for value in (candidate.content, candidate.subject, candidate.predicate, candidate.object)
    )
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _match_score(gold: MemoryCandidate, predicted: MemoryCandidate) -> float:
    score = 0.0
    for attribute, weight in (("subject", 4.0), ("predicate", 4.0), ("object", 3.0)):
        left = getattr(gold, attribute)
        right = getattr(predicted, attribute)
        if left and right and left.casefold() == right.casefold():
            score += weight
    gold_tokens, predicted_tokens = _tokens(gold), _tokens(predicted)
    union = gold_tokens | predicted_tokens
    if union:
        score += len(gold_tokens & predicted_tokens) / len(union)
    return score


def _summary(
    pairs: list[ScopeClassificationPair],
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    gold_pairs = [pair for pair in pairs if pair.gold_candidate_index is not None]
    matched = [pair for pair in gold_pairs if pair.predicted_candidate_index is not None]
    extra = [pair for pair in pairs if pair.gold_candidate_index is None]
    total = len(gold_pairs)
    level_correct = sum(pair.scope_level_correct for pair in gold_pairs)
    target_correct = sum(pair.scope_target_correct for pair in gold_pairs)
    confusion: dict[str, dict[str, int]] = {}
    for pair in pairs:
        gold = pair.gold_scope_level or "__extra__"
        predicted = pair.predicted_scope_level or "__missing__"
        confusion.setdefault(gold, {})[predicted] = (
            confusion.setdefault(gold, {}).get(predicted, 0) + 1
        )
    return (
        {
            "gold_candidate_count": total,
            "matched_candidate_count": len(matched),
            "missing_candidate_count": total - len(matched),
            "extra_candidate_count": len(extra),
            "candidate_coverage": len(matched) / total if total else None,
            "scope_level_accuracy": level_correct / total if total else None,
            "scope_target_accuracy": target_correct / total if total else None,
            "scope_level_accuracy_matched": (
                sum(pair.scope_level_correct for pair in matched) / len(matched)
                if matched
                else None
            ),
            "scope_target_accuracy_matched": (
                sum(pair.scope_target_correct for pair in matched) / len(matched)
                if matched
                else None
            ),
        },
        confusion,
    )


def evaluate_scope_classification(
    scenarios: list[CrossScopeScenario],
    extracted: dict[str, dict[str, list[MemoryCandidate]]],
    *,
    live_extraction: bool,
    policy: PromotionPolicy | None = None,
) -> ScopeClassificationEvaluation:
    if not live_extraction:
        return ScopeClassificationEvaluation(
            evaluated=False,
            mode="not-evaluated/oracle-extraction",
            summary={
                "scope_level_accuracy": None,
                "scope_target_accuracy": None,
                "reason": "oracle candidates already contain the gold placement",
            },
        )
    effective_policy = policy or PromotionPolicy()
    pairs: list[ScopeClassificationPair] = []
    for scenario in scenarios:
        gold_by_message = candidates_by_message(scenario)
        predicted_by_message = extracted.get(scenario.scenario_id, {})
        scopes = {scope.id: scope for scope in scenario.scopes}
        global_scope = next(
            (scope for scope in scenario.scopes if scope.scope_type is ScopeType.GLOBAL), None
        )
        for session in scenario.sessions:
            scope = scopes[session.scope_id]
            current_scope = ScopeRef(
                id=scope.id,
                name=scope.name,
                scope_type=scope.scope_type,
                session_id=session.id,
            )
            for message in session.messages:
                gold = gold_by_message.get(message.id, [])
                predicted = predicted_by_message.get(message.id, [])
                remaining = set(range(len(predicted)))
                for gold_index, gold_candidate in enumerate(gold):
                    predicted_index = max(
                        remaining,
                        key=lambda index: (_match_score(gold_candidate, predicted[index]), -index),
                        default=None,
                    )
                    gold_placement = _placement(
                        gold_candidate,
                        current_scope=current_scope,
                        global_scope_id=global_scope.id if global_scope else None,
                        policy=effective_policy,
                    )
                    if predicted_index is None:
                        pairs.append(
                            ScopeClassificationPair(
                                scenario_id=scenario.scenario_id,
                                source_message_id=message.id,
                                gold_candidate_index=gold_index,
                                gold_scope_level=gold_placement.level,
                                gold_scope_id=gold_placement.scope_id,
                                gold_session_id=gold_placement.session_id,
                            )
                        )
                        continue
                    remaining.remove(predicted_index)
                    predicted_candidate = predicted[predicted_index]
                    predicted_placement = _placement(
                        predicted_candidate,
                        current_scope=current_scope,
                        global_scope_id=global_scope.id if global_scope else None,
                        policy=effective_policy,
                    )
                    pairs.append(
                        ScopeClassificationPair(
                            scenario_id=scenario.scenario_id,
                            source_message_id=message.id,
                            gold_candidate_index=gold_index,
                            predicted_candidate_index=predicted_index,
                            gold_scope_level=gold_placement.level,
                            predicted_scope_level=predicted_placement.level,
                            gold_scope_id=gold_placement.scope_id,
                            predicted_scope_id=predicted_placement.scope_id,
                            gold_session_id=gold_placement.session_id,
                            predicted_session_id=predicted_placement.session_id,
                            match_score=_match_score(gold_candidate, predicted_candidate),
                            scope_level_correct=gold_placement.level == predicted_placement.level,
                            scope_target_correct=gold_placement.target
                            == predicted_placement.target,
                        )
                    )
                for predicted_index in sorted(remaining):
                    predicted_placement = _placement(
                        predicted[predicted_index],
                        current_scope=current_scope,
                        global_scope_id=global_scope.id if global_scope else None,
                        policy=effective_policy,
                    )
                    pairs.append(
                        ScopeClassificationPair(
                            scenario_id=scenario.scenario_id,
                            source_message_id=message.id,
                            predicted_candidate_index=predicted_index,
                            predicted_scope_level=predicted_placement.level,
                            predicted_scope_id=predicted_placement.scope_id,
                            predicted_session_id=predicted_placement.session_id,
                        )
                    )
    summary, confusion = _summary(pairs)
    return ScopeClassificationEvaluation(
        evaluated=True,
        mode="live-extraction/effective-placement",
        pairs=pairs,
        summary=summary,
        confusion_matrix=confusion,
    )


def write_scope_classification_report(
    evaluation: ScopeClassificationEvaluation, output_root: Path
) -> None:
    processed = output_root / "processed" / "scope_classification.json"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text(evaluation.model_dump_json(indent=2) + "\n")
    lines = ["# Scope classification", ""]
    if not evaluation.evaluated:
        lines.extend(["Not evaluated: oracle extraction supplies the gold placement.", ""])
    else:
        lines.extend(
            [
                "| metric | value |",
                "|---|---:|",
                *[
                    f"| {name} | {value:.4f} |"
                    if isinstance(value, float)
                    else f"| {name} | {value} |"
                    for name, value in evaluation.summary.items()
                ],
                "",
                "## Scope-level confusion matrix",
                "",
                "Rows are gold levels and columns are predicted levels.",
                "",
            ]
        )
        predicted_levels = sorted(
            {predicted for row in evaluation.confusion_matrix.values() for predicted in row}
        )
        lines.append("| gold \\ predicted | " + " | ".join(predicted_levels) + " |")
        lines.append("|---|" + "---:|" * len(predicted_levels))
        for gold, row in sorted(evaluation.confusion_matrix.items()):
            lines.append(
                f"| {gold} | "
                + " | ".join(str(row.get(predicted, 0)) for predicted in predicted_levels)
                + " |"
            )
        lines.append("")
    table = output_root / "tables" / "scope_classification.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines))
