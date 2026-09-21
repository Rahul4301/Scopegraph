from typing import Literal

from scopegraph.models.memory import MemoryCandidate
from scopegraph.models.scope import ScopeDecision, ScopeRef, ScopeType


def resolve_candidate_scope(
    candidate: MemoryCandidate,
    *,
    current_scope: ScopeRef | None,
    global_scope_id: str | None,
) -> ScopeDecision:
    if candidate.proposed_scope_level == "global" and candidate.explicit_global_signal:
        if global_scope_id is None:
            return ScopeDecision(
                scope_id=None,
                scope_level="unknown",
                confidence=0.0,
                reason="explicit global memory requested but no global root exists",
            )
        return ScopeDecision(
            scope_id=global_scope_id,
            scope_level="global",
            confidence=candidate.confidence,
            reason="user explicitly indicated cross-context validity",
        )
    if current_scope is not None:
        level: Literal["session", "scope", "global"] = (
            "session" if candidate.proposed_scope_level == "session" else "scope"
        )
        if level == "scope" and current_scope.scope_type is ScopeType.GLOBAL:
            level = "global"
        return ScopeDecision(
            scope_id=current_scope.id,
            scope_level=level,
            confidence=max(candidate.confidence, 0.9),
            reason="explicit current scope supplied by caller",
        )
    if candidate.proposed_scope_id is not None:
        return ScopeDecision(
            scope_id=candidate.proposed_scope_id,
            scope_level="scope",
            confidence=candidate.confidence,
            reason="extractor identified a directly mentioned scope",
        )
    return ScopeDecision(
        scope_id=None,
        scope_level="unknown",
        confidence=0.0,
        reason="no explicit or classified scope is available",
    )
