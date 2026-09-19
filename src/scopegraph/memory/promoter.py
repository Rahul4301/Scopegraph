from dataclasses import dataclass
from typing import Any

from scopegraph.models.memory import Memory, MemoryCandidate


@dataclass(frozen=True)
class PromotionPolicy:
    session_to_scope_durability_threshold: float = 0.55
    minimum_distinct_scopes: int = 3
    global_confidence_threshold: float = 0.8
    require_explicit_global_signal: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PromotionPolicy":
        promotion = config.get("promotion", {})
        session_policy = promotion.get("session_to_scope", {})
        global_policy = promotion.get("scope_to_global", {})
        return cls(
            session_to_scope_durability_threshold=float(
                session_policy.get("durability_threshold", 0.55)
            ),
            minimum_distinct_scopes=int(global_policy.get("minimum_distinct_scopes", 3)),
            global_confidence_threshold=float(
                global_policy.get("confidence_threshold", 0.8)
            ),
            require_explicit_global_signal=bool(
                global_policy.get("require_explicit_global_signal", False)
            ),
        )

    def should_promote_session_to_scope(self, candidate: MemoryCandidate) -> bool:
        return (
            candidate.proposed_scope_level != "session"
            and candidate.durability >= self.session_to_scope_durability_threshold
        )

    def should_promote_scope_to_global(
        self, candidate: MemoryCandidate, equivalent_scope_memories: list[Memory]
    ) -> bool:
        if candidate.explicit_global_signal:
            return candidate.confidence >= self.global_confidence_threshold
        if self.require_explicit_global_signal:
            return False
        scopes = {memory.scope_id for memory in equivalent_scope_memories}
        return (
            candidate.confidence >= self.global_confidence_threshold
            and len(scopes) >= self.minimum_distinct_scopes
        )
