"""Memory router: detect query intent and choose the right memory backends."""
from enum import Enum
from typing import List, Tuple

# ── keyword lists ──────────────────────────────────────────────────────────────

_PREFERENCE_KW = [
    "thích", "không thích", "prefer", "like", "dislike", "want", "don't want",
    "favorite", "yêu thích", "ghét", "muốn", "không muốn", "ưa", "sở thích",
]

_EPISODIC_KW = [
    "nhớ", "remember", "recall", "lần trước", "trước đó", "trước đây",
    "hôm qua", "tuần trước", "last time", "previously", "before", "mentioned",
    "đã nói", "đã hỏi", "từng", "bị confused", "episode",
]

_SEMANTIC_KW = [
    "tương tự", "similar", "giống", "liên quan", "related",
    "tìm kiếm", "search", "find", "lookup", "về chủ đề",
]

_FACTUAL_KW = [
    "là gì", "what is", "define", "explain", "how to", "why", "when",
    "nghĩa là", "giải thích", "cách", "làm thế nào", "tại sao",
]


# ── intent enum ────────────────────────────────────────────────────────────────

class MemoryIntent(Enum):
    USER_PREFERENCE = "user_preference"
    FACTUAL_RECALL = "factual_recall"
    EXPERIENCE_RECALL = "experience_recall"
    SEMANTIC_SEARCH = "semantic_search"
    GENERAL = "general"


# ── router ─────────────────────────────────────────────────────────────────────

class MemoryRouter:
    """Routes a query to the most relevant memory backends."""

    def detect_intent(self, query: str) -> Tuple[MemoryIntent, float]:
        q = query.lower()
        scores = {
            MemoryIntent.USER_PREFERENCE: sum(kw in q for kw in _PREFERENCE_KW),
            MemoryIntent.EXPERIENCE_RECALL: sum(kw in q for kw in _EPISODIC_KW),
            MemoryIntent.SEMANTIC_SEARCH: sum(kw in q for kw in _SEMANTIC_KW),
            MemoryIntent.FACTUAL_RECALL: sum(kw in q for kw in _FACTUAL_KW),
            MemoryIntent.GENERAL: 0,
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return MemoryIntent.GENERAL, 0.5
        return best, min(scores[best] / 3.0, 1.0)

    def get_backends(self, intent: MemoryIntent) -> List[str]:
        """Return ordered backend names for the given intent."""
        mapping = {
            MemoryIntent.USER_PREFERENCE:  ["long_term", "short_term"],
            MemoryIntent.FACTUAL_RECALL:   ["long_term", "semantic", "short_term"],
            MemoryIntent.EXPERIENCE_RECALL: ["episodic", "short_term"],
            MemoryIntent.SEMANTIC_SEARCH:  ["semantic", "episodic"],
            MemoryIntent.GENERAL:          ["short_term", "long_term", "episodic"],
        }
        return mapping.get(intent, ["short_term"])

    def route(self, query: str) -> Tuple[MemoryIntent, List[str]]:
        intent, _ = self.detect_intent(query)
        return intent, self.get_backends(intent)
