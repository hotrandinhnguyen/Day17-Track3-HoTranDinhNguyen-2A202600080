"""Memory router: detect query intent and choose the right memory backends."""
import re
from enum import Enum
from typing import List, Tuple

# ── keyword lists ─────────────────────────────────────────────────────────────
# Use word-boundary patterns to avoid substring false positives
# e.g. "thích" must not match inside "giải thích"

_PREFERENCE_PATTERNS = [
    r"\btôi thích\b", r"\byêu thích\b", r"\bkhông thích\b",
    r"\b(toi|tôi|ban|bạn)\s+thich\b",      # "tôi thich X" — NOT "giai thich"
    r"\bghét\b", r"\bsở thích\b",
    r"\bprefer\b", r"\blike\b", r"\bdislike\b",
    r"\bmuốn\b", r"\bkhông muốn\b",
    r"\bdị ứng\b", r"\bdi ung\b",
    r"\btên (tôi|là)\b", r"\btoi ten\b",
]

_EPISODIC_PATTERNS = [
    r"\bnhớ\b", r"\bremember\b", r"\brecall\b",
    r"\blần trước\b", r"\btrước đó\b", r"\btrước đây\b",
    r"\btruoc day\b", r"\btruoc do\b", r"\blan truoc\b",
    r"\bhôm qua\b", r"\btuần trước\b", r"\blast time\b",
    r"\bpreviously\b", r"\bmentioned\b",
    r"\bđã (nói|hỏi|học)\b", r"\bda (noi|hoi|hoc)\b",
    r"\btừng\b", r"\btung\b",
    r"\bbị confused\b", r"\bbi confused\b",
]

_SEMANTIC_PATTERNS = [
    r"\btương tự\b", r"\btuong tu\b", r"\bsimilar\b", r"\bgiống\b",
    r"\bliên quan\b", r"\blien quan\b", r"\brelated\b",
    r"\btìm (kiếm|cho tôi)\b", r"\btim\s+(kiem|noi dung|cho)\b",
    r"\bsearch\b", r"\bfind\b",
    r"\bvề chủ đề\b", r"\bnội dung tương tự\b", r"\bnoi dung tuong tu\b",
]

_FACTUAL_PATTERNS = [
    r"\blà gì\b", r"\bwhat is\b", r"\bdefine\b",
    r"\bexplain\b", r"\bhow to\b", r"\bwhy\b",
    r"\bnghĩa là\b", r"\bgiải thích\b", r"\bgiai thich\b",
    r"\bcách\b", r"\blàm thế nào\b", r"\btại sao\b",
    r"\bso sánh\b", r"\bkhác nhau\b", r"\bhoạt động\b",
]


def _score(patterns: List[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


# ── intent enum ───────────────────────────────────────────────────────────────

class MemoryIntent(Enum):
    USER_PREFERENCE  = "user_preference"
    FACTUAL_RECALL   = "factual_recall"
    EXPERIENCE_RECALL = "experience_recall"
    SEMANTIC_SEARCH  = "semantic_search"
    GENERAL          = "general"


# ── router ────────────────────────────────────────────────────────────────────

class MemoryRouter:
    """Routes a query to the most relevant memory backends using regex patterns."""

    def detect_intent(self, query: str) -> Tuple[MemoryIntent, float]:
        q = query.lower()
        scores = {
            MemoryIntent.USER_PREFERENCE:   _score(_PREFERENCE_PATTERNS, q),
            MemoryIntent.EXPERIENCE_RECALL: _score(_EPISODIC_PATTERNS,   q),
            MemoryIntent.SEMANTIC_SEARCH:   _score(_SEMANTIC_PATTERNS,   q),
            MemoryIntent.FACTUAL_RECALL:    _score(_FACTUAL_PATTERNS,    q),
            MemoryIntent.GENERAL: 0,
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return MemoryIntent.GENERAL, 0.5
        return best, min(scores[best] / 3.0, 1.0)

    def get_backends(self, intent: MemoryIntent) -> List[str]:
        mapping = {
            MemoryIntent.USER_PREFERENCE:   ["long_term", "short_term"],
            MemoryIntent.FACTUAL_RECALL:    ["long_term", "semantic", "short_term"],
            MemoryIntent.EXPERIENCE_RECALL: ["episodic", "short_term"],
            MemoryIntent.SEMANTIC_SEARCH:   ["semantic", "episodic"],
            MemoryIntent.GENERAL:           ["short_term", "long_term", "episodic"],
        }
        return mapping.get(intent, ["short_term"])

    def route(self, query: str) -> Tuple[MemoryIntent, List[str]]:
        intent, _ = self.detect_intent(query)
        return intent, self.get_backends(intent)
