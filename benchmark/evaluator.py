"""Benchmark evaluator: measures response quality metrics per turn / conversation."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TurnResult:
    turn_id: int
    query: str
    response: str
    intent: str
    memory_context_len: int
    response_time: float
    has_memory: bool

    # computed after construction
    memory_hit: bool = False
    keyword_relevance: float = 0.0

    @property
    def response_word_count(self) -> int:
        return len(self.response.split())


@dataclass
class ConversationResult:
    conv_id: str
    conv_name: str
    has_memory: bool
    turns: List[TurnResult] = field(default_factory=list)

    @property
    def avg_response_time(self) -> float:
        return (sum(t.response_time for t in self.turns) / len(self.turns)
                if self.turns else 0.0)

    @property
    def memory_hit_rate(self) -> float:
        return (sum(t.memory_hit for t in self.turns) / len(self.turns)
                if self.turns else 0.0)

    @property
    def avg_context_len(self) -> float:
        return (sum(t.memory_context_len for t in self.turns) / len(self.turns)
                if self.turns else 0.0)

    @property
    def avg_keyword_relevance(self) -> float:
        return (sum(t.keyword_relevance for t in self.turns) / len(self.turns)
                if self.turns else 0.0)


class BenchmarkEvaluator:

    # ── per-turn metrics ───────────────────────────────────────────────────────

    def evaluate_memory_hit(self, turn: TurnResult, expected: str) -> bool:
        """Memory hit if context was retrieved when expected."""
        if expected == "none":
            return True                 # no memory needed — always pass
        if not turn.has_memory:
            return False
        return turn.memory_context_len > 0

    def evaluate_keyword_relevance(
        self, query: str, response: str, history: List[str]
    ) -> float:
        """Fraction of query+history keywords that appear in the response."""
        def words(text: str):
            return {w.lower() for w in text.split() if len(w) > 3}

        important = words(query)
        for h in history[-4:]:         # last 4 history items
            important |= words(h)

        if not important:
            return 1.0
        response_words = words(response)
        return len(important & response_words) / len(important)

    # ── aggregate metrics ──────────────────────────────────────────────────────

    def compute_metrics(
        self,
        with_memory: List[ConversationResult],
        without_memory: List[ConversationResult],
    ) -> Dict[str, Any]:
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "memory_hit_rate": {
                "with_memory": avg([r.memory_hit_rate for r in with_memory]),
                "without_memory": avg([r.memory_hit_rate for r in without_memory]),
            },
            "avg_response_time": {
                "with_memory": avg([r.avg_response_time for r in with_memory]),
                "without_memory": avg([r.avg_response_time for r in without_memory]),
            },
            "avg_context_size": {
                "with_memory": avg([r.avg_context_len for r in with_memory]),
                "without_memory": avg([r.avg_context_len for r in without_memory]),
            },
            "avg_keyword_relevance": {
                "with_memory": avg([r.avg_keyword_relevance for r in with_memory]),
                "without_memory": avg([r.avg_keyword_relevance for r in without_memory]),
            },
        }
