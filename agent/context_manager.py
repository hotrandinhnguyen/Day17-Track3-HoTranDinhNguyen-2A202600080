"""Context window manager: assembles context sections with priority-based eviction."""
from typing import Dict, Tuple

# Priority: higher = kept longer when evicting
_PRIORITY = {
    "system":     4,
    "short_term": 3,
    "long_term":  3,
    "episodic":   2,
    "semantic":   1,
}

_SECTION_ORDER = ["system", "long_term", "episodic", "semantic", "short_term"]


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class ContextManager:
    """
    Stores named sections (system, long_term, episodic, semantic, short_term)
    and produces a single context string that fits within max_tokens.
    Evicts lowest-priority sections first when over budget.
    """

    def __init__(self, max_tokens: int = 6000, reserve_tokens: int = 800):
        self.budget = max(max_tokens - reserve_tokens, 500)
        # name → (content, priority)
        self._sections: Dict[str, Tuple[str, int]] = {}

    def set_section(self, name: str, content: str, priority: int = None) -> None:
        if not content:
            return
        p = priority if priority is not None else _PRIORITY.get(name, 1)
        self._sections[name] = (content, p)

    def clear(self) -> None:
        self._sections.clear()

    def get_token_counts(self) -> Dict[str, int]:
        return {name: _estimate_tokens(content)
                for name, (content, _) in self._sections.items()}

    def build_context(self) -> Tuple[str, Dict[str, int]]:
        """
        Returns (assembled_context_string, token_stats_dict).
        Evicts low-priority sections until the budget is met.
        """
        stats = self.get_token_counts()
        total = sum(stats.values())

        remaining = dict(self._sections)

        if total > self.budget:
            # evict by ascending priority
            sorted_by_priority = sorted(
                remaining.items(), key=lambda kv: kv[1][1]
            )
            for name, (content, _) in sorted_by_priority:
                if total <= self.budget:
                    break
                # never evict system or short_term entirely
                if _PRIORITY.get(name, 1) < _PRIORITY["short_term"]:
                    total -= stats[name]
                    del remaining[name]
                    stats[name] = 0

            # if still over budget, trim short_term from top
            if total > self.budget and "short_term" in remaining:
                content, prio = remaining["short_term"]
                lines = content.split("\n")
                while lines and total > self.budget:
                    removed = lines.pop(0)
                    total -= _estimate_tokens(removed)
                remaining["short_term"] = ("\n".join(lines), prio)
                stats["short_term"] = _estimate_tokens(remaining["short_term"][0])

        parts = []
        for name in _SECTION_ORDER:
            if name in remaining:
                text, _ = remaining[name]
                if text.strip():
                    parts.append(text.strip())

        return "\n\n".join(parts), stats

    def utilization(self) -> float:
        used = sum(self.get_token_counts().values())
        return min(used / max(self.budget, 1), 1.0)
