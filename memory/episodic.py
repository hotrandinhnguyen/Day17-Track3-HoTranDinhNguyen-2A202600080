"""Episodic memory: JSON file log of past user–agent exchanges."""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Episode:
    session_id: str
    user_message: str
    assistant_response: str
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_id: str = ""

    def __post_init__(self) -> None:
        if not self.episode_id:
            self.episode_id = f"ep_{int(self.timestamp * 1000)}"


class EpisodicMemory:
    """Append-only JSON episodic log with keyword search."""

    def __init__(self, user_id: str, data_dir: str = "data"):
        self.user_id = user_id
        log_path = Path(data_dir) / f"episodic_{user_id}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path
        self._episodes: List[Episode] = self._load()

    def _load(self) -> List[Episode]:
        if self._log_path.exists():
            with open(self._log_path, encoding="utf-8") as f:
                return [Episode(**ep) for ep in json.load(f)]
        return []

    def _save(self) -> None:
        with open(self._log_path, "w", encoding="utf-8") as f:
            json.dump([asdict(ep) for ep in self._episodes], f,
                      indent=2, ensure_ascii=False)

    # ── write ─────────────────────────────────────────────────────────────────

    def add_episode(self, session_id: str, user_message: str,
                    assistant_response: str, tags: List[str] = None,
                    metadata: Dict[str, Any] = None) -> Episode:
        ep = Episode(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._episodes.append(ep)
        self._save()
        return ep

    # ── read ──────────────────────────────────────────────────────────────────

    def search_by_keywords(self, query: str, top_k: int = 3) -> List[Episode]:
        query_words = set(query.lower().split())
        scored: List[tuple] = []
        for ep in self._episodes:
            text = f"{ep.user_message} {ep.assistant_response}".lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [ep for _, ep in scored[:top_k]]

    def get_recent(self, n: int = 5) -> List[Episode]:
        return sorted(self._episodes, key=lambda ep: ep.timestamp, reverse=True)[:n]

    def get_by_session(self, session_id: str) -> List[Episode]:
        return [ep for ep in self._episodes if ep.session_id == session_id]

    # ── formatting ────────────────────────────────────────────────────────────

    def format_episodes(self, episodes: List[Episode]) -> str:
        if not episodes:
            return ""
        lines = ["[Episodic Memory — past interactions]"]
        for ep in episodes:
            q = ep.user_message[:80]
            a = ep.assistant_response[:100]
            lines.append(f"  [{ep.session_id}] Q: {q!r} → A: {a!r}")
        return "\n".join(lines)

    def get_summary(self) -> str:
        if not self._episodes:
            return "No past episodes."
        recent = self.get_recent(3)
        lines = [f"Total {len(self._episodes)} episodes. Recent:"]
        for ep in recent:
            lines.append(f"  - [{ep.session_id}] {ep.user_message[:50]}…")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._episodes)
