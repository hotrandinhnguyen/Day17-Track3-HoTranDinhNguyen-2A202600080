"""Short-term memory: in-memory conversation buffer for current session."""
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class ShortTermMemory:
    """ConversationBufferMemory — keeps last N turns in RAM."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._buffer: List[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self._buffer.append(Message(role=role, content=content))
        # keep only the last max_turns user+assistant pairs
        cap = self.max_turns * 2
        if len(self._buffer) > cap:
            self._buffer = self._buffer[-cap:]

    def get_history(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._buffer]

    def get_formatted(self) -> str:
        if not self._buffer:
            return ""
        lines = []
        for m in self._buffer:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
