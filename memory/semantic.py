"""Semantic memory: ChromaDB vector store for similarity-based retrieval."""
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    print("[WARN] chromadb not installed — semantic memory disabled")


class SemanticMemory:
    """Persistent Chroma collection per user; falls back silently if unavailable."""

    def __init__(self, user_id: str, data_dir: str = "data"):
        self.user_id = user_id
        self._enabled = _CHROMA_AVAILABLE

        if self._enabled:
            db_path = Path(data_dir) / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
            try:
                self._client = chromadb.PersistentClient(path=str(db_path))
                self._col = self._client.get_or_create_collection(
                    name=f"user_{user_id}",
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"[OK] Chroma ready (user={user_id}, docs={self._col.count()})")
            except Exception as exc:
                print(f"[WARN] Chroma init failed: {exc}")
                self._enabled = False

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, text: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        if not self._enabled:
            return None
        doc_id = str(uuid.uuid4())
        meta = {"timestamp": time.time(), "user_id": self.user_id}
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        try:
            self._col.add(documents=[text], metadatas=[meta], ids=[doc_id])
            return doc_id
        except Exception as exc:
            print(f"[WARN] Chroma add error: {exc}")
            return None

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._enabled:
            return []
        try:
            count = self._col.count()
            if count == 0:
                return []
            results = self._col.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
            items = []
            for i, doc in enumerate(results["documents"][0]):
                items.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                    if "distances" in results else 0.0,
                })
            return items
        except Exception as exc:
            print(f"[WARN] Chroma search error: {exc}")
            return []

    # ── formatting ────────────────────────────────────────────────────────────

    def format_results(self, results: List[Dict]) -> str:
        if not results:
            return ""
        lines = ["[Semantic Memory — similar past exchanges]"]
        for r in results:
            lines.append(f"  - {r['text'][:150]}")
        return "\n".join(lines)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def count(self) -> int:
        if not self._enabled:
            return 0
        return self._col.count()
