"""Long-term memory: Redis-backed with JSON file fallback for user preferences & facts."""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class LongTermMemory:
    """Persistent key-value store (Redis preferred, JSON file fallback)."""

    def __init__(self, user_id: str, redis_url: str = "redis://localhost:6379",
                 data_dir: str = "data"):
        self.user_id = user_id
        self._store_path = Path(data_dir) / "long_term_store.json"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        self._redis: Optional[Any] = None
        if _REDIS_AVAILABLE:
            try:
                r = _redis_lib.from_url(redis_url, decode_responses=True,
                                        socket_connect_timeout=1)
                r.ping()
                self._redis = r
                print("[OK] Redis connected")
            except Exception:
                print("[WARN] Redis unavailable - using JSON fallback")

        self._store: Dict[str, Any] = self._load_store()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_store(self) -> Dict:
        if self._store_path.exists():
            with open(self._store_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _persist(self) -> None:
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)

    def _key(self, key: str) -> str:
        return f"user:{self.user_id}:{key}"

    # ── public API ────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        full_key = self._key(key)
        entry: Dict[str, Any] = {"value": value, "timestamp": time.time()}
        if ttl:
            entry["expires_at"] = time.time() + ttl

        if self._redis:
            self._redis.set(full_key, json.dumps(entry, ensure_ascii=False))
            if ttl:
                self._redis.expire(full_key, ttl)
        else:
            self._store[full_key] = entry
            self._persist()

    def get(self, key: str) -> Optional[Any]:
        full_key = self._key(key)
        if self._redis:
            raw = self._redis.get(full_key)
            return json.loads(raw)["value"] if raw else None

        entry = self._store.get(full_key)
        if not entry:
            return None
        if "expires_at" in entry and time.time() > entry["expires_at"]:
            del self._store[full_key]
            self._persist()
            return None
        return entry["value"]

    def _all_from_redis(self, prefix: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for full_key in self._redis.keys(f"{prefix}*"):
            raw = self._redis.get(full_key)
            if raw:
                result[full_key[len(prefix):]] = json.loads(raw)["value"]
        return result

    def _all_from_store(self, prefix: str) -> Dict[str, Any]:
        now = time.time()
        result: Dict[str, Any] = {}
        for full_key, entry in self._store.items():
            if not full_key.startswith(prefix):
                continue
            expired = "expires_at" in entry and now > entry["expires_at"]
            if not expired:
                result[full_key[len(prefix):]] = entry["value"]
        return result

    def get_all_user_data(self) -> Dict[str, Any]:
        prefix = f"user:{self.user_id}:"
        if self._redis:
            return self._all_from_redis(prefix)
        return self._all_from_store(prefix)

    # ── convenience methods ───────────────────────────────────────────────────

    def save_preference(self, pref_key: str, pref_value: str) -> None:
        prefs = self.get("preferences") or {}
        prefs[pref_key] = pref_value
        self.set("preferences", prefs)

    def get_preferences(self) -> Dict[str, str]:
        return self.get("preferences") or {}

    def save_user_fact(self, fact_key: str, fact_value: str) -> None:
        facts = self.get("facts") or {}
        facts[fact_key] = fact_value
        self.set("facts", facts)

    def get_user_facts(self) -> Dict[str, str]:
        return self.get("facts") or {}
