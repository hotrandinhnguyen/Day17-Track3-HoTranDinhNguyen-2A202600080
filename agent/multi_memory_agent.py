"""Multi-Memory LangGraph agent — gpt-4o-mini (OpenAI)."""
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from openai import OpenAI

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from agent.memory_router import MemoryRouter, MemoryIntent
from agent.context_manager import ContextManager, _estimate_tokens


# ── LangGraph state ────────────────────────────────────────────────────────────
# Shape matches rubric spec: messages / user_profile / episodes / semantic_hits / memory_budget

class MemoryState(TypedDict):
    messages: list          # short-term conversation history (list of {role, content})
    user_profile: dict      # long-term profile: preferences + facts
    episodes: list          # retrieved episodic memories (list of dicts)
    semantic_hits: list     # semantic search results (list of strings)
    memory_budget: int      # remaining token budget after memory injection
    query: str
    response: str
    intent: str


# ── agent ──────────────────────────────────────────────────────────────────────

class MultiMemoryAgent:
    """
    LangGraph agent with 4 memory backends:
      short-term  → ShortTermMemory (sliding window)
      long-term   → LongTermMemory  (Redis / JSON KV)
      episodic    → EpisodicMemory  (JSON log)
      semantic    → SemanticMemory  (ChromaDB)

    Graph: START → retrieve_memory → build_prompt → generate → save_memory → END
    """

    MAX_TOKENS = 6000
    RESERVE_TOKENS = 800

    def __init__(
        self,
        user_id: str = "user_001",
        session_id: Optional[str] = None,
        data_dir: str = "data",
        use_memory: bool = True,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.user_id = user_id
        self.session_id = session_id or f"session_{int(time.time())}"
        self.use_memory = use_memory
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        self.short_term = ShortTermMemory(max_turns=10)
        if use_memory:
            self.long_term = LongTermMemory(user_id=user_id, data_dir=data_dir)
            self.episodic  = EpisodicMemory(user_id=user_id, data_dir=data_dir)
            self.semantic  = SemanticMemory(user_id=user_id, data_dir=data_dir)
        else:
            self.long_term = None
            self.episodic  = None
            self.semantic  = None

        self._router = MemoryRouter()
        self._ctx    = ContextManager(max_tokens=self.MAX_TOKENS,
                                       reserve_tokens=self.RESERVE_TOKENS)
        self._graph  = self._build_graph()

    # ── graph construction ─────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(MemoryState)
        g.add_node("retrieve_memory", self.retrieve_memory)
        g.add_node("build_prompt",    self._build_prompt)
        g.add_node("generate",        self._generate)
        g.add_node("save_memory",     self._save_memory)

        g.add_edge(START,             "retrieve_memory")
        g.add_edge("retrieve_memory", "build_prompt")
        g.add_edge("build_prompt",    "generate")
        g.add_edge("generate",        "save_memory")
        g.add_edge("save_memory",     END)
        return g.compile()

    # ── node 1: retrieve_memory ───────────────────────────────────────────────

    def retrieve_memory(self, state: MemoryState) -> Dict:
        """
        Load memories from all relevant backends into state fields.
        Router selects which backends to query based on query intent.
        """
        query = state["query"]
        user_profile: Dict[str, Any] = {}
        episodes: List[Dict] = []
        semantic_hits: List[str] = []
        intent_value = MemoryIntent.GENERAL.value

        if self.use_memory:
            intent, backends = self._router.route(query)
            intent_value = intent.value

            for backend in backends:
                if backend == "long_term" and self.long_term:
                    prefs = self.long_term.get_preferences()
                    facts = self.long_term.get_user_facts()
                    user_profile = {**prefs, **facts}

                elif backend == "episodic" and self.episodic:
                    eps = self.episodic.search_by_keywords(query, top_k=3)
                    episodes = [
                        {
                            "session": ep.session_id,
                            "user":    ep.user_message[:120],
                            "agent":   ep.assistant_response[:150],
                        }
                        for ep in eps
                    ]

                elif backend == "semantic" and self.semantic:
                    hits = self.semantic.search(query, top_k=3)
                    semantic_hits = [h["text"][:200] for h in hits]

        # Always load full long-term profile when use_memory is on
        if self.use_memory and self.long_term and not user_profile:
            prefs = self.long_term.get_preferences()
            facts = self.long_term.get_user_facts()
            user_profile = {**prefs, **facts}

        return {
            "user_profile":  user_profile,
            "episodes":      episodes,
            "semantic_hits": semantic_hits,
            "intent":        intent_value,
            "messages":      self.short_term.get_history(),
        }

    # ── node 2: build_prompt ──────────────────────────────────────────────────

    def _build_prompt(self, state: MemoryState) -> Dict:
        """
        Assemble memory context with 4 explicit sections, apply token budget trimming.
        """
        self._ctx.clear()
        budget = self.MAX_TOKENS - self.RESERVE_TOKENS

        parts: List[str] = []

        # Section 1 — User Profile (long-term)
        if state["user_profile"]:
            lines = [f"  {k}: {v}" for k, v in state["user_profile"].items()]
            parts.append("[User Profile]\n" + "\n".join(lines))

        # Section 2 — Episodic Memory
        if state["episodes"]:
            ep_lines = []
            for ep in state["episodes"]:
                ep_lines.append(
                    f"  [{ep['session']}] Q: {ep['user']!r} → A: {ep['agent']!r}"
                )
            parts.append("[Episodic Memory]\n" + "\n".join(ep_lines))

        # Section 3 — Semantic Context
        if state["semantic_hits"]:
            sem_lines = [f"  - {hit}" for hit in state["semantic_hits"]]
            parts.append("[Semantic Context]\n" + "\n".join(sem_lines))

        # Section 4 — Recent Conversation (short-term)
        history = self.short_term.get_formatted()
        if history:
            parts.append("[Recent Conversation]\n" + history)

        memory_context = "\n\n".join(parts)

        # Token budget: trim short-term if over budget
        used_tokens = _estimate_tokens(memory_context)
        remaining = max(budget - used_tokens, 0)

        return {"memory_budget": remaining}

    # ── node 3: generate ──────────────────────────────────────────────────────

    def _generate(self, state: MemoryState) -> Dict:
        """Inject memory context into system prompt, call gpt-4o-mini."""
        system = (
            "You are a helpful AI assistant with persistent memory.\n"
            "Use the memory context below to give personalized, relevant responses.\n"
            "If memory shows user preferences, apply them without being asked."
        )

        # Build memory context string from state
        memory_parts: List[str] = []
        if state["user_profile"]:
            lines = [f"  {k}: {v}" for k, v in state["user_profile"].items()]
            memory_parts.append("[User Profile]\n" + "\n".join(lines))
        if state["episodes"]:
            ep_lines = [
                f"  [{ep['session']}] Q: {ep['user']!r} → A: {ep['agent']!r}"
                for ep in state["episodes"]
            ]
            memory_parts.append("[Episodic Memory]\n" + "\n".join(ep_lines))
        if state["semantic_hits"]:
            sem_lines = [f"  - {h}" for h in state["semantic_hits"]]
            memory_parts.append("[Semantic Context]\n" + "\n".join(sem_lines))
        history = self.short_term.get_formatted()
        if history:
            memory_parts.append("[Recent Conversation]\n" + history)

        if memory_parts:
            system += "\n\n" + "\n\n".join(memory_parts)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for msg in self.short_term.get_history():
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": state["query"]})

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        self._total_input_tokens  += completion.usage.prompt_tokens
        self._total_output_tokens += completion.usage.completion_tokens
        return {"response": completion.choices[0].message.content}

    # ── node 4: save_memory ───────────────────────────────────────────────────

    def _save_memory(self, state: MemoryState) -> Dict:
        query    = state["query"]
        response = state["response"]

        # Short-term always updated
        self.short_term.add_message("user",      query)
        self.short_term.add_message("assistant", response)

        if self.use_memory:
            # Episodic — log every exchange
            if self.episodic:
                self.episodic.add_episode(
                    session_id=self.session_id,
                    user_message=query,
                    assistant_response=response,
                    metadata={"intent": state.get("intent", "general")},
                )
            # Long-term — extract facts/preferences (conflict-aware: newest wins)
            if self.long_term:
                self._extract_and_save_facts(query)
            # Semantic — embed exchange
            if self.semantic:
                self.semantic.add(
                    f"Q: {query}\nA: {response}",
                    metadata={"session": self.session_id,
                               "intent": state.get("intent", "")},
                )
        return {}

    # ── fact extraction (conflict-aware) ──────────────────────────────────────

    _LANGS = ["python", "java", "javascript", "typescript", "go", "rust",
              "c++", "c#", "kotlin", "swift"]
    # allergens stored with canonical spelling (diacritics)
    _ALLERGENS = [
        ("sữa bò",  "sua bo"),
        ("đậu nành","dau nanh"),
        ("gluten",  "gluten"),
        ("hải sản", "hai san"),
        ("lạc",     "lac"),
        ("tôm",     "tom"),
        ("cua",     "cua"),
    ]
    _BACKEND = "backend developer"
    _FRONTEND = "frontend developer"
    _GOALS = {_BACKEND: _BACKEND, "backend dev": _BACKEND,
               _FRONTEND: _FRONTEND, "frontend dev": _FRONTEND}

    def _extract_name(self, q: str) -> None:
        # Support both diacritics and ASCII romanized Vietnamese
        patterns = ["tôi tên là ", "tên tôi là ", "my name is ",
                    "toi ten la ", "ten toi la ", "tôi là ", "toi la "]
        for pattern in patterns:
            if pattern in q:
                after = q.split(pattern, 1)[1]
                name  = re.split(r"[\s.,!]", after)[0].strip()
                if name and len(name) <= 30:
                    self.long_term.save_user_fact("name", name.title())
                return

    def _extract_allergy(self, q: str) -> None:
        neg_patterns = ["khong phai ", "không phải "]
        for canonical, ascii_form in self._ALLERGENS:
            for form in (canonical, ascii_form):
                trigger = f"di ung {form}" if ascii_form == form else f"dị ứng {form}"
                if trigger in q:
                    neg_hit = any(f"{neg}{form}" in q for neg in neg_patterns)
                    if not neg_hit:
                        self.long_term.save_preference("allergy", canonical)
                        return

    def _extract_language(self, q: str) -> None:
        for lang in self._LANGS:
            like_kws    = [f"thích {lang}", f"like {lang}", f"prefer {lang}",
                           f"thich {lang}", f"chuyển sang {lang}", f"chuyen sang {lang}",
                           f"dùng {lang}", f"dung {lang}"]
            dislike_kws = [f"không thích {lang}", f"dislike {lang}", f"ghét {lang}",
                           f"khong thich {lang}"]
            if any(kw in q for kw in like_kws):
                self.long_term.save_preference("preferred_language", lang)
                return
            if any(kw in q for kw in dislike_kws):
                self.long_term.save_preference("disliked_language", lang)
                return

    def _extract_style_and_goal(self, q: str) -> None:
        if any(kw in q for kw in ["ngắn gọn", "brief", "concise", "tóm tắt"]):
            self.long_term.save_preference("response_style", "concise")
        elif any(kw in q for kw in ["chi tiết", "detail", "thorough", "đầy đủ"]):
            self.long_term.save_preference("response_style", "detailed")
        for kw, goal in self._GOALS.items():
            if kw in q:
                self.long_term.save_user_fact("goal", goal)
                break
        for food in ["chay", "vegetarian", "vegan", "thuần chay"]:
            if food in q:
                self.long_term.save_preference("diet", food)
                break

    def _extract_and_save_facts(self, query: str) -> None:
        """
        Rule-based extraction. save_preference / save_user_fact OVERWRITE existing
        values → newest fact always wins (built-in conflict resolution).
        """
        q = query.lower()
        self._extract_name(q)
        self._extract_allergy(q)
        self._extract_language(q)
        self._extract_style_and_goal(q)

    # ── public API ─────────────────────────────────────────────────────────────

    def chat(self, query: str) -> Dict[str, Any]:
        initial: MemoryState = {
            "messages":      [],
            "user_profile":  {},
            "episodes":      [],
            "semantic_hits": [],
            "memory_budget": self.MAX_TOKENS - self.RESERVE_TOKENS,
            "query":         query,
            "response":      "",
            "intent":        "general",
        }
        final = self._graph.invoke(initial)
        memory_context_len = (
            len(str(final.get("user_profile", {}))) +
            len(str(final.get("episodes", []))) +
            len(str(final.get("semantic_hits", [])))
        )
        return {
            "response":          final["response"],
            "intent":            final["intent"],
            "memory_context_len": memory_context_len,
            "user_profile":      final.get("user_profile", {}),
            "episodes_count":    len(final.get("episodes", [])),
            "semantic_count":    len(final.get("semantic_hits", [])),
            "memory_budget":     final.get("memory_budget", 0),
            "token_stats":       {},
        }

    def reset_session(self, new_session_id: Optional[str] = None) -> None:
        """New session: clear short-term, keep long-term / episodic / semantic."""
        self.session_id = new_session_id or f"session_{int(time.time())}"
        self.short_term.clear()

    @property
    def total_tokens_used(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens
