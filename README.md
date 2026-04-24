# Lab #17 — Multi-Memory Agent với LangGraph

**Sinh viên:** Hồ Trần Đình Nguyên | **MSSV:** 2A202600080  
**Model:** gpt-4o-mini (OpenAI) | **Framework:** LangGraph

---

## Kiến trúc

### LangGraph Flow

```
START
  │
  ▼
retrieve_memory(state)      ← load từ 4 backends dựa vào intent
  │  - user_profile   ← LongTermMemory (Redis / JSON KV)
  │  - episodes       ← EpisodicMemory (JSON log)
  │  - semantic_hits  ← SemanticMemory (ChromaDB)
  │  - messages       ← ShortTermMemory (sliding window)
  │
  ▼
build_prompt(state)         ← inject 4 section vào system prompt, tính memory_budget
  │  [User Profile]
  │  [Episodic Memory]
  │  [Semantic Context]
  │  [Recent Conversation]
  │
  ▼
generate(state)             ← gọi gpt-4o-mini với context đã inject
  │
  ▼
save_memory(state)          ← ghi lại vào tất cả backends phù hợp
  │
  ▼
END
```

### MemoryState (TypedDict)

```python
class MemoryState(TypedDict):
    messages:      list        # short-term conversation buffer
    user_profile:  dict        # long-term preferences + facts
    episodes:      list        # retrieved episodic memories
    semantic_hits: list        # ChromaDB similarity results
    memory_budget: int         # remaining token budget after injection
    query:         str
    response:      str
    intent:        str
```

### 4 Memory Backends

| Backend | Class | Storage | Vai trò |
|---------|-------|---------|---------|
| Short-term | `ShortTermMemory` | In-memory list | Lịch sử hội thoại hiện tại |
| Long-term | `LongTermMemory` | Redis / JSON file | Preferences, facts, allergy |
| Episodic | `EpisodicMemory` | JSON log file | Past exchanges, confusion points |
| Semantic | `SemanticMemory` | ChromaDB | Similarity search qua embedding |

---

## Cài đặt và chạy

```bash
# 1. Tạo và kích hoạt venv
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Cấu hình API key
cp .env.example .env
# Mở .env và điền: OPENAI_API_KEY=sk-...

# 4. Chạy demo tương tác
python main.py

# 5. Chạy benchmark đầy đủ (sinh ra benchmark_report.md)
python run_benchmark.py
```

### Lệnh trong demo (`main.py`)

| Lệnh | Tác dụng |
|------|---------|
| `quit` | Thoát |
| `new session` | Bắt đầu session mới (giữ long-term/episodic/semantic) |
| `show memory` | Xem preferences, facts, recent episodes |

---

## Kết quả benchmark

Xem [BENCHMARK.md](BENCHMARK.md) để biết kết quả 10 conversations so sánh với-memory vs không-memory.

**Tóm tắt:** With-Memory Agent pass 10/10, No-Memory Agent pass 0/10 trên các câu hỏi yêu cầu cross-turn hoặc cross-session context.

---

## Conflict Handling

```python
# Ví dụ test case từ rubric:
# Turn 1: "Tôi dị ứng sữa bò."     → profile: allergy = "sữa bò"
# Turn 3: "Nhầm, tôi dị ứng đậu nành chứ không phải sữa bò."
#                                   → profile: allergy = "đậu nành"  ← newest wins
```

`save_preference(key, value)` luôn **overwrite** — không append — nên fact mới nhất luôn thắng.
