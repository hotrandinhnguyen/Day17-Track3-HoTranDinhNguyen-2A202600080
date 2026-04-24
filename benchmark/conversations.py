"""
10 multi-turn conversation scenarios for the benchmark.
Covers all 5 test groups required by the rubric:
  Group A — Profile Recall        (conv 01, 02)
  Group B — Conflict Update       (conv 03, 04)
  Group C — Episodic Recall       (conv 05, 06)
  Group D — Semantic Retrieval    (conv 07, 08)
  Group E — Trim / Token Budget   (conv 09, 10)
"""
from dataclasses import dataclass, field
from typing import List

# Memory backend tag constants (used in Turn.expected_memory_use)
LT_PYTHON    = "long_term_python"
SEM_ST       = "semantic_short_term"
ST_SEM       = "short_term_semantic"
LT_RECALL    = "long_term_recall"
LT_SAVE      = "long_term_save"
EP_RECALL    = "episodic_recall"


@dataclass
class Turn:
    user: str
    expected_memory_use: str = ""   # which backend should fire
    tags: List[str] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    name: str
    group: str
    session_id: str
    description: str
    turns: List[Turn]


CONVERSATIONS: List[Conversation] = [

    # ══════════════════════════════════════════════════════════════════
    # GROUP A — Profile Recall
    # ══════════════════════════════════════════════════════════════════

    Conversation(
        id="conv_01",
        name="Profile Recall — User Name",
        group="A: Profile Recall",
        session_id="sess_01",
        description="User states their name; agent must recall it several turns later.",
        turns=[
            Turn("Tôi tên là Minh.",
                 "long_term_save", ["name"]),
            Turn("Giải thích về list comprehension trong Python.",
                 "none", ["python"]),
            Turn("Cho ví dụ thực tế với list comprehension.",
                 "none", ["example"]),
            Turn("Rất hay! Bạn có nhớ tên tôi là gì không?",
                 "long_term_recall", ["name_recall"]),
            Turn("Cảm ơn. Tóm tắt lại những gì chúng ta đã nói hôm nay.",
                 "short_term", ["summary"]),
        ],
    ),

    Conversation(
        id="conv_02",
        name="Profile Recall — Language Preference",
        group="A: Profile Recall",
        session_id="sess_02",
        description="User states Python preference; new session agent applies it without being asked.",
        turns=[
            Turn("Tôi thích Python hơn Java vì cú pháp đơn giản.",
                 "long_term_save", ["preference"]),
            Turn("Giải thích về async/await.",
                 "long_term→python_focus", ["async"]),
            Turn("Cho ví dụ code async thực tế.",
                 LT_PYTHON, ["example"]),
            Turn("Tôi muốn học thêm về decorators.",
                 LT_PYTHON, ["decorator"]),
            Turn("Ngôn ngữ nào tôi đang dùng chính?",
                 "long_term_recall", ["preference_recall"]),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # GROUP B — Conflict Update
    # ══════════════════════════════════════════════════════════════════

    Conversation(
        id="conv_03",
        name="Conflict Update — Allergy Correction",
        group="B: Conflict Update",
        session_id="sess_03",
        description=(
            "User states wrong allergy then corrects it. "
            "Agent must store the CORRECTED value (newest wins)."
        ),
        turns=[
            Turn("Tôi dị ứng sữa bò.",
                 "long_term_save", ["allergy_wrong"]),
            Turn("Gợi ý thực đơn ăn sáng cho tôi.",
                 "long_term_recall→allergy", ["menu"]),
            Turn("Ồ nhầm rồi, tôi dị ứng đậu nành chứ không phải sữa bò.",
                 "long_term_overwrite", ["conflict_correction"]),
            Turn("Gợi ý lại thực đơn ăn sáng cho tôi nhé.",
                 "long_term_recall→new_allergy", ["menu_updated"]),
            Turn("Bạn đang lưu tôi dị ứng gì vậy?",
                 "long_term_recall→confirm", ["confirm"]),
        ],
    ),

    Conversation(
        id="conv_04",
        name="Conflict Update — Language Switch",
        group="B: Conflict Update",
        session_id="sess_04",
        description="User switches from Python to TypeScript; agent uses new preference.",
        turns=[
            Turn("Tôi thích Python và muốn học thêm.",
                 "long_term_save", ["pref_python"]),
            Turn("Cho tôi ví dụ về OOP.",
                 LT_PYTHON, ["oop"]),
            Turn("Gần đây tôi chuyển sang TypeScript rồi, thích TypeScript hơn.",
                 "long_term_overwrite", ["pref_change"]),
            Turn("Cho ví dụ OOP tương tự nhưng trong ngôn ngữ tôi đang dùng.",
                 "long_term→typescript", ["oop_ts"]),
            Turn("Tôi hiện đang dùng ngôn ngữ gì?",
                 "long_term_recall→confirm", ["confirm"]),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # GROUP C — Episodic Recall
    # ══════════════════════════════════════════════════════════════════

    Conversation(
        id="conv_05",
        name="Episodic Recall — Past Confusion",
        group="C: Episodic Recall",
        session_id="sess_05",
        description="Agent recalls that user was confused about async/await in a past session.",
        turns=[
            Turn("Trước đây tôi có hỏi về async/await và bị confused phải không?",
                 "episodic_recall", ["past_confusion"]),
            Turn("Bây giờ tôi đã hiểu hơn. Giải thích về generators.",
                 "short_term", ["generators"]),
            Turn("Generators và coroutines khác nhau thế nào?",
                 SEM_ST, ["comparison"]),
            Turn("Tôi từng bị confused về gì liên quan đến concurrency?",
                 "episodic_recall", ["recall_query"]),
            Turn("Cho bài tập thực hành phù hợp với trình độ của tôi.",
                 "episodic+long_term", ["exercise"]),
        ],
    ),

    Conversation(
        id="conv_06",
        name="Episodic Recall — Past Debug Session",
        group="C: Episodic Recall",
        session_id="sess_06",
        description="Agent recalls a specific bug the user debugged before.",
        turns=[
            Turn("Tôi gặp lỗi TypeError: 'NoneType' object is not iterable.",
                 "short_term", ["error"]),
            Turn("Lỗi xảy ra khi tôi gọi hàm get_items() trong vòng lặp.",
                 "short_term", ["context"]),
            Turn("Code: for item in get_items(): print(item)",
                 "episodic_save", ["code"]),
            Turn("Tôi có từng gặp lỗi tương tự trước đây không?",
                 "episodic_recall", ["recall"]),
            Turn("Tổng hợp các lỗi phổ biến tôi hay gặp.",
                 "episodic_recall", ["summary"]),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # GROUP D — Semantic Retrieval
    # ══════════════════════════════════════════════════════════════════

    Conversation(
        id="conv_07",
        name="Semantic Retrieval — Functional Programming",
        group="D: Semantic Retrieval",
        session_id="sess_07",
        description="Agent finds semantically related past exchanges about FP in Python.",
        turns=[
            Turn("Python có những tính năng functional programming nào?",
                 "semantic", ["functional"]),
            Turn("map(), filter(), reduce() hoạt động thế nào?",
                 SEM_ST, ["hof"]),
            Turn("List comprehension có liên quan đến functional programming không?",
                 SEM_ST, ["comprehension"]),
            Turn("Tìm cho tôi nội dung tương tự về FP mà tôi đã hỏi trước đây.",
                 "semantic_recall", ["semantic_query"]),
            Turn("Lambdas trong Python — so sánh với các ngôn ngữ khác.",
                 "semantic+long_term", ["lambda"]),
        ],
    ),

    Conversation(
        id="conv_08",
        name="Semantic Retrieval — Concurrency Topics",
        group="D: Semantic Retrieval",
        session_id="sess_08",
        description="Agent retrieves semantically related content about GIL and concurrency.",
        turns=[
            Turn("Giải thích về GIL trong Python.",
                 "semantic", ["GIL"]),
            Turn("GIL ảnh hưởng đến multithreading thế nào?",
                 SEM_ST, ["threading"]),
            Turn("Có cách nào bypass GIL không?",
                 SEM_ST, ["bypass"]),
            Turn("Tìm nội dung liên quan đến concurrency tôi đã hỏi.",
                 "semantic_recall", ["semantic_query"]),
            Turn("Asyncio so với multiprocessing — khi nào dùng cái nào?",
                 SEM_ST, ["comparison"]),
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # GROUP E — Trim / Token Budget
    # ══════════════════════════════════════════════════════════════════

    Conversation(
        id="conv_09",
        name="Token Budget — Long Context Auto-trim",
        group="E: Trim / Token Budget",
        session_id="sess_09",
        description=(
            "Long conversation accumulates many episodic + semantic entries. "
            "ContextManager must auto-trim low-priority sections to stay within budget."
        ),
        turns=[
            Turn("Giải thích về design patterns — Singleton, Factory, Observer, Strategy.",
                 "short_term", ["design_patterns"]),
            Turn("So sánh Singleton và Monostate pattern.",
                 ST_SEM, ["singleton"]),
            Turn("Factory Method vs Abstract Factory — khi nào dùng cái nào?",
                 ST_SEM, ["factory"]),
            Turn("Observer pattern trong event-driven architecture?",
                 ST_SEM, ["observer"]),
            Turn("Tóm tắt toàn bộ các design patterns đã nói và memory budget còn lại.",
                 "short_term+trim_check", ["summary_budget"]),
        ],
    ),

    Conversation(
        id="conv_10",
        name="Cross-Session — All Memory Types",
        group="E: Trim / Token Budget",
        session_id="sess_10",
        description=(
            "New session pulling from ALL 4 memory backends: "
            "profile (name + language), episodic (past confusion), semantic (past topics)."
        ),
        turns=[
            Turn("Chào! Tôi muốn tiếp tục học từ buổi trước.",
                 "long_term+episodic", ["cross_session_start"]),
            Turn("Nhắc lại tôi tên gì và đang học ngôn ngữ gì?",
                 "long_term_recall", ["profile_recall"]),
            Turn("Tôi từng bị confused về gì?",
                 "episodic_recall", ["episodic_recall"]),
            Turn("Tìm chủ đề tương tự với những gì tôi đã hỏi.",
                 "semantic_recall", ["semantic_recall"]),
            Turn("Gợi ý bài học tiếp theo phù hợp với tôi.",
                 "all_memory", ["recommendation"]),
        ],
    ),
]
