"""
run_benchmark.py
Run the full benchmark: 10 conversations × (with memory | without memory).
Produces benchmark_report.md when done.

Usage:
    # set OPENAI_API_KEY in .env first
    python run_benchmark.py
"""
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from benchmark.conversations import CONVERSATIONS
from benchmark.evaluator import BenchmarkEvaluator, ConversationResult, TurnResult
from benchmark.report_generator import generate_report
from agent.multi_memory_agent import MultiMemoryAgent


# ── helpers ────────────────────────────────────────────────────────────────────

def run_conversation(
    agent: MultiMemoryAgent,
    conv,
    evaluator: BenchmarkEvaluator,
) -> ConversationResult:
    result = ConversationResult(
        conv_id=conv.id,
        conv_name=conv.name,
        has_memory=agent.use_memory,
    )
    agent.reset_session(conv.session_id)
    context_history: list[str] = []

    for i, turn in enumerate(conv.turns):
        t0 = time.time()
        out = agent.chat(turn.user)
        elapsed = time.time() - t0

        tr = TurnResult(
            turn_id=i,
            query=turn.user,
            response=out["response"],
            intent=out["intent"],
            memory_context_len=out["memory_context_len"],
            response_time=elapsed,
            has_memory=agent.use_memory,
        )
        tr.memory_hit = evaluator.evaluate_memory_hit(tr, turn.expected_memory_use)
        tr.keyword_relevance = evaluator.evaluate_keyword_relevance(
            turn.user, out["response"], context_history
        )
        context_history += [turn.user, out["response"]]
        result.turns.append(tr)

        icon = "OK" if tr.memory_hit else "XX"
        print(f"  [{icon}] T{i+1} ({out['intent'][:12]:12s}) "
              f"{elapsed:.1f}s  Q: {turn.user[:45]!r}")

    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env")
        return

    evaluator = BenchmarkEvaluator()
    mem_dir   = "data/with_memory"
    nomem_dir = "data/without_memory"
    Path(mem_dir).mkdir(parents=True, exist_ok=True)
    Path(nomem_dir).mkdir(parents=True, exist_ok=True)

    # ── Phase 1: WITH memory ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Agent WITH MEMORY (gpt-4o-mini)")
    print("=" * 60)
    mem_agent = MultiMemoryAgent(
        user_id="bench_user",
        data_dir=mem_dir,
        use_memory=True,
        api_key=api_key,
    )
    with_results: list[ConversationResult] = []
    for conv in CONVERSATIONS:
        print(f"\n[{conv.id}] {conv.name}")
        with_results.append(run_conversation(mem_agent, conv, evaluator))

    # ── Phase 2: WITHOUT memory ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Agent WITHOUT MEMORY (gpt-4o-mini)")
    print("=" * 60)
    nomem_agent = MultiMemoryAgent(
        user_id="bench_user_nomem",
        data_dir=nomem_dir,
        use_memory=False,
        api_key=api_key,
    )
    without_results: list[ConversationResult] = []
    for conv in CONVERSATIONS:
        print(f"\n[{conv.id}] {conv.name}")
        without_results.append(run_conversation(nomem_agent, conv, evaluator))

    # ── metrics + report ──────────────────────────────────────────────────────
    metrics = evaluator.compute_metrics(with_results, without_results)
    report  = generate_report(
        with_results, without_results, metrics,
        mem_agent.total_tokens_used,
        nomem_agent.total_tokens_used,
    )

    report_path = "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print(f"  Report saved → {report_path}")
    print(f"  Memory Hit Rate : "
          f"{metrics['memory_hit_rate']['with_memory']:.1%} (with) vs "
          f"{metrics['memory_hit_rate']['without_memory']:.1%} (without)")
    print(f"  Tokens (with)   : {mem_agent.total_tokens_used:,}")
    print(f"  Tokens (without): {nomem_agent.total_tokens_used:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
