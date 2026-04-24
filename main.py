"""
main.py — interactive demo of the Multi-Memory Agent.

Usage:
    python main.py

Commands inside the chat:
    quit          → exit
    new session   → start fresh short-term, keep long-term / episodic / semantic
    show memory   → print current stored preferences, facts, recent episodes
"""
import os

from dotenv import load_dotenv

load_dotenv()

from agent.multi_memory_agent import MultiMemoryAgent


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env")
        return

    agent = MultiMemoryAgent(
        user_id="demo_user",
        data_dir="data/demo",
        use_memory=True,
        api_key=api_key,
    )

    print("=" * 60)
    print("Multi-Memory Agent — Lab #17 (gpt-4o-mini + LangGraph)")
    print("=" * 60)
    print(f"Session: {agent.session_id}")
    print("Commands: 'quit' | 'new session' | 'show memory'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Bye!")
            break

        if user_input.lower() == "new session":
            agent.reset_session()
            print(f"[New session started: {agent.session_id}]\n")
            continue

        if user_input.lower() == "show memory":
            prefs   = agent.long_term.get_preferences()
            facts   = agent.long_term.get_user_facts()
            recent  = agent.episodic.get_recent(3)
            sem_cnt = agent.semantic.count()
            print(f"\n  Preferences  : {prefs or '(none)'}")
            print(f"  Facts        : {facts or '(none)'}")
            print(f"  Episodes     : {len(agent.episodic)} total, "
                  f"{len(recent)} shown below")
            for ep in recent:
                print(f"    [{ep.session_id}] {ep.user_message[:60]!r}")
            print(f"  Semantic docs: {sem_cnt}\n")
            continue

        result = agent.chat(user_input)
        print(f"\nAgent [{result['intent']}]: {result['response']}\n")


if __name__ == "__main__":
    main()
