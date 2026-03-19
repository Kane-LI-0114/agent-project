"""
demo.py
=======
Main CLI demo entry point for the CSIT5900 Multi-turn Homework Tutoring Agent.

Provides an interactive command-line interface that demonstrates:
- Valid math, history, and geography homework questions
- Guardrail rejections for non-homework or off-subject inputs
- Multi-turn follow-up conversations
- Academic level adaptation
- Practice exercise generation
- Conversation summary

Usage
-----
    python demo.py

Built-in shortcut commands (type the keyword and press Enter):
    demo-math     -> Valid math question
    demo-history  -> Valid history question
    demo-geography -> Valid geography question
    demo-reject1  -> Non-homework daily-life question (rejected)
    demo-reject2  -> Off-subject question about a local university (rejected)
    demo-summary  -> Request conversation summary
    demo-level    -> Set academic level
    demo-exercise -> Request practice exercises
    clear         -> Clear conversation history and start fresh
    exit / quit   -> Exit the program
"""

import asyncio
import logging
import sys

from config.settings import DEMO_PROMPTS
from core.conversation import ConversationManager
from core.response_handler import ResponseHandler
from core.search import SearchService
from llm import get_llm_client

# Configure basic logging (only show warnings+ for cleaner CLI output)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def print_header() -> None:
    """Print the CLI welcome banner and available commands."""
    print("=" * 65)
    print("  CSIT5900 SmartTutor – Multi-turn Homework Tutoring Agent")
    print("=" * 65)
    print()
    print("Type a homework question and press Enter.")
    print("Type 'exit' or 'quit' to stop.  Type 'clear' to reset history.")
    print()
    print("Demo shortcuts:")
    for key, prompt in DEMO_PROMPTS.items():
        print(f"  {key:14s} -> {prompt}")
    print()


async def main() -> None:
    """Run the interactive CLI loop."""
    print_header()

    # Instantiate components
    try:
        llm_client = get_llm_client()
    except RuntimeError as exc:
        print(f"[FATAL] Cannot initialise LLM client: {exc}")
        sys.exit(1)

    conversation = ConversationManager()
    handler = ResponseHandler(
        llm_client,
        conversation,
        search_service=SearchService(query_optimizer=get_llm_client("query_optimizer")),
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Bye!")
            break

        # Clear conversation
        if user_input.lower() == "clear":
            conversation.clear()
            print("[INFO] Conversation history cleared.\n")
            continue

        # Demo shortcut expansion
        if user_input.lower() in DEMO_PROMPTS:
            user_input = DEMO_PROMPTS[user_input.lower()]
            print(f"[DEMO] {user_input}")

        # Process the turn
        payload = await handler.handle(user_input)
        print(f"SmartTutor: {payload.reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
