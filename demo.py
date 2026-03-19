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
- Search mode switching
- Strict mode switching

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
    mode normal   -> Use the normal response pipeline
    mode strict   -> Use the strict reviewed pipeline
    search auto   -> Search only when needed
    search on     -> Always search
    search off    -> Disable search
    status        -> Show current mode settings
    clear         -> Clear conversation history and start fresh
    exit / quit   -> Exit the program
"""

import asyncio
import logging
import sys

from config.settings import DEMO_PROMPTS
from config.settings import SEARCH_ENABLED, STRICT_MODE_ENABLED
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
    print("Use 'mode <normal|strict>' and 'search <auto|on|off>' to switch pipelines.")
    print()
    print("Demo shortcuts:")
    for key, prompt in DEMO_PROMPTS.items():
        print(f"  {key:14s} -> {prompt}")
    print()
    print("Controls:")
    print("  mode normal    -> normal response pipeline")
    print("  mode strict    -> strict reviewed pipeline")
    print("  search auto    -> search only when needed")
    print("  search on      -> always search")
    print("  search off     -> disable search")
    print("  status         -> show current settings")
    print()


def print_status(mode: str, search_mode: str) -> None:
    """Print current runtime control settings."""
    print(f"[STATUS] Mode: {mode} | Search: {search_mode}")
    if not STRICT_MODE_ENABLED:
        print("[INFO] Strict mode is disabled by configuration.")
    if not SEARCH_ENABLED:
        print("[INFO] Search is disabled by configuration.")
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
        strict_reviewer=get_llm_client("strict_reviewer"),
        strict_generator=get_llm_client("strict_generator"),
        strict_auditor=get_llm_client("strict_auditor"),
    )
    current_mode = "normal"
    current_search_mode = "auto" if SEARCH_ENABLED else "off"
    print_status(current_mode, current_search_mode)

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

        if user_input.lower() == "status":
            print_status(current_mode, current_search_mode)
            continue

        if user_input.lower().startswith("mode "):
            requested_mode = user_input.lower().split(maxsplit=1)[1].strip()
            if requested_mode not in {"normal", "strict"}:
                print("[ERROR] Invalid mode. Use 'mode normal' or 'mode strict'.\n")
                continue
            if requested_mode == "strict" and not STRICT_MODE_ENABLED:
                print("[ERROR] Strict mode is disabled in configuration.\n")
                continue
            current_mode = requested_mode
            print_status(current_mode, current_search_mode)
            continue

        if user_input.lower().startswith("search "):
            requested_search_mode = user_input.lower().split(maxsplit=1)[1].strip()
            if requested_search_mode not in {"auto", "on", "off"}:
                print("[ERROR] Invalid search mode. Use 'search auto', 'search on', or 'search off'.\n")
                continue
            if requested_search_mode != "off" and not SEARCH_ENABLED:
                print("[ERROR] Search is disabled in configuration.\n")
                continue
            current_search_mode = requested_search_mode
            print_status(current_mode, current_search_mode)
            continue

        # Demo shortcut expansion
        if user_input.lower() in DEMO_PROMPTS:
            user_input = DEMO_PROMPTS[user_input.lower()]
            print(f"[DEMO] {user_input}")

        # Process the turn
        payload = await handler.handle(user_input, current_search_mode, current_mode)
        print(f"SmartTutor: {payload.reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
