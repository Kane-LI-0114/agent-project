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
    subjects      -> Review or change the allowed subject scope
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
from typing import cast

from config.settings import (
    ALLOWED_SUBJECTS,
    DEMO_PROMPTS,
    MANDATORY_SUBJECTS,
    SEARCH_ENABLED,
    STRICT_MODE_ENABLED,
    build_subject_change_note,
    format_subject_display_name,
    normalize_subject_selection,
)
from core.conversation import ConversationManager
from core.response_handler import ChatMode, ResponseHandler
from core.search import SearchMode, SearchService
from llm import get_llm_client

# Configure basic logging (only show warnings+ for cleaner CLI output)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def format_subject_scope(subjects: list[str]) -> str:
    """Format the active subject scope for CLI display."""
    return ", ".join(format_subject_display_name(subject) for subject in normalize_subject_selection(subjects))


def print_header(selected_subjects: list[str]) -> None:
    """Print the CLI welcome banner and available commands."""
    print("=" * 65)
    print("  CSIT5900 SmartTutor – Multi-turn Homework Tutoring Agent")
    print("=" * 65)
    print()
    print("Type a homework question and press Enter.")
    print("Type 'exit' or 'quit' to stop.  Type 'clear' to reset history.")
    print("Use 'mode <normal|strict>' and 'search <auto|on|off>' to switch pipelines.")
    print("Type 'subjects' to change the allowed subject scope.")
    print(f"Current subjects: {format_subject_scope(selected_subjects)}")
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
    print("  subjects       -> change allowed subject scope")
    print("  status         -> show current settings")
    print()


def print_status(mode: ChatMode, search_mode: SearchMode) -> None:
    """Print current runtime control settings."""
    print(f"[STATUS] Mode: {mode} | Search: {search_mode}")
    if not STRICT_MODE_ENABLED:
        print("[INFO] Strict mode is disabled by configuration.")
    if not SEARCH_ENABLED:
        print("[INFO] Search is disabled by configuration.")
    print()


def configure_subjects(current_subjects: list[str]) -> list[str]:
    """Interactively update the CLI subject scope."""
    selected = normalize_subject_selection(current_subjects)
    option_map = {
        str(index): subject
        for index, subject in enumerate(ALLOWED_SUBJECTS[len(MANDATORY_SUBJECTS):], start=1)
    }

    while True:
        print("\nAllowed subjects")
        print("  Required:")
        for subject in MANDATORY_SUBJECTS:
            print(f"    - {format_subject_display_name(subject)} (required)")
        print("  Optional:")
        for index, subject in option_map.items():
            mark = "x" if subject in selected else " "
            print(f"    {index}. [{mark}] {format_subject_display_name(subject)}")
        print("\nCommands: number=toggle optional subject · all=select all · core=math+history only · done=apply · cancel=keep current")

        try:
            choice = input("Subjects: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return current_subjects

        if not choice:
            continue
        if choice == "done":
            return normalize_subject_selection(selected)
        if choice == "cancel":
            return current_subjects
        if choice == "all":
            selected = list(ALLOWED_SUBJECTS)
            continue
        if choice == "core":
            selected = list(MANDATORY_SUBJECTS)
            continue
        if choice in option_map:
            subject = option_map[choice]
            if subject in selected:
                selected = [item for item in selected if item != subject]
            else:
                selected = normalize_subject_selection(selected + [subject])
            continue

        print("[WARN] Invalid subject command.")


async def main() -> None:
    """Run the interactive CLI loop."""
    selected_subjects = list(ALLOWED_SUBJECTS)
    pending_subject_change_note: str | None = None
    print_header(selected_subjects)

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
    current_mode: ChatMode = "normal"
    current_search_mode: SearchMode = "auto" if SEARCH_ENABLED else "off"
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

        if user_input.lower() == "subjects":
            updated_subjects = configure_subjects(selected_subjects)
            if normalize_subject_selection(updated_subjects) != normalize_subject_selection(selected_subjects):
                selected_subjects = normalize_subject_selection(updated_subjects)
                pending_subject_change_note = build_subject_change_note(selected_subjects)
                print(f"[INFO] Subjects updated: {format_subject_scope(selected_subjects)}\n")
            else:
                print(f"[INFO] Subjects unchanged: {format_subject_scope(selected_subjects)}\n")
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
            current_mode = cast(ChatMode, requested_mode)
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
            current_search_mode = cast(SearchMode, requested_search_mode)
            print_status(current_mode, current_search_mode)
            continue

        # Demo shortcut expansion
        if user_input.lower() in DEMO_PROMPTS:
            user_input = DEMO_PROMPTS[user_input.lower()]
            print(f"[DEMO] {user_input}")

        # Process the turn
        payload = await handler.handle(
            user_input,
            current_search_mode,
            current_mode,
            selected_subjects=selected_subjects,
            subject_change_note=pending_subject_change_note,
        )
        pending_subject_change_note = None
        print(f"SmartTutor: {payload.reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
