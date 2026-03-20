"""
core/conversation.py
====================
Conversation memory and context management for multi-turn dialog.

Maintains a list of message dicts (OpenAI Chat Completions format) and
provides automatic truncation to prevent context-window overflow, using
token counting via the ``tiktoken`` library.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import tiktoken  # type: ignore[import-not-found]  # pylint: disable=import-error

from config.settings import MAX_HISTORY_TOKENS, MAX_HISTORY_TURNS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Default tiktoken encoding for GPT-4-class models
_ENCODING: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the approximate token count for *text*."""
    return len(_ENCODING.encode(text))


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Return the total token count across all messages.

    Each message has a small overhead (~4 tokens for role/name framing).
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += count_tokens(msg.get("content", ""))
    return total


class ConversationManager:
    """
    Manages multi-turn conversation history for the tutoring agent.

    Responsibilities
    ----------------
    - Store the evolving conversation (system + user/assistant turns).
    - Provide the full message list for each LLM call.
    - Automatically truncate oldest non-system messages when the token
      budget or turn limit is exceeded.
    - Track the user's stated academic level for adaptive responses.
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = MAX_HISTORY_TOKENS,
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> None:
        self._system_message: Dict[str, str] = {
            "role": "system",
            "content": system_prompt,
        }
        self._history: List[Dict[str, str]] = []
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._academic_level: Optional[str] = None

    # ----- public interface ------------------------------------------------ #

    @property
    def academic_level(self) -> Optional[str]:
        """The user's currently stated academic level, or None."""
        return self._academic_level

    @academic_level.setter
    def academic_level(self, value: str) -> None:
        self._academic_level = value

    def add_user_message(self, content: str) -> None:
        """Append a user turn to the history."""
        self._history.append({"role": "user", "content": content})
        self._truncate_if_needed()

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant turn to the history."""
        self._history.append({"role": "assistant", "content": content})
        self._truncate_if_needed()

    def get_messages(
        self,
        system_prompt_override: Optional[str] = None,
        system_notes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build and return the full message list for a Chat Completions call,
        including the system prompt and all retained history.

        If an academic level has been set, it is injected into the system
        prompt so the LLM can adapt its responses accordingly.
        """
        system = dict(self._system_message)  # shallow copy
        if system_prompt_override:
            system["content"] = system_prompt_override
        if self._academic_level:
            system["content"] += (
                f"\n\n# User Academic Level\n"
                f"The user is a {self._academic_level}. "
                f"Adjust answer depth accordingly."
            )
        messages = [system]
        for note in system_notes or ():
            if note and note.strip():
                messages.append({"role": "system", "content": note.strip()})
        return messages + list(self._history)

    def get_history_text(self) -> str:
        """
        Return a plain-text representation of the conversation history
        (used for summaries, logging, etc.).
        """
        lines: List[str] = []
        for msg in self._history:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset conversation history (keeps system prompt)."""
        self._history.clear()
        self._academic_level = None

    def turn_count(self) -> int:
        """Return the number of stored turns (user + assistant messages)."""
        return len(self._history)

    # ----- internal helpers ------------------------------------------------ #

    def _truncate_if_needed(self) -> None:
        """
        Remove oldest non-system messages when:
        - The number of turns exceeds ``_max_turns``, OR
        - The total token count exceeds ``_max_tokens``.

        Truncation removes the oldest completed exchange (user + assistant
        pair) to keep the dialog coherent.
        """
        # Hard turn limit
        while len(self._history) > self._max_turns:
            self._pop_oldest_exchange("turn limit")

        # Token budget
        while (
            len(self._history) > 2
            and count_messages_tokens(self.get_messages()) > self._max_tokens
        ):
            self._pop_oldest_exchange("token limit")

    def _pop_oldest_exchange(self, reason: str) -> None:
        """Remove the oldest coherent exchange from history."""
        if not self._history:
            return

        removed_count = min(2, len(self._history))
        removed = self._history[:removed_count]
        del self._history[:removed_count]
        logger.debug(
            "Truncated oldest exchange (%s): roles=%s",
            reason,
            [msg["role"] for msg in removed],
        )
