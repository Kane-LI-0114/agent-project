"""
core/response_handler.py
========================
Response generation and formatting utilities for the CSIT5900 Homework
Tutoring Agent.

Orchestrates the interaction between guardrails, conversation memory, and the
LLM client to produce a final response for each user turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncGenerator

from core.conversation import ConversationManager
from core.guardrails import check_input, detect_academic_level
from core.search import SearchMode, SearchResult, SearchService
from llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class ResponsePayload:
    """Final assistant response plus optional structured sources."""

    reply: str
    sources: list[dict[str, str]]


class ResponseHandler:
    """
    High-level handler that processes a single user turn:

    1. Run the code-level guardrail pre-check.
    2. Detect and store any academic-level declaration.
    3. Build the full message list from conversation memory.
    4. Call the LLM client.
    5. Store the assistant reply in conversation memory.
    6. Return the formatted response string.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        conversation: ConversationManager,
        search_service: SearchService | None = None,
    ) -> None:
        self._llm = llm_client
        self._conv = conversation
        self._search = search_service or SearchService()

    async def handle(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
    ) -> ResponsePayload:
        """
        Process *user_input* and return the assistant's response string.

        If the input is blocked by the code-level guardrail, the rejection
        reason is returned directly without calling the LLM.
        """
        # --- Step 1: Code-level guardrail ---
        is_allowed, rejection_reason = check_input(user_input)
        if not is_allowed:
            # Record the exchange even for rejections so the conversation
            # summary remains complete.
            self._conv.add_user_message(user_input)
            reply = rejection_reason or "Sorry, I cannot help with that."
            self._conv.add_assistant_message(reply)
            return ResponsePayload(reply=reply, sources=[])

        # --- Step 2: Detect academic level ---
        level = detect_academic_level(user_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        # --- Step 3: Build messages & call LLM ---
        self._conv.add_user_message(user_input)
        search_result = await self._search.maybe_search(user_input, search_mode)
        messages = self._build_messages(search_result)

        try:
            reply = await self._llm.chat(messages)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("LLM call failed: %s", exc)
            reply = f"[ERROR] Failed to get a response from the LLM: {exc}"

        # --- Step 4: Store assistant reply ---
        self._conv.add_assistant_message(reply)
        return ResponsePayload(
            reply=reply,
            sources=search_result.to_dict_list() if search_result else [],
        )

    async def handle_stream(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
    ) -> AsyncGenerator[dict[str, object], None]:
        """
        Process *user_input* and stream the assistant's response chunk by chunk.

        The guardrail check runs first (non-streaming).  If the input is
        blocked, the rejection message is yielded as a single chunk.
        The full assembled reply is stored in conversation history after the
        stream completes.
        """
        # --- Step 1: Code-level guardrail ---
        is_allowed, rejection_reason = check_input(user_input)
        if not is_allowed:
            self._conv.add_user_message(user_input)
            reply = rejection_reason or "Sorry, I cannot help with that."
            self._conv.add_assistant_message(reply)
            yield {"type": "token", "token": reply}
            yield {"type": "done"}
            return

        # --- Step 2: Detect academic level ---
        level = detect_academic_level(user_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        # --- Step 3: Build messages & stream from LLM ---
        self._conv.add_user_message(user_input)
        should_search = self._search.should_execute(user_input, search_mode)
        if should_search:
            yield {
                "type": "search_start",
                "sources": self._search.get_pending_sources(user_input),
            }
        search_result = await self._search.maybe_search(user_input, search_mode)
        if should_search:
            if search_result is not None:
                yield {
                    "type": "search_end",
                    "sources": search_result.to_dict_list(),
                }
            else:
                yield {"type": "search_end", "sources": []}
        messages = self._build_messages(search_result)

        full_reply = ""
        try:
            async for chunk in self._llm.chat_stream(messages):
                full_reply += chunk
                yield {"type": "token", "token": chunk}
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("LLM stream failed: %s", exc)
            error_msg = f"[ERROR] Failed to get a response from the LLM: {exc}"
            full_reply = error_msg
            yield {"type": "token", "token": error_msg}

        # --- Step 4: Store the complete assembled reply ---
        self._conv.add_assistant_message(full_reply)
        yield {"type": "done"}

    def _build_messages(self, search_result: SearchResult | None) -> list[dict[str, str]]:
        """Insert live search context immediately before the current user turn."""
        messages = self._conv.get_messages()
        if search_result is None or not messages:
            return messages
        search_message = {
            "role": "system",
            "content": search_result.to_system_message(),
        }
        return messages[:-1] + [search_message, messages[-1]]
