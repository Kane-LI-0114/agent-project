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

from core.conversation import ConversationManager
from core.guardrails import check_input, detect_academic_level
from llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._llm = llm_client
        self._conv = conversation

    async def handle(self, user_input: str) -> str:
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
            return reply

        # --- Step 2: Detect academic level ---
        level = detect_academic_level(user_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        # --- Step 3: Build messages & call LLM ---
        self._conv.add_user_message(user_input)
        messages = self._conv.get_messages()

        try:
            reply = await self._llm.chat(messages)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("LLM call failed: %s", exc)
            reply = f"[ERROR] Failed to get a response from the LLM: {exc}"

        # --- Step 4: Store assistant reply ---
        self._conv.add_assistant_message(reply)
        return reply
