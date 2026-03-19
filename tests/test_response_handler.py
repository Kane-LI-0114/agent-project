import unittest
from typing import Any, AsyncGenerator, List

from core.conversation import ConversationManager
from core.response_handler import ResponseHandler
from llm.base_client import BaseLLMClient


class ScriptedLLMClient(BaseLLMClient):
    def __init__(self, responses: List[str] | None = None, stream_responses: List[List[str]] | None = None) -> None:
        self._responses = list(responses or [])
        self._stream_responses = list(stream_responses or [])
        self.chat_calls: list[list[Any]] = []
        self.stream_calls: list[list[Any]] = []

    async def chat(self, messages: List[Any]) -> str:
        self.chat_calls.append(messages)
        if not self._responses:
            raise AssertionError("Unexpected chat() call with no scripted response left.")
        return self._responses.pop(0)

    async def chat_stream(self, messages: List[Any]) -> AsyncGenerator[str, None]:
        self.stream_calls.append(messages)
        chunks = self._stream_responses.pop(0) if self._stream_responses else []
        for chunk in chunks:
            yield chunk


class ResponseHandlerReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_mode_refuses_before_calling_llm(self) -> None:
        llm = ScriptedLLMClient()
        handler = ResponseHandler(llm, ConversationManager())

        payload = await handler.handle("Explain photosynthesis for my biology homework.", search_mode="off")

        self.assertIn("Sorry", payload.reply)
        self.assertEqual(llm.chat_calls, [])

    async def test_strict_mode_fails_closed_on_malformed_reviewer_json(self) -> None:
        reviewer = ScriptedLLMClient(responses=["not json"])
        generator = ScriptedLLMClient()
        auditor = ScriptedLLMClient()
        handler = ResponseHandler(
            llm_client=ScriptedLLMClient(),
            conversation=ConversationManager(),
            strict_reviewer=reviewer,
            strict_generator=generator,
            strict_auditor=auditor,
        )

        payload = await handler.handle(
            "Please explain the French Revolution.",
            search_mode="off",
            mode="strict",
        )

        self.assertIn("Sorry", payload.reply)
        self.assertEqual(generator.chat_calls, [])
        self.assertEqual(auditor.chat_calls, [])

    async def test_strict_mode_prefilter_blocks_biology_before_llm_pipeline(self) -> None:
        reviewer = ScriptedLLMClient(
            responses=[
                '{"decision":"allow","reason_code":"allowed","summary":"in scope","normalized_input":"Explain photosynthesis for my biology homework."}'
            ]
        )
        generator = ScriptedLLMClient(
            responses=["Photosynthesis converts light into chemical energy."]
        )
        auditor = ScriptedLLMClient(
            responses=[
                '{"decision":"refuse","reason_code":"out_of_scope","summary":"biology is not allowed","approved":false}',
                '{"decision":"refuse","reason_code":"out_of_scope","summary":"biology is not allowed","approved":false}',
                '{"decision":"refuse","reason_code":"out_of_scope","summary":"biology is not allowed","approved":false}',
            ]
        )
        handler = ResponseHandler(
            llm_client=ScriptedLLMClient(),
            conversation=ConversationManager(),
            strict_reviewer=reviewer,
            strict_generator=generator,
            strict_auditor=auditor,
        )

        payload = await handler.handle(
            "Explain photosynthesis for my biology homework.",
            search_mode="off",
            mode="strict",
        )

        self.assertIn("Sorry", payload.reply)
        self.assertEqual(len(reviewer.chat_calls), 0)
        self.assertEqual(len(generator.chat_calls), 0)
        self.assertEqual(len(auditor.chat_calls), 0)

    async def test_strict_mode_allows_safe_history_answer(self) -> None:
        reviewer = ScriptedLLMClient(
            responses=[
                '{"decision":"allow","reason_code":"allowed","summary":"valid history question","normalized_input":"Who was the first president of France during the Second Republic?"}'
            ]
        )
        generator = ScriptedLLMClient(
            responses=["Louis-Napoléon Bonaparte was elected in 1848 and later became Napoleon III."]
        )
        auditor = ScriptedLLMClient(
            responses=[
                '{"decision":"approve","reason_code":"approved","summary":"safe and in scope","approved":true}'
            ]
        )
        handler = ResponseHandler(
            llm_client=ScriptedLLMClient(),
            conversation=ConversationManager(),
            strict_reviewer=reviewer,
            strict_generator=generator,
            strict_auditor=auditor,
        )

        payload = await handler.handle(
            "Who was the first president of France during the Second Republic?",
            search_mode="off",
            mode="strict",
        )

        self.assertIn("Louis-Napoléon", payload.reply)
        self.assertEqual(len(auditor.chat_calls), 1)
