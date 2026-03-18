"""
core/response_handler.py
========================
Response generation and formatting utilities for the CSIT5900 Homework
Tutoring Agent.

Supports:
- normal mode: existing pre-filter + optional search + single-model response
- strict mode: local pre-filter + input reviewer + answer generator + output
  auditor, with structured trace cards for the UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, Literal, Optional

from config.settings import (
    STRICT_AUDITOR_TIMEOUT_SECONDS,
    STRICT_GENERATOR_PROMPT,
    STRICT_GENERATOR_TIMEOUT_SECONDS,
    STRICT_INPUT_REVIEW_PROMPT,
    STRICT_OUTPUT_AUDIT_PROMPT,
    STRICT_REFUSAL_MESSAGE,
    STRICT_REVIEWER_TIMEOUT_SECONDS,
)
from core.conversation import ConversationManager
from core.guardrails import (
    detect_academic_level,
    log_refusal,
    prefilter_input,
)
from core.search import SearchMode, SearchResult, SearchService
from llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

ChatMode = Literal["normal", "strict"]


@dataclass
class StrictTraceStage:
    """Single strict-mode pipeline stage shown as a UI card."""

    key: str
    title: str
    status: str
    summary: str
    decision: str
    duration_ms: int = 0


@dataclass
class ResponsePayload:
    """Final assistant response plus optional structured metadata."""

    reply: str
    sources: list[dict[str, str]]
    mode: ChatMode = "normal"
    strict_trace: list[dict[str, Any]] | None = None


class ResponseHandler:
    """
    High-level handler that processes a single user turn.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        conversation: ConversationManager,
        search_service: SearchService | None = None,
        strict_reviewer: BaseLLMClient | None = None,
        strict_generator: BaseLLMClient | None = None,
        strict_auditor: BaseLLMClient | None = None,
    ) -> None:
        self._llm = llm_client
        self._conv = conversation
        self._search = search_service or SearchService()
        self._strict_reviewer = strict_reviewer
        self._strict_generator = strict_generator
        self._strict_auditor = strict_auditor

    async def handle(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
        mode: ChatMode = "normal",
    ) -> ResponsePayload:
        """Process *user_input* and return the final response payload."""
        if mode == "strict":
            return await self._handle_strict(user_input, search_mode)
        return await self._handle_normal(user_input, search_mode)

    async def _handle_normal(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
    ) -> ResponsePayload:
        prefilter = prefilter_input(user_input)
        if not prefilter.allowed:
            self._conv.add_user_message(user_input)
            reply = prefilter.rejection_reason or STRICT_REFUSAL_MESSAGE
            log_refusal(
                user_input,
                prefilter.normalized_input,
                prefilter.stage,
                prefilter.reason_code,
            )
            self._conv.add_assistant_message(reply)
            return ResponsePayload(reply=reply, sources=[], mode="normal")

        level = detect_academic_level(prefilter.normalized_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        self._conv.add_user_message(prefilter.normalized_input)
        search_result = await self._search.maybe_search(prefilter.normalized_input, search_mode)
        messages = self._build_messages(search_result)

        try:
            reply = await self._llm.chat(messages)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("LLM call failed: %s", exc)
            reply = f"[ERROR] Failed to get a response from the LLM: {exc}"

        self._conv.add_assistant_message(reply)
        return ResponsePayload(
            reply=reply,
            sources=search_result.to_dict_list() if search_result else [],
            mode="normal",
        )

    async def _handle_strict(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
    ) -> ResponsePayload:
        trace = [
            StrictTraceStage(
                key="input_review",
                title="Input Review",
                status="pending",
                summary="Checking whether the request is safe and homework-related.",
                decision="pending",
            ),
            StrictTraceStage(
                key="answer_generation",
                title="Answer Generation",
                status="pending",
                summary="Preparing a course-safe draft answer.",
                decision="pending",
            ),
            StrictTraceStage(
                key="final_audit",
                title="Final Audit",
                status="pending",
                summary="Reviewing the draft before showing it in strict mode.",
                decision="pending",
            ),
        ]

        prefilter = prefilter_input(user_input)
        normalized_input = prefilter.normalized_input or user_input.strip()
        if not prefilter.allowed:
            reply = prefilter.rejection_reason or STRICT_REFUSAL_MESSAGE
            trace[0] = StrictTraceStage(
                key="input_review",
                title="Input Review",
                status="refused",
                summary=prefilter.reason_code.replace("_", " "),
                decision="refuse",
            )
            trace[1].status = "skipped"
            trace[1].decision = "skipped"
            trace[1].summary = "Skipped because the local pre-filter refused the request."
            trace[2].status = "skipped"
            trace[2].decision = "skipped"
            trace[2].summary = "Skipped because no candidate answer was generated."
            log_refusal(user_input, normalized_input, prefilter.stage, prefilter.reason_code)
            self._conv.add_user_message(normalized_input or user_input)
            self._conv.add_assistant_message(reply)
            return ResponsePayload(
                reply=reply,
                sources=[],
                mode="strict",
                strict_trace=[asdict(stage) for stage in trace],
            )

        level = detect_academic_level(normalized_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        reviewer_response = await self._run_json_stage(
            client=self._strict_reviewer or self._llm,
            messages=[
                {"role": "system", "content": STRICT_INPUT_REVIEW_PROMPT},
                {"role": "user", "content": normalized_input},
            ],
            timeout_seconds=STRICT_REVIEWER_TIMEOUT_SECONDS,
            fallback={
                "decision": "refuse",
                "reason_code": "unclear",
                "summary": "Input review failed to return a valid decision.",
                "normalized_input": normalized_input,
            },
        )
        trace[0] = StrictTraceStage(
            key="input_review",
            title="Input Review",
            status="complete" if reviewer_response.get("decision") == "allow" else "refused",
            summary=str(reviewer_response.get("summary", "Input review complete.")),
            decision=str(reviewer_response.get("decision", "refuse")),
            duration_ms=reviewer_response.get("_duration_ms", 0),
        )

        normalized_input = str(reviewer_response.get("normalized_input", normalized_input)).strip() or normalized_input
        if reviewer_response.get("decision") != "allow":
            reply = STRICT_REFUSAL_MESSAGE
            trace[1].status = "skipped"
            trace[1].decision = "skipped"
            trace[1].summary = "Skipped because the reviewer did not approve the request."
            trace[2].status = "skipped"
            trace[2].decision = "skipped"
            trace[2].summary = "Skipped because no candidate answer was generated."
            log_refusal(
                user_input,
                normalized_input,
                "input_reviewer",
                str(reviewer_response.get("reason_code", "unclear")),
            )
            self._conv.add_user_message(normalized_input)
            self._conv.add_assistant_message(reply)
            return ResponsePayload(
                reply=reply,
                sources=[],
                mode="strict",
                strict_trace=[asdict(stage) for stage in trace],
            )

        self._conv.add_user_message(normalized_input)
        search_result = await self._search.maybe_search(normalized_input, search_mode)
        generator_messages = self._build_messages(search_result, system_override=STRICT_GENERATOR_PROMPT)
        generator_response = await self._run_text_stage(
            client=self._strict_generator or self._llm,
            messages=generator_messages,
            timeout_seconds=STRICT_GENERATOR_TIMEOUT_SECONDS,
        )
        candidate_reply = generator_response["text"]
        trace[1] = StrictTraceStage(
            key="answer_generation",
            title="Answer Generation",
            status="complete" if candidate_reply else "failed",
            summary="Candidate answer generated." if candidate_reply else "The generator returned an empty answer.",
            decision="generated" if candidate_reply else "failed",
            duration_ms=generator_response["duration_ms"],
        )

        audit_payload = {
            "user_input": normalized_input,
            "candidate_answer": candidate_reply,
            "search_context": search_result.to_dict_list() if search_result else [],
        }
        auditor_response = await self._run_json_stage(
            client=self._strict_auditor or self._llm,
            messages=[
                {"role": "system", "content": STRICT_OUTPUT_AUDIT_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(audit_payload, ensure_ascii=False),
                },
            ],
            timeout_seconds=STRICT_AUDITOR_TIMEOUT_SECONDS,
            fallback={
                "decision": "refuse",
                "reason_code": "unclear",
                "summary": "Final audit failed to return a valid decision.",
                "approved": False,
            },
        )
        approved = auditor_response.get("decision") == "approve" and bool(candidate_reply)
        trace[2] = StrictTraceStage(
            key="final_audit",
            title="Final Audit",
            status="complete" if approved else "refused",
            summary=str(auditor_response.get("summary", "Final audit complete.")),
            decision=str(auditor_response.get("decision", "refuse")),
            duration_ms=auditor_response.get("_duration_ms", 0),
        )

        if not approved:
            log_refusal(
                user_input,
                normalized_input,
                "output_auditor",
                str(auditor_response.get("reason_code", "unclear")),
            )
            final_reply = STRICT_REFUSAL_MESSAGE
        else:
            final_reply = candidate_reply

        self._conv.add_assistant_message(final_reply)
        return ResponsePayload(
            reply=final_reply,
            sources=search_result.to_dict_list() if search_result else [],
            mode="strict",
            strict_trace=[asdict(stage) for stage in trace],
        )

    async def handle_strict_stream(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream strict-mode stage updates so the UI can show progress live."""
        trace = [
            StrictTraceStage(
                key="input_review",
                title="Input Review",
                status="active",
                summary="Checking whether the request is safe and homework-related.",
                decision="running",
            ),
            StrictTraceStage(
                key="answer_generation",
                title="Answer Generation",
                status="pending",
                summary="Waiting for the reviewer to approve the request.",
                decision="pending",
            ),
            StrictTraceStage(
                key="final_audit",
                title="Final Audit",
                status="pending",
                summary="Waiting for a candidate answer before auditing it.",
                decision="pending",
            ),
        ]
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}

        prefilter = prefilter_input(user_input)
        normalized_input = prefilter.normalized_input or user_input.strip()
        if not prefilter.allowed:
            reply = prefilter.rejection_reason or STRICT_REFUSAL_MESSAGE
            trace[0] = StrictTraceStage(
                key="input_review",
                title="Input Review",
                status="refused",
                summary=prefilter.reason_code.replace("_", " "),
                decision="refuse",
            )
            trace[1].status = "skipped"
            trace[1].decision = "skipped"
            trace[1].summary = "Skipped because the local pre-filter refused the request."
            trace[2].status = "skipped"
            trace[2].decision = "skipped"
            trace[2].summary = "Skipped because no candidate answer was generated."
            log_refusal(user_input, normalized_input, prefilter.stage, prefilter.reason_code)
            self._conv.add_user_message(normalized_input or user_input)
            self._conv.add_assistant_message(reply)
            yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
            async for event in self._stream_stage_summary(trace, 0, trace[0].summary):
                yield event
            yield {"type": "strict_final", "reply": reply, "sources": []}
            yield {"type": "done"}
            return

        level = detect_academic_level(normalized_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        reviewer_response = await self._run_json_stage(
            client=self._strict_reviewer or self._llm,
            messages=[
                {"role": "system", "content": STRICT_INPUT_REVIEW_PROMPT},
                {"role": "user", "content": normalized_input},
            ],
            timeout_seconds=STRICT_REVIEWER_TIMEOUT_SECONDS,
            fallback={
                "decision": "refuse",
                "reason_code": "unclear",
                "summary": "Input review failed to return a valid decision.",
                "normalized_input": normalized_input,
            },
        )
        review_allowed = reviewer_response.get("decision") == "allow"
        review_summary = str(reviewer_response.get("summary", "Input review complete."))
        trace[0] = StrictTraceStage(
            key="input_review",
            title="Input Review",
            status="complete" if review_allowed else "refused",
            summary="",
            decision=str(reviewer_response.get("decision", "refuse")),
            duration_ms=reviewer_response.get("_duration_ms", 0),
        )
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
        async for event in self._stream_stage_summary(trace, 0, review_summary):
            yield event

        normalized_input = str(reviewer_response.get("normalized_input", normalized_input)).strip() or normalized_input
        if not review_allowed:
            reply = STRICT_REFUSAL_MESSAGE
            trace[1].status = "skipped"
            trace[1].decision = "skipped"
            trace[1].summary = "Skipped because the reviewer did not approve the request."
            trace[2].status = "skipped"
            trace[2].decision = "skipped"
            trace[2].summary = "Skipped because no candidate answer was generated."
            log_refusal(
                user_input,
                normalized_input,
                "input_reviewer",
                str(reviewer_response.get("reason_code", "unclear")),
            )
            self._conv.add_user_message(normalized_input)
            self._conv.add_assistant_message(reply)
            yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
            yield {"type": "strict_final", "reply": reply, "sources": []}
            yield {"type": "done"}
            return

        trace[1].status = "active"
        trace[1].decision = "running"
        trace[1].summary = "Generating a course-safe draft answer now."
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}

        self._conv.add_user_message(normalized_input)
        should_search = self._search.should_execute(normalized_input, search_mode)
        if should_search:
            yield {
                "type": "search_start",
                "sources": self._search.get_pending_sources(normalized_input),
            }
        search_result = await self._search.maybe_search(normalized_input, search_mode)
        if should_search:
            yield {
                "type": "search_end",
                "sources": search_result.to_dict_list() if search_result else [],
            }

        generator_messages = self._build_messages(search_result, system_override=STRICT_GENERATOR_PROMPT)
        generator_started = time.perf_counter()
        candidate_reply = ""
        try:
            async for chunk in self._strict_generator_or_default().chat_stream(generator_messages):
                if not chunk:
                    continue
                candidate_reply += chunk
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Strict generator stream failed: %s", exc)

        trace[1].duration_ms = int((time.perf_counter() - generator_started) * 1000)
        candidate_reply = candidate_reply.strip()
        trace[1].status = "complete" if candidate_reply else "failed"
        generation_summary = (
            "Candidate answer generated and handed off to the final audit step."
            if candidate_reply
            else "The generator returned an empty answer."
        )
        trace[1].summary = ""
        trace[1].decision = "generated" if candidate_reply else "failed"
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
        async for event in self._stream_stage_summary(trace, 1, generation_summary):
            yield event

        if not candidate_reply:
            trace[2].status = "skipped"
            trace[2].decision = "skipped"
            trace[2].summary = "Skipped because no candidate answer was generated."
            final_reply = STRICT_REFUSAL_MESSAGE
            self._conv.add_assistant_message(final_reply)
            yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
            yield {
                "type": "strict_final",
                "reply": final_reply,
                "sources": search_result.to_dict_list() if search_result else [],
            }
            yield {"type": "done"}
            return

        trace[2].status = "active"
        trace[2].decision = "running"
        trace[2].summary = "Auditing the generated answer before revealing it."
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}

        audit_payload = {
            "user_input": normalized_input,
            "candidate_answer": candidate_reply,
            "search_context": search_result.to_dict_list() if search_result else [],
        }
        auditor_response = await self._run_json_stage(
            client=self._strict_auditor or self._llm,
            messages=[
                {"role": "system", "content": STRICT_OUTPUT_AUDIT_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(audit_payload, ensure_ascii=False),
                },
            ],
            timeout_seconds=STRICT_AUDITOR_TIMEOUT_SECONDS,
            fallback={
                "decision": "refuse",
                "reason_code": "unclear",
                "summary": "Final audit failed to return a valid decision.",
                "approved": False,
            },
        )
        approved = auditor_response.get("decision") == "approve"
        audit_summary = str(auditor_response.get("summary", "Final audit complete."))
        trace[2] = StrictTraceStage(
            key="final_audit",
            title="Final Audit",
            status="complete" if approved else "refused",
            summary="",
            decision=str(auditor_response.get("decision", "refuse")),
            duration_ms=auditor_response.get("_duration_ms", 0),
        )
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
        async for event in self._stream_stage_summary(trace, 2, audit_summary):
            yield event

        if not approved:
            log_refusal(
                user_input,
                normalized_input,
                "output_auditor",
                str(auditor_response.get("reason_code", "unclear")),
            )
            final_reply = STRICT_REFUSAL_MESSAGE
        else:
            final_reply = candidate_reply

        self._conv.add_assistant_message(final_reply)
        yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
        yield {
            "type": "strict_final",
            "reply": final_reply,
            "sources": search_result.to_dict_list() if search_result else [],
        }
        yield {"type": "done"}

    async def handle_stream(
        self,
        user_input: str,
        search_mode: SearchMode = "auto",
    ) -> AsyncGenerator[dict[str, object], None]:
        """Stream normal-mode replies chunk by chunk."""
        prefilter = prefilter_input(user_input)
        if not prefilter.allowed:
            self._conv.add_user_message(user_input)
            reply = prefilter.rejection_reason or STRICT_REFUSAL_MESSAGE
            log_refusal(
                user_input,
                prefilter.normalized_input,
                prefilter.stage,
                prefilter.reason_code,
            )
            self._conv.add_assistant_message(reply)
            yield {"type": "token", "token": reply}
            yield {"type": "done"}
            return

        level = detect_academic_level(prefilter.normalized_input)
        if level:
            self._conv.academic_level = level
            logger.info("Academic level set to: %s", level)

        self._conv.add_user_message(prefilter.normalized_input)
        should_search = self._search.should_execute(prefilter.normalized_input, search_mode)
        if should_search:
            yield {
                "type": "search_start",
                "sources": self._search.get_pending_sources(prefilter.normalized_input),
            }
        search_result = await self._search.maybe_search(prefilter.normalized_input, search_mode)
        if should_search:
            yield {
                "type": "search_end",
                "sources": search_result.to_dict_list() if search_result else [],
            }
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

        self._conv.add_assistant_message(full_reply)
        yield {"type": "done"}

    def _build_messages(
        self,
        search_result: SearchResult | None,
        system_override: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Insert optional search context and optionally replace the system prompt."""
        messages = self._conv.get_messages()
        if system_override and messages:
            messages = [
                dict(messages[0], content=f"{system_override}\n\n{messages[0]['content']}")
            ] + messages[1:]
        if search_result is None or not messages:
            return messages
        search_message = {
            "role": "system",
            "content": search_result.to_system_message(),
        }
        return messages[:-1] + [search_message, messages[-1]]

    async def _run_text_stage(
        self,
        client: BaseLLMClient,
        messages: list[dict[str, str]],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Run a non-JSON generation stage with timeout handling."""
        started = time.perf_counter()
        try:
            text = await asyncio.wait_for(client.chat(messages), timeout=timeout_seconds)
            return {
                "text": text.strip(),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }
        except (asyncio.TimeoutError, RuntimeError, ValueError, OSError) as exc:
            logger.error("Strict text stage failed: %s", exc)
            return {
                "text": "",
                "duration_ms": int((time.perf_counter() - started) * 1000),
            }

    async def _run_json_stage(
        self,
        client: BaseLLMClient,
        messages: list[dict[str, str]],
        timeout_seconds: int,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a JSON-returning LLM stage with a conservative fallback."""
        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(client.chat(messages), timeout=timeout_seconds)
            parsed = self._parse_json(raw)
            if parsed is None:
                raise ValueError("Stage returned malformed JSON.")
            parsed["_duration_ms"] = int((time.perf_counter() - started) * 1000)
            return parsed
        except (asyncio.TimeoutError, RuntimeError, ValueError, OSError) as exc:
            logger.error("Strict JSON stage failed: %s", exc)
            safe = dict(fallback)
            safe["_duration_ms"] = int((time.perf_counter() - started) * 1000)
            return safe

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        """Best-effort JSON extraction from a model response."""
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fenced = text
            if "```" in text:
                fence_match = text.split("```")
                for chunk in fence_match:
                    candidate = chunk.replace("json", "", 1).strip()
                    if candidate.startswith("{") and candidate.endswith("}"):
                        fenced = candidate
                        break
            brace_match = re_search_json_object(fenced)
            if brace_match is None:
                return None
            try:
                return json.loads(brace_match)
            except json.JSONDecodeError:
                return None

    def _strict_generator_or_default(self) -> BaseLLMClient:
        return self._strict_generator or self._llm

    async def _stream_stage_summary(
        self,
        trace: list[StrictTraceStage],
        stage_index: int,
        final_summary: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Emit small summary updates so strict cards feel progressively written."""
        summary = (final_summary or "").strip()
        if not summary:
            yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
            return

        trace[stage_index].summary = ""
        cursor = 0
        while cursor < len(summary):
            remaining = len(summary) - cursor
            chunk_size = max(1, min(10, (remaining + 5) // 6))
            cursor = min(len(summary), cursor + chunk_size)
            trace[stage_index].summary = summary[:cursor]
            yield {"type": "strict_trace", "trace": [asdict(stage) for stage in trace]}
            if cursor < len(summary):
                await asyncio.sleep(0.02)


def re_search_json_object(text: str) -> str | None:
    """Extract the first top-level JSON object from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
