"""
webui.py
========
Web UI for the CSIT5900 Multi-turn Homework Tutoring Agent.

Provides a modern ChatGPT/Claude-style chat interface with:
- Preset demo question buttons for quick validation
- Free-form chat input
- Streaming-style response display
- Conversation history management

Usage
-----
    python webui.py

Then open http://localhost:8000 in your browser.
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from config.settings import (
    ALLOWED_SUBJECTS,
    DEMO_PROMPTS,
    ENABLE_BOT_MARKDOWN_LATEX,
    MANDATORY_SUBJECTS,
    SEARCH_ENABLED,
    STRICT_MODE_ENABLED,
)
from core.conversation import ConversationManager
from core.response_handler import ResponseHandler
from core.search import SearchService
from llm import get_llm_client

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# App & global state
# --------------------------------------------------------------------------- #
app = FastAPI(title="CSIT5900 SmartTutor")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

conversation = ConversationManager()
handler: ResponseHandler | None = None
handler_init_error: str | None = None


def get_handler() -> ResponseHandler:
    """
    Lazily initialise the response handler.

    This avoids import-time crashes when the LLM backend is not yet configured.
    """
    global handler, handler_init_error

    if handler is not None:
        return handler

    if handler_init_error is not None:
        raise RuntimeError(handler_init_error)

    try:
        llm_client = get_llm_client()
        handler = ResponseHandler(
            llm_client,
            conversation,
            search_service=SearchService(
                query_optimizer=get_llm_client("query_optimizer"),
                result_reviewer=get_llm_client("search_reviewer"),
            ),
            strict_reviewer=get_llm_client("strict_reviewer"),
            strict_generator=get_llm_client("strict_generator"),
            strict_auditor=get_llm_client("strict_auditor"),
            followup_suggester=get_llm_client("followup"),
        )
        return handler
    except (RuntimeError, ValueError) as exc:
        handler_init_error = str(exc)
        logger.error("Failed to initialise handler: %s", exc)
        raise RuntimeError(handler_init_error) from exc


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str
    search_mode: Literal["auto", "on", "off"] = "auto"
    mode: Literal["normal", "strict"] = "normal"
    selected_subjects: list[str] = Field(default_factory=lambda: list(ALLOWED_SUBJECTS))
    subject_change_note: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict[str, str]]
    mode: Literal["normal", "strict"] = "normal"
    strict_trace: list[dict[str, object]] | None = None
    follow_up_suggestions: list[str] = Field(default_factory=list)


class FollowUpRequest(BaseModel):
    user_message: str
    assistant_reply: str
    mode: Literal["normal", "strict"] = "normal"
    selected_subjects: list[str] = Field(default_factory=lambda: list(ALLOWED_SUBJECTS))
    exclude_suggestions: list[str] = Field(default_factory=list)


class FollowUpResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a user message and return the complete reply (non-streaming)."""
    try:
        payload = await get_handler().handle(
            req.message,
            req.search_mode,
            req.mode,
            selected_subjects=req.selected_subjects,
            subject_change_note=req.subject_change_note,
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "reply": f"[ERROR] {exc}",
                "sources": [],
                "mode": req.mode,
                "strict_trace": None,
                "follow_up_suggestions": [],
            },
        )
    return ChatResponse(
        reply=payload.reply,
        sources=payload.sources,
        mode=payload.mode,
        strict_trace=payload.strict_trace,
        follow_up_suggestions=payload.follow_up_suggestions,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Process a user message and stream the reply as Server-Sent Events.

    Each event carries a JSON payload:
        data: {"token": "<chunk>"}\n\n
    The stream ends with:
        data: [DONE]\n\n
    """
    async def event_generator():
        try:
            response_handler = get_handler()
        except RuntimeError as exc:
            payload = json.dumps(
                {"type": "token", "token": f"[ERROR] {exc}"},
                ensure_ascii=False,
            )
            yield f"data: {payload}\n\n"
            done_payload = json.dumps({"type": "done"}, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"
            return

        if req.mode == "strict":
            async for event in response_handler.handle_strict_stream(
                req.message,
                req.search_mode,
                selected_subjects=req.selected_subjects,
                subject_change_note=req.subject_change_note,
            ):
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            return

        async for event in response_handler.handle_stream(
            req.message,
            req.search_mode,
            selected_subjects=req.selected_subjects,
            subject_change_note=req.subject_change_note,
        ):
            # JSON-encode the chunk so special characters (newlines, quotes)
            # don't break the SSE framing.
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/followups", response_model=FollowUpResponse)
async def followups(req: FollowUpRequest):
    """Generate or regenerate reply suggestions without mutating chat history."""
    try:
        suggestions = await get_handler().generate_follow_up_suggestions(
            user_input=req.user_message,
            assistant_reply=req.assistant_reply,
            selected_subjects=req.selected_subjects,
            mode=req.mode,
            exclude_suggestions=req.exclude_suggestions,
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "suggestions": [],
                "error": str(exc),
            },
        )
    return FollowUpResponse(suggestions=suggestions)


@app.post("/api/clear")
async def clear():
    """Clear the conversation history."""
    conversation.clear()
    return JSONResponse({"status": "ok"})


@app.get("/api/demos")
async def demos():
    """Return the list of demo prompts for the preset buttons."""
    return JSONResponse(
        [{"key": k, "prompt": v} for k, v in DEMO_PROMPTS.items()]
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the tab icon from the static asset path expected by browsers."""
    return RedirectResponse(url="/static/favicon.svg", status_code=307)


# --------------------------------------------------------------------------- #
# Serve the frontend
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace(
        "__ENABLE_BOT_MARKDOWN_LATEX__",
        "true" if ENABLE_BOT_MARKDOWN_LATEX else "false",
    )
    html = html.replace(
        "__SEARCH_ENABLED__",
        "true" if SEARCH_ENABLED else "false",
    )
    html = html.replace(
        "__STRICT_MODE_ENABLED__",
        "true" if STRICT_MODE_ENABLED else "false",
    )
    html = html.replace(
        "__ALLOWED_SUBJECTS__",
        json.dumps(ALLOWED_SUBJECTS, ensure_ascii=False),
    )
    html = html.replace(
        "__MANDATORY_SUBJECTS__",
        json.dumps(MANDATORY_SUBJECTS, ensure_ascii=False),
    )
    return HTMLResponse(html)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    print("=" * 55)
    print("  CSIT5900 SmartTutor Web UI")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000)
