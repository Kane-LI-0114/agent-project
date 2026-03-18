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
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal

from config.settings import DEMO_PROMPTS, ENABLE_BOT_MARKDOWN_LATEX, SEARCH_ENABLED
from core.conversation import ConversationManager
from core.response_handler import ResponseHandler
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
        handler = ResponseHandler(llm_client, conversation)
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


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict[str, str]]


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a user message and return the complete reply (non-streaming)."""
    try:
        payload = await get_handler().handle(req.message, req.search_mode)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"reply": f"[ERROR] {exc}", "sources": []},
        )
    return ChatResponse(reply=payload.reply, sources=payload.sources)


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

        async for event in response_handler.handle_stream(req.message, req.search_mode):
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
    """Avoid browser default favicon 404 noise in logs."""
    return Response(status_code=204)


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
