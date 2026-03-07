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

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config.settings import DEMO_PROMPTS
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

# Initialise LLM client, conversation manager, and response handler once
llm_client = get_llm_client()
conversation = ConversationManager()
handler = ResponseHandler(llm_client, conversation)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a user message and return the assistant's reply."""
    reply = await handler.handle(req.message)
    return ChatResponse(reply=reply)


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


# --------------------------------------------------------------------------- #
# Serve the frontend
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


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
