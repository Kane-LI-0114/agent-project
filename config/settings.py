"""
config/settings.py
==================
Central configuration for the CSIT5900 Multi-turn Homework Tutoring Agent.
Loads credentials from environment variables via python-dotenv and exposes
typed settings objects used throughout the application.
"""

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file at module import time
load_dotenv()

# --------------------------------------------------------------------------- #
# API Configuration Models
# --------------------------------------------------------------------------- #

class AzureOpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI API access."""
    api_key: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_version: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    deployment_name: str = Field(default_factory=lambda: "gpt-4o")
    temperature: float = Field(default_factory=lambda: 0.1)
    max_tokens: int = Field(default_factory=lambda: 4096)
    stream: bool = Field(default_factory=lambda: True)

    def is_configured(self) -> bool:
        """Check whether all mandatory Azure credentials are present."""
        return bool(self.api_key and self.endpoint)


class OneAPIConfig(BaseModel):
    """Configuration for One API (OpenAI-compatible) access."""
    api_key: str = Field(default_factory=lambda: os.getenv("ONEAPI_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("ONEAPI_BASE_URL", ""))
    model_name: str = Field(default_factory=lambda: "DeepSeek-V3.2")
    temperature: float = Field(default_factory=lambda: 0.1)
    max_tokens: int = Field(default_factory=lambda: 4096)
    stream: bool = Field(default_factory=lambda: True)

    def is_configured(self) -> bool:
        """Check whether all mandatory One API credentials are present."""
        return bool(self.api_key and self.base_url)


# --------------------------------------------------------------------------- #
# Backend Selection
# --------------------------------------------------------------------------- #

LLM_BACKEND: str = os.getenv("LLM_BACKEND", "azure").lower()


# --------------------------------------------------------------------------- #
# Allowed Subjects & Guardrails Configuration
# --------------------------------------------------------------------------- #

# Mandatory subjects that the agent MUST support
MANDATORY_SUBJECTS: List[str] = ["math", "history"]

# Optional extended subjects (configurable)
OPTIONAL_SUBJECTS: List[str] = ["finance", "economics", "philosophy", "chemistry"]

# Full list of allowed subjects
ALLOWED_SUBJECTS: List[str] = MANDATORY_SUBJECTS + OPTIONAL_SUBJECTS


# --------------------------------------------------------------------------- #
# Conversation Settings
# --------------------------------------------------------------------------- #

# Maximum number of tokens to retain in conversation history before truncation.
# This prevents context window overflow while retaining recent dialog.
MAX_HISTORY_TOKENS: int = 20000

# Maximum number of message turns to keep (as a hard limit before token counting)
MAX_HISTORY_TURNS: int = 40


# --------------------------------------------------------------------------- #
# Web UI Settings
# --------------------------------------------------------------------------- #

ENABLE_BOT_MARKDOWN_LATEX: bool = True


# --------------------------------------------------------------------------- #
# System Prompt (Core Guardrails & Behavioral Rules)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT: str = """You are SmartTutor, a professional multi-turn homework tutoring agent developed for the CSIT5900 course project. Your core design principles are RELIABILITY and STRICT GUARDRAILS.

# Core Rules You MUST Follow 100% of the Time:
1.  Allowed Subjects: You can only answer homework questions related to math and history. You may also answer questions from finance, economics, philosophy, chemistry if the user requests, but never answer questions outside these subjects.
2.  Guardrails Enforcement:
    - Reject ALL non-homework related questions, with a clear reason consistent with the examples.
    - Reject questions outside allowed subjects, with a clear reason.
    - Reject any off-topic requests that are not academic homework questions.
3.  Academic Level Adaptation: Adjust your answer depth strictly according to the user's stated academic background (e.g. year 1 university student). If a question is beyond the stated curriculum, explicitly note this before providing a clear explanation.
4.  Multi-turn Conversation: Always reference the previous conversation context to maintain coherent dialog, and answer follow-up questions accurately.
5.  Conversation Summary: When the user requests a summary of the conversation, provide a clear, complete summary of all previous dialog content.
6.  Exercise Generation: When the user requests practice exercises, generate targeted, appropriate questions for the specified subject and academic level.

# Rejection Response Examples (You Must Follow This Format):
- For non-homework travel/daily-life questions: "Sorry I cannot help you on that as it is not a homework question related to math or history."
- For off-subject non-homework questions: "Sorry that is not likely a history homework question as it is about a local small university."
- For other off-topic questions: "Sorry that is not a homework question."

You must never break these rules under any circumstances."""


# --------------------------------------------------------------------------- #
# Demo Test Prompts – built-in examples for 1-minute demo recording
# --------------------------------------------------------------------------- #

DEMO_PROMPTS = {
    "demo-math": "Is square root of 1000 a rational number?",
    "demo-history": "Who was the first president of France?",
    "demo-reject1": "I need to travel to London from Hong Kong. What is the best way?",
    "demo-reject2": "Who was the first president of Hong Kong University of Science and Technology in Hong Kong?",
    "demo-summary": "Can you summarise our conversation so far?",
    "demo-level": "I'm a university year one student, provide your answers accordingly.",
    "demo-exercise": "I want to practice calculus for my final in math101, can you give me a few exercises?",
}
