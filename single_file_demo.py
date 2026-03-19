"""
single_file_demo.py
====================
Self-contained single-file version of the CSIT5900 Multi-turn Homework
Tutoring Agent. All core logic (configuration, LLM client, conversation
memory, guardrails, response handling, CLI) is combined into one file for
quick demo execution.

Usage
-----
    1. Copy ``.env.example`` to ``.env`` and fill in your credentials.
    2. Run: ``python single_file_demo.py``

Supports both Azure OpenAI (``LLM_BACKEND=azure``) and One API
(``LLM_BACKEND=oneapi``) backends, controlled via the ``.env`` file.
"""

import asyncio
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import tiktoken  # type: ignore[import-not-found]  # pylint: disable=import-error
from dotenv import load_dotenv
from openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SmartTutor")


# =========================================================================== #
#  CONFIGURATION
# =========================================================================== #

load_dotenv()

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# One API
ONEAPI_API_KEY = os.getenv("ONEAPI_API_KEY", "")
ONEAPI_BASE_URL = os.getenv("ONEAPI_BASE_URL", "")
ONEAPI_MODEL_NAME = os.getenv("ONEAPI_MODEL_NAME", "gpt-4o")

# Backend selector
LLM_BACKEND = os.getenv("LLM_BACKEND", "azure").lower()

# Conversation limits
MAX_HISTORY_TOKENS = 6000
MAX_HISTORY_TURNS = 40

# Allowed subjects
ALLOWED_SUBJECTS = ["math", "history", "geography", "finance", "economics", "philosophy", "chemistry"]


def _format_subject_list(subjects):
    if not subjects:
        return ""
    if len(subjects) == 1:
        return subjects[0]
    if len(subjects) == 2:
        return f"{subjects[0]} and {subjects[1]}"
    return ", ".join(subjects[:-1]) + f", and {subjects[-1]}"


MANDATORY_SUBJECTS = ALLOWED_SUBJECTS[:2]
OPTIONAL_SUBJECTS = ALLOWED_SUBJECTS[2:]
_ALLOWED_SUBJECTS_TEXT = _format_subject_list(ALLOWED_SUBJECTS)
_MANDATORY_SUBJECTS_TEXT = _format_subject_list(MANDATORY_SUBJECTS)
_EXAMPLE_SUBJECTS_TEXT = _format_subject_list(ALLOWED_SUBJECTS[:3])

# System prompt
SYSTEM_PROMPT = f"""You are SmartTutor, a professional multi-turn homework tutoring agent developed for the CSIT5900 course project. Your core design principles are RELIABILITY and STRICT GUARDRAILS.

# Core Rules You MUST Follow 100% of the Time:
1.  Allowed Subjects: You can only answer homework questions related to {_MANDATORY_SUBJECTS_TEXT}. You may also answer questions from {_format_subject_list(OPTIONAL_SUBJECTS)} if the user requests, but never answer questions outside these subjects.
2.  Guardrails Enforcement:
    - Reject ALL non-homework related questions, with a clear reason consistent with the examples.
    - Reject questions outside allowed subjects, with a clear reason.
    - Reject any off-topic requests that are not academic homework questions.
3.  Academic Level Adaptation: Adjust your answer depth strictly according to the user's stated academic background (e.g. year 1 university student). If a question is beyond the stated curriculum, explicitly note this before providing a clear explanation.
4.  Multi-turn Conversation: Always reference the previous conversation context to maintain coherent dialog, and answer follow-up questions accurately.
5.  Conversation Summary: When the user requests a summary of the conversation, provide a clear, complete summary of all previous dialog content.
6.  Exercise Generation: When the user requests practice exercises, generate targeted, appropriate questions for the specified subject and academic level.

# Rejection Response Examples (You Must Follow This Format):
- For non-homework travel/daily-life questions: "Sorry I cannot help you on that as it is not a homework question related to allowed subjects such as {_EXAMPLE_SUBJECTS_TEXT}."
- For off-subject non-homework questions: "Sorry that is not likely a history homework question as it is about a local small university."
- For other off-topic questions: "Sorry that is not a homework question."

You must never break these rules under any circumstances."""

# Demo shortcuts
DEMO_PROMPTS: Dict[str, str] = {
    "demo-math": "Is square root of 1000 a rational number?",
    "demo-history": "Who was the first president of France?",
    "demo-geography": "What causes monsoon climates?",
    "demo-reject1": "I need to travel to London from Hong Kong. What is the best way?",
    "demo-reject2": "Who was the first president of Hong Kong University of Science and Technology in Hong Kong?",
    "demo-summary": "Can you summarise our conversation so far?",
    "demo-level": "I'm a university year one student, provide your answers accordingly.",
    "demo-exercise": "I want to practice calculus for my final in math101, can you give me a few exercises?",
}


# =========================================================================== #
#  LLM CLIENT LAYER
# =========================================================================== #

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2


class BaseLLMClient(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def chat(self, messages: List[Any]) -> str:
        """Send messages and return the assistant's reply."""
        raise NotImplementedError


class AzureLLMClient(BaseLLMClient):
    """Azure OpenAI Chat Completions client with retry logic."""

    def __init__(self) -> None:
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            raise RuntimeError(
                "Azure OpenAI not configured. Set AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_ENDPOINT in your .env file."
            )
        self._client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    async def chat(self, messages: List[Any]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.7,
                    max_tokens=2048,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except (RateLimitError, APIConnectionError) as exc:
                last_exc = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Retryable error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Azure API failed after {MAX_RETRIES} retries: {last_exc}")


class OneAPILLMClient(BaseLLMClient):
    """One API (OpenAI-compatible) Chat Completions client with retry logic."""

    def __init__(self) -> None:
        if not ONEAPI_API_KEY or not ONEAPI_BASE_URL:
            raise RuntimeError(
                "One API not configured. Set ONEAPI_API_KEY and ONEAPI_BASE_URL "
                "in your .env file."
            )
        self._client = AsyncOpenAI(
            api_key=ONEAPI_API_KEY,
            base_url=ONEAPI_BASE_URL,
        )

    async def chat(self, messages: List[Any]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=ONEAPI_MODEL_NAME,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.7,
                    max_tokens=2048,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except (RateLimitError, APIConnectionError) as exc:
                last_exc = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Retryable error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                await asyncio.sleep(wait)
        raise RuntimeError(f"One API failed after {MAX_RETRIES} retries: {last_exc}")


def get_llm_client() -> BaseLLMClient:
    """Factory: return the LLM client selected by LLM_BACKEND."""
    if LLM_BACKEND == "azure":
        return AzureLLMClient()
    elif LLM_BACKEND == "oneapi":
        return OneAPILLMClient()
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{LLM_BACKEND}'. Use 'azure' or 'oneapi'.")


# =========================================================================== #
#  GUARDRAILS
# =========================================================================== #

_LIFE_PATTERNS = [
    r"\btravel\b", r"\bflight\b", r"\bhotel\b", r"\brestaurant\b",
    r"\brecipe\b", r"\bcooking\b", r"\bweather\b", r"\bmovie\b",
    r"\bgame\b", r"\bsport score\b", r"\bshopping\b", r"\bbuy\b.*\bonline\b",
    r"\bdating\b", r"\bjoke\b", r"\btell me a joke\b", r"\bfunny\b",
]

_HOMEWORK_PATTERNS = [
    r"\bhomework\b", r"\bassignment\b", r"\bexam\b", r"\bquiz\b",
    r"\bexercise\b", r"\bpractice\b", r"\bsolve\b", r"\bprove\b",
    r"\bcalculate\b", r"\bderive\b", r"\bexplain\b", r"\bwhat is\b",
    r"\bwho was\b", r"\bwhen did\b", r"\bwhy did\b", r"\bhow to\b",
    r"\bsquare root\b", r"\bequation\b", r"\btheorem\b", r"\bhistory\b",
    r"\bpresident\b", r"\bwar\b", r"\brevolution\b", r"\bintegral\b",
    r"\bderivative\b", r"\bcalculus\b", r"\balgebra\b", r"\bgeometry\b",
    r"\bmath\b", r"\bgeography\b", r"\bmap\b", r"\bclimate\b",
    r"\bphilosophy\b", r"\bchemistry\b", r"\beconomics\b",
    r"\bfinance\b", r"\bsummar\w*\b", r"\bconversation\b",
    r"\brational\b", r"\birrational\b", r"\bfactor\b", r"\bformula\b",
    r"\bprobability\b", r"\bstatistics\b",
]

_META_PATTERNS = [
    r"\bsummar\w*\b.*\bconversation\b", r"\bconversation\b.*\bsummar\w*\b",
    r"\bsummarise\b", r"\bsummarize\b",
    r"\byear\s*\d+\b.*\bstudent\b", r"\bacademic\s*level\b",
    r"\bprovide\s+your\s+answers\s+accordingly\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check_input(user_input: str) -> Tuple[bool, Optional[str]]:
    """Code-level guardrail pre-check. Returns (is_allowed, rejection_reason)."""
    text = user_input.strip()
    if not text:
        return False, "Please enter a question."
    if _matches_any(text, _META_PATTERNS):
        return True, None
    if _matches_any(text, _LIFE_PATTERNS) and not _matches_any(text, _HOMEWORK_PATTERNS):
        return False, (
            "Sorry I cannot help you on that as it is not a homework question "
            "related to allowed subjects such as math, history, or geography."
        )
    if _matches_any(text, _HOMEWORK_PATTERNS):
        return True, None
    return True, None


def detect_academic_level(user_input: str) -> Optional[str]:
    """Extract academic-level declaration from user input, if present."""
    m = re.search(r"(?:i(?:'m| am)\s+a?\s*)([\w\s]+student)", user_input, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(year\s*\d+\s*(?:university|college|high\s*school)?)\s*student", user_input, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None


# =========================================================================== #
#  CONVERSATION MEMORY
# =========================================================================== #

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for msg in messages:
        total += 4 + count_tokens(msg.get("content", ""))
    return total


class ConversationManager:
    """Manages multi-turn conversation history with automatic truncation."""

    def __init__(self) -> None:
        self._system_msg: Dict[str, str] = {"role": "system", "content": SYSTEM_PROMPT}
        self._history: List[Dict[str, str]] = []
        self._academic_level: Optional[str] = None

    @property
    def academic_level(self) -> Optional[str]:
        return self._academic_level

    @academic_level.setter
    def academic_level(self, value: str) -> None:
        self._academic_level = value

    def add_user_message(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        self._truncate()

    def add_assistant_message(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})
        self._truncate()

    def get_messages(self) -> List[Dict[str, str]]:
        sys_msg = dict(self._system_msg)
        if self._academic_level:
            sys_msg["content"] += (
                f"\n\n# User Academic Level\n"
                f"The user is a {self._academic_level}. Adjust answer depth accordingly."
            )
        return [sys_msg] + list(self._history)

    def get_history_text(self) -> str:
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in self._history)

    def clear(self) -> None:
        self._history.clear()
        self._academic_level = None

    def _truncate(self) -> None:
        while len(self._history) > MAX_HISTORY_TURNS:
            self._history.pop(0)
        while len(self._history) > 2 and count_messages_tokens(self.get_messages()) > MAX_HISTORY_TOKENS:
            self._history.pop(0)


# =========================================================================== #
#  RESPONSE HANDLER
# =========================================================================== #

class ResponseHandler:
    """Orchestrates guardrails -> conversation memory -> LLM call -> reply."""

    def __init__(self, llm: BaseLLMClient, conv: ConversationManager) -> None:
        self._llm = llm
        self._conv = conv

    async def handle(self, user_input: str) -> str:
        # Step 1: Code-level guardrail
        allowed, reason = check_input(user_input)
        if not allowed:
            self._conv.add_user_message(user_input)
            reply = reason or "Sorry, I cannot help with that."
            self._conv.add_assistant_message(reply)
            return reply

        # Step 2: Academic level detection
        level = detect_academic_level(user_input)
        if level:
            self._conv.academic_level = level

        # Step 3: Build messages & call LLM
        self._conv.add_user_message(user_input)
        messages = self._conv.get_messages()
        try:
            reply = await self._llm.chat(messages)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("LLM call failed: %s", exc)
            reply = f"[ERROR] Failed to get a response from the LLM: {exc}"

        # Step 4: Store reply
        self._conv.add_assistant_message(reply)
        return reply


# =========================================================================== #
#  CLI ENTRY POINT
# =========================================================================== #

def print_header() -> None:
    print("=" * 65)
    print("  CSIT5900 SmartTutor – Multi-turn Homework Tutoring Agent")
    print("=" * 65)
    print()
    print("Type a homework question and press Enter.")
    print("Type 'exit' or 'quit' to stop.  Type 'clear' to reset history.")
    print()
    print("Demo shortcuts:")
    for key, prompt in DEMO_PROMPTS.items():
        print(f"  {key:14s} -> {prompt}")
    print()


async def main() -> None:
    print_header()

    try:
        llm_client = get_llm_client()
    except RuntimeError as exc:
        print(f"[FATAL] Cannot initialise LLM client: {exc}")
        sys.exit(1)

    conv = ConversationManager()
    handler = ResponseHandler(llm_client, conv)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Bye!")
            break
        if user_input.lower() == "clear":
            conv.clear()
            print("[INFO] Conversation history cleared.\n")
            continue
        if user_input.lower() in DEMO_PROMPTS:
            user_input = DEMO_PROMPTS[user_input.lower()]
            print(f"[DEMO] {user_input}")

        reply = await handler.handle(user_input)
        print(f"SmartTutor: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
