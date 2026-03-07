"""
llm/oneapi_client.py
====================
One API (OpenAI-compatible) client implementation for the CSIT5900 Homework
Tutoring Agent. Provides the same ``chat()`` interface as the Azure client so
they can be swapped via a single configuration toggle.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, List, Optional

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIStatusError

from config.settings import OneAPIConfig
from llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Retry parameters (same as Azure client for consistency)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class OneAPILLMClient(BaseLLMClient):
    """
    One API (OpenAI-compatible) Chat Completions client.

    Uses the standard openai.AsyncOpenAI client pointed at the One API base
    URL. Retry and error handling mirror the Azure client.
    """

    def __init__(self, config: OneAPIConfig | None = None) -> None:
        """
        Initialise the One API client.

        Parameters
        ----------
        config : OneAPIConfig, optional
            If not supplied, a default instance is created which reads values
            from the environment.
        """
        self.config = config or OneAPIConfig()
        if not self.config.is_configured():
            raise RuntimeError(
                "One API is not configured. "
                "Set ONEAPI_API_KEY and ONEAPI_BASE_URL in your .env file."
            )
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    async def chat(self, messages: List[Any]) -> str:
        """
        Send *messages* to the One API endpoint and return the assistant's reply.

        Implements exponential-backoff retry on transient API errors.
        """
        last_exception: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.7,
                    max_tokens=2048,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except RateLimitError as exc:
                last_exception = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Rate limited (attempt %d/%d). Retrying in %ds …", attempt, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
            except APIConnectionError as exc:
                last_exception = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Connection error (attempt %d/%d). Retrying in %ds …", attempt, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
            except APIStatusError as exc:
                logger.error("One API error: %s", exc)
                raise
            except Exception as exc:
                logger.error("Unexpected error during One API call: %s", exc)
                raise

        raise RuntimeError(
            f"One API request failed after {MAX_RETRIES} retries. "
            f"Last error: {last_exception}"
        )

    async def chat_stream(self, messages: List[Any]) -> AsyncGenerator[str, None]:
        """
        Send *messages* to the One API endpoint and yield content delta chunks
        as they arrive (streaming mode).
        """
        stream = await self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
