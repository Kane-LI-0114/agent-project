"""
llm/azure_client.py
====================
Azure OpenAI API client implementation for the CSIT5900 Homework Tutoring Agent.
Uses the openai Python SDK v1.x with AsyncAzureOpenAI for asynchronous chat
completions. Includes retry logic and comprehensive exception handling.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, List, Optional

from openai import AsyncAzureOpenAI, APIConnectionError, RateLimitError, APIStatusError

from config.settings import AzureOpenAIConfig
from llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Retry parameters
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class AzureLLMClient(BaseLLMClient):
    """
    Azure OpenAI Chat Completions client.

    Wraps AsyncAzureOpenAI to provide a simple ``chat()`` interface with
    automatic retries on transient errors (rate-limit, connection failures).
    """

    def __init__(self, config: AzureOpenAIConfig | None = None) -> None:
        """
        Initialise the Azure client.

        Parameters
        ----------
        config : AzureOpenAIConfig, optional
            If not supplied, a default instance is created which reads values
            from the environment.
        """
        self.config = config or AzureOpenAIConfig()
        if not self.config.is_configured():
            raise RuntimeError(
                "Azure OpenAI is not configured. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in your .env file."
            )
        self._client = AsyncAzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.endpoint,
            api_version=self.config.api_version,
        )

    async def chat(self, messages: List[Any]) -> str:
        """
        Send *messages* to Azure OpenAI and return the assistant's reply.

        Implements exponential-backoff retry on transient API errors.
        """
        last_exception: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.deployment_name,
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
                # Non-retryable status errors (4xx other than 429)
                logger.error("Azure API error: %s", exc)
                raise
            except Exception as exc:
                logger.error("Unexpected error during Azure API call: %s", exc)
                raise

        # All retries exhausted
        raise RuntimeError(
            f"Azure OpenAI request failed after {MAX_RETRIES} retries. "
            f"Last error: {last_exception}"
        )

    async def chat_stream(self, messages: List[Any]) -> AsyncGenerator[str, None]:
        """
        Send *messages* to Azure OpenAI and yield content delta chunks as they
        arrive (streaming mode).  No retry is applied because partial output
        cannot be replayed cleanly.
        """
        stream = await self._client.chat.completions.create(
            model=self.config.deployment_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.7,
            max_tokens=2048,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
