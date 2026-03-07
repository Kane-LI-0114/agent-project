"""
llm/base_client.py
==================
Abstract base class for LLM clients. Both Azure OpenAI and One API clients
inherit from this class to ensure a consistent interface throughout the
application.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List


class BaseLLMClient(ABC):
    """
    Abstract base class that every LLM backend must implement.

    Subclasses must provide:
        - chat(messages)        : one-shot call, returns complete reply string.
        - chat_stream(messages) : async generator that yields text chunks.
    """

    @abstractmethod
    async def chat(self, messages: List[Any]) -> str:
        """
        Send a conversation (list of message dicts) to the LLM and return the
        assistant's response text.

        Parameters
        ----------
        messages : list[dict]
            A list of message dicts following the OpenAI Chat Completions
            format, e.g. [{"role": "system", "content": "..."}, ...].

        Returns
        -------
        str
            The assistant's reply content.
        """
        raise NotImplementedError

    @abstractmethod
    async def chat_stream(self, messages: List[Any]) -> AsyncGenerator[str, None]:
        """
        Send a conversation to the LLM and stream back incremental text chunks.

        Yields
        ------
        str
            A content delta string from the model.
        """
        raise NotImplementedError
        yield  # type: ignore[misc]  # marks this as an async generator
