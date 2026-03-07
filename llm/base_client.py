"""
llm/base_client.py
==================
Abstract base class for LLM clients. Both Azure OpenAI and One API clients
inherit from this class to ensure a consistent interface throughout the
application.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class BaseLLMClient(ABC):
    """
    Abstract base class that every LLM backend must implement.

    Subclasses must provide:
        - chat(messages): Send a list of chat messages and return the
          assistant's reply as a plain string.
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
