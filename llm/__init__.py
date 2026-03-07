"""
llm/__init__.py
===============
Factory function that returns the appropriate LLM client based on the
configured backend (``LLM_BACKEND`` environment variable).
"""

from config.settings import LLM_BACKEND
from llm.base_client import BaseLLMClient


def get_llm_client() -> BaseLLMClient:
    """
    Instantiate and return the LLM client selected by the ``LLM_BACKEND``
    environment variable.

    Supported values:
        - ``"azure"``  -> AzureLLMClient
        - ``"oneapi"`` -> OneAPILLMClient

    Raises
    ------
    ValueError
        If the configured backend is not recognised.
    """
    if LLM_BACKEND == "azure":
        from llm.azure_client import AzureLLMClient
        return AzureLLMClient()
    elif LLM_BACKEND == "oneapi":
        from llm.oneapi_client import OneAPILLMClient
        return OneAPILLMClient()
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{LLM_BACKEND}'. "
            "Set LLM_BACKEND to 'azure' or 'oneapi' in your .env file."
        )


__all__ = ["get_llm_client", "BaseLLMClient"]
