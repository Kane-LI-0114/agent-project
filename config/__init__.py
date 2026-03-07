"""
config/__init__.py
==================
Expose key configuration objects for convenient imports.
"""

from config.settings import (
    ALLOWED_SUBJECTS,
    DEMO_PROMPTS,
    AzureOpenAIConfig,
    LLM_BACKEND,
    MAX_HISTORY_TOKENS,
    MAX_HISTORY_TURNS,
    MANDATORY_SUBJECTS,
    OneAPIConfig,
    SYSTEM_PROMPT,
)

__all__ = [
    "ALLOWED_SUBJECTS",
    "AzureOpenAIConfig",
    "DEMO_PROMPTS",
    "LLM_BACKEND",
    "MANDATORY_SUBJECTS",
    "MAX_HISTORY_TOKENS",
    "MAX_HISTORY_TURNS",
    "OneAPIConfig",
    "SYSTEM_PROMPT",
]
