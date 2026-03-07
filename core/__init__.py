"""
core/__init__.py
================
Expose core components for convenient imports.
"""

from core.conversation import ConversationManager
from core.guardrails import check_input, detect_academic_level
from core.response_handler import ResponseHandler

__all__ = [
    "ConversationManager",
    "ResponseHandler",
    "check_input",
    "detect_academic_level",
]
