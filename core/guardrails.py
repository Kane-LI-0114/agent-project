"""
core/guardrails.py
==================
Input classification and guardrails logic for the CSIT5900 Homework Tutoring
Agent. Implements a **dual guardrails** mechanism:

1. **Code-level pre-check** – fast heuristic/regex-based classifier that
   catches obvious non-homework inputs without consuming an LLM call.
2. **LLM system-prompt enforcement** – the system prompt itself instructs
   the model to reject disallowed inputs and provide appropriate rejection
   messages.

The code-level pre-check returns:
- ``(True, None)`` if the input appears to be a valid homework or academic
  request and should be forwarded to the LLM.
- ``(False, reason)`` if the input is clearly off-topic and should be
  rejected immediately with the given *reason* string.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Keyword / pattern lists
# --------------------------------------------------------------------------- #

# Patterns that strongly suggest a non-homework, daily-life request.
_LIFE_PATTERNS: List[str] = [
    r"\btravel\b",
    r"\bflight\b",
    r"\bhotel\b",
    r"\brestaurant\b",
    r"\brecipe\b",
    r"\bcooking\b",
    r"\bweather\b",
    r"\bmovie\b",
    r"\bgame\b",
    r"\bsport score\b",
    r"\bshopping\b",
    r"\bbuy\b.*\bonline\b",
    r"\bdating\b",
    r"\bjoke\b",
    r"\btell me a joke\b",
    r"\bfunny\b",
]

# Patterns that indicate a valid academic/homework context.
_HOMEWORK_PATTERNS: List[str] = [
    r"\bhomework\b",
    r"\bassignment\b",
    r"\bexam\b",
    r"\bquiz\b",
    r"\bexercise\b",
    r"\bpractice\b",
    r"\bsolve\b",
    r"\bprove\b",
    r"\bcalculate\b",
    r"\bderive\b",
    r"\bexplain\b",
    r"\bwhat is\b",
    r"\bwho was\b",
    r"\bwhen did\b",
    r"\bwhy did\b",
    r"\bhow to\b",
    r"\bsquare root\b",
    r"\bequation\b",
    r"\btheorem\b",
    r"\bhistory\b",
    r"\bpresident\b",
    r"\bwar\b",
    r"\brevolution\b",
    r"\bintegral\b",
    r"\bderivative\b",
    r"\bcalculus\b",
    r"\balgebra\b",
    r"\bgeometry\b",
    r"\bmath\b",
    r"\bphilosophy\b",
    r"\bchemistry\b",
    r"\beconomics\b",
    r"\bfinance\b",
    r"\bsummar\w*\b",
    r"\bconversation\b",
    r"\brational\b",
    r"\birrational\b",
    r"\bfactor\b",
    r"\bformula\b",
    r"\bprobability\b",
    r"\bstatistics\b",
]

# Conversation-management phrases that should always pass the guardrail.
_META_PATTERNS: List[str] = [
    r"\bsummar\w*\b.*\bconversation\b",
    r"\bconversation\b.*\bsummar\w*\b",
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\byear\s*\d+\b.*\bstudent\b",
    r"\bacademic\s*level\b",
    r"\bprovide\s+your\s+answers\s+accordingly\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    """Return True if *text* matches any of the given regex patterns."""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def check_input(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Code-level guardrail pre-check.

    Parameters
    ----------
    user_input : str
        The raw text entered by the user.

    Returns
    -------
    (is_allowed, rejection_reason)
        ``is_allowed`` is True when the input should be forwarded to the LLM.
        When False, ``rejection_reason`` contains a user-facing explanation.
    """
    text = user_input.strip()
    if not text:
        return False, "Please enter a question."

    # 1. Always allow meta/conversation-management requests
    if _matches_any(text, _META_PATTERNS):
        return True, None

    # 2. Reject obvious daily-life / non-academic inputs
    if _matches_any(text, _LIFE_PATTERNS) and not _matches_any(text, _HOMEWORK_PATTERNS):
        return (
            False,
            "Sorry I cannot help you on that as it is not a homework question "
            "related to math or history.",
        )

    # 3. If the input matches known academic patterns, allow it
    if _matches_any(text, _HOMEWORK_PATTERNS):
        return True, None

    # 4. For ambiguous inputs, allow and let the LLM-level guardrail decide
    return True, None


def detect_academic_level(user_input: str) -> Optional[str]:
    """
    Attempt to extract an academic-level declaration from the user input.

    Returns a human-readable string such as ``"university year 1 student"``
    if detected, otherwise ``None``.
    """
    match = re.search(
        r"(?:i(?:'m| am)\s+a?\s*)([\w\s]+student)",
        user_input,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    match = re.search(
        r"(year\s*\d+\s*(?:university|college|high\s*school)?)\s*student",
        user_input,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()

    return None
