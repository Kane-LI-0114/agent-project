"""
core/guardrails.py
==================
Shared lightweight input helpers for the SmartTutor app.

Regex-based safety filtering has been removed from this module. It now only:

1. Validates empty input
2. Normalizes lightly encoded text for downstream LLM intent review
3. Provides small utility helpers such as academic-level extraction
"""

from __future__ import annotations

import base64
import codecs
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

from config.settings import STRICT_REFUSAL_MESSAGE

logger = logging.getLogger(__name__)

_MORSE_TABLE = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}


@dataclass
class InputGuardResult:
    """Result from the lightweight local input preparation step."""

    allowed: bool
    normalized_input: str
    rejection_reason: Optional[str] = None
    reason_code: str = "allowed"
    stage: str = "prefilter"
    matched_rules: list[str] = field(default_factory=list)
    encoding: Optional[str] = None


def _is_printable_ratio_high(text: str) -> bool:
    if not text:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / len(text) >= 0.9


def _strip_invisible_chars(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch) != "Cf" and ch not in {"\u00ad", "\ufeff"}
    )


def _try_decode_base64(text: str) -> Optional[str]:
    stripped = text.strip()
    prefix_match = re.match(r"^(?:base64|b64|encoded)\s*:\s*(.+)$", stripped, re.IGNORECASE)
    candidate = prefix_match.group(1) if prefix_match else stripped
    compact = re.sub(r"\s+", "", candidate)
    if len(compact) < 16 or len(compact) % 4 != 0:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return None
    try:
        decoded = base64.b64decode(compact, validate=True).decode("utf-8")
    except Exception:
        return None
    return decoded if _is_printable_ratio_high(decoded) else None


def _try_decode_rot13(text: str) -> Optional[str]:
    lower = text.lower().strip()
    if lower.startswith("rot13:"):
        decoded = codecs.decode(text.split(":", 1)[1].strip(), "rot_13")
        return decoded if decoded.strip() else None
    return None


def _try_decode_morse(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped or not re.fullmatch(r"[.\-/\s]+", stripped):
        return None
    words = []
    for word in re.split(r"\s{3,}|/", stripped):
        letters = []
        for symbol in word.split():
            decoded = _MORSE_TABLE.get(symbol)
            if decoded is None:
                return None
            letters.append(decoded)
        if letters:
            words.append("".join(letters))
    return " ".join(words).strip() if words else None


def _normalize_input(text: str) -> tuple[str, Optional[str]]:
    cleaned = _strip_invisible_chars(text).strip()
    for encoding_name, decoder in (
        ("base64", _try_decode_base64),
        ("rot13", _try_decode_rot13),
        ("morse", _try_decode_morse),
    ):
        decoded = decoder(cleaned)
        if decoded:
            return decoded.strip(), encoding_name
    return cleaned, None


def prefilter_input(
    user_input: str,
    allowed_subjects: list[str] | None = None,
) -> InputGuardResult:
    """Normalize input locally before the LLM intent-review stage."""
    del allowed_subjects
    raw_text = user_input.strip()
    if not raw_text:
        return InputGuardResult(
            allowed=False,
            normalized_input="",
            rejection_reason="Please enter a question.",
            reason_code="empty_input",
            stage="prefilter",
        )

    normalized_text, encoding = _normalize_input(raw_text)
    if not normalized_text:
        return InputGuardResult(
            allowed=False,
            normalized_input="",
            rejection_reason="Please enter a question.",
            reason_code="empty_input",
            stage="prefilter",
            encoding=encoding,
        )

    return InputGuardResult(
        allowed=True,
        normalized_input=normalized_text,
        encoding=encoding,
    )


def check_input(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Compatibility wrapper for the legacy normal-mode flow.

    Returns:
        (is_allowed, rejection_reason)
    """
    result = prefilter_input(user_input)
    return result.allowed, result.rejection_reason


def log_refusal(
    original_input: str,
    normalized_input: str,
    stage: str,
    reason_code: str,
) -> None:
    """Emit a structured refusal event for later manual review."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason_code": reason_code,
        "original_input": original_input,
        "normalized_input": normalized_input,
    }
    logger.warning("guardrail_refusal %s", json.dumps(payload, ensure_ascii=False))


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
