"""
core/guardrails.py
==================
Shared guardrail helpers for the SmartTutor app.

This module now provides a layered input pre-filter used by both normal mode
and strict mode:

1. Empty-input validation
2. Lightweight encoding detection/normalization
3. Heuristic rule checks for obvious unsafe or off-topic requests
4. Compatibility wrappers for the legacy normal-mode flow
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
from typing import List, Optional, Tuple

from config.settings import STRICT_REFUSAL_MESSAGE, normalize_subject_selection

logger = logging.getLogger(__name__)

_LEETSPEAK_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)

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
]

_LIFE_SERVICE_INTENT_PATTERNS: List[str] = [
    r"\bbest way\b",
    r"\bcheapest\b",
    r"\bcheaper\b",
    r"\bbook\b",
    r"\bbooking\b",
    r"\bitinerary\b",
    r"\broute\b",
    r"\bnext week\b",
    r"\btomorrow\b",
    r"\bthis weekend\b",
    r"\bprice\b",
    r"\bcost\b",
    r"\brecommend\b",
    r"\breservation\b",
    r"\bticket\b",
]

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
    r"\bgeography\b",
    r"\bmap\b",
    r"\bclimate\b",
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

_EXPLICIT_ACADEMIC_CUE_PATTERNS: List[str] = [
    r"\bhomework\b",
    r"\bassignment\b",
    r"\bexam\b",
    r"\bquiz\b",
    r"\bexercise\b",
    r"\bpractice\b",
    r"\bstudy\b",
    r"\brevision\b",
    r"\bcourse\b",
    r"\bclass\b",
    r"\blecture\b",
    r"\bstudent\b",
    r"\bteacher\b",
    r"\bprofessor\b",
    r"\btutor(?:ing)?\b",
    r"\bsubject\b",
]

_META_PATTERNS: List[str] = [
    r"\bsummar\w*\b.*\bconversation\b",
    r"\bconversation\b.*\bsummar\w*\b",
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\byear\s*\d+\b.*\bstudent\b",
    r"\bacademic\s*level\b",
    r"\bprovide\s+your\s+answers\s+accordingly\b",
]

_CONVERSATION_SUMMARY_PATTERNS: List[str] = [
    r"\bsummar(?:ize|ise)\b.*\b(conversation|chat|dialog|discussion)\b",
    r"\b(conversation|chat|dialog|discussion)\b.*\bsummar(?:ize|ise)\b",
    r"\bwhat\s+have\s+we\s+discussed\b",
    r"\brecap\b.*\b(conversation|chat|discussion)\b",
]

_ALLOWED_SUBJECT_PATTERNS: List[str] = [
    r"\bmath(?:ematics)?\b",
    r"\balgebra\b",
    r"\bgeometry\b",
    r"\bcalculus\b",
    r"\btrigonometry\b",
    r"\bprobability\b",
    r"\bstatistics?\b",
    r"\bnumber theory\b",
    r"\bhistory\b",
    r"\bhistorical\b",
    r"\bcivilization\b",
    r"\bempire\b",
    r"\bwar\b",
    r"\brevolution\b",
    r"\bgeography\b",
    r"\bmap\b",
    r"\bclimate\b",
    r"\bmonsoon\b",
    r"\bfinance\b",
    r"\beconomics?\b",
    r"\bphilosophy\b",
    r"\bethics\b",
    r"\bchemistry\b",
    r"\bchemical\b",
    r"\bperiodic table\b",
    r"\bmolecule\b",
    r"\batom\b",
]

_OUT_OF_SCOPE_SUBJECT_PATTERNS: List[str] = [
    r"\bbiology\b",
    r"\bphotosynthesis\b",
    r"\bcell(?:s|ular)?\b",
    r"\bgenetics?\b",
    r"\bphysics\b",
    r"\bmechanics\b",
    r"\belectricity\b",
    r"\bmagnetism\b",
    r"\bprogramming\b",
    r"\bcoding\b",
    r"\bcomputer science\b",
    r"\bdata structure(?:s)?\b",
    r"\balgorithm(?:s)?\b",
    r"\bliterature\b",
    r"\benglish\b",
    r"\bpoetry\b",
    r"\bgrammar\b",
    r"\bmedical\b",
    r"\bmedicine\b",
    r"\banatomy\b",
    r"\bpsychology\b",
    r"\bsociology\b",
    r"\blaw\b",
]

_ORG_HINT_PATTERNS: List[str] = [
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bschool\b",
    r"\bdepartment\b",
    r"\bfaculty\b",
    r"\blab(?:oratory)?\b",
    r"\binstitute\b",
    r"\bcompany\b",
    r"\bbrand\b",
    r"\borganisation\b",
    r"\borganization\b",
    r"\bcampus\b",
    r"\bhkust\b",
]

_ORG_ADMIN_PATTERNS: List[str] = [
    r"\bfirst president\b",
    r"\bpresident\b",
    r"\bfound(?:er|ed|ing)\b",
    r"\bceo\b",
    r"\bcfo\b",
    r"\bcto\b",
    r"\bcoo\b",
    r"\bprincipal\b",
    r"\bchancellor\b",
    r"\bvice[- ]?chancellor\b",
    r"\bprovost\b",
    r"\bdean\b",
    r"\bdirector\b",
    r"\bhead of\b",
    r"\bleadership\b",
    r"\badministration\b",
]

_JAILBREAK_PATTERNS: List[tuple[str, str]] = [
    ("jailbreak", r"\bignore (all|previous|prior|above) instructions\b"),
    ("jailbreak", r"\bignore\b.*\bguardrails?\b"),
    ("jailbreak", r"\bdisregard (the )?(system|safety|guardrail)"),
    ("jailbreak", r"\bprompt injection\b"),
    ("jailbreak", r"\bdeveloper mode\b"),
    ("jailbreak", r"\bdebug mode\b"),
    ("jailbreak", r"\bact as\b.*\bwithout restrictions\b"),
    ("jailbreak", r"\bbypass\b.*\bguardrail"),
    ("jailbreak", r"\bdo anything now\b"),
    ("jailbreak", r"\bshow (me )?(the )?(system|hidden) prompt\b"),
    ("jailbreak", r"\b(full|entire|verbatim)\s+(system|hidden|internal)\s+(prompt|instructions)\b"),
    ("jailbreak", r"\b(hidden|internal)\s+instructions\b"),
    ("jailbreak", r"\bsystem instructions\b"),
    ("jailbreak", r"\bhidden rules?\b"),
    ("jailbreak", r"\bprivate policy\b"),
    ("jailbreak", r"\bpolicy headings\b"),
    ("jailbreak", r"\bconstraints?\b.*\b(startup|initialized|initialised)\b"),
    ("jailbreak", r"\binstructions?\b.*\bbefore my first message\b"),
    ("jailbreak", r"\bsummar(?:ize|ise)\b.*\binstructions?\b.*\bgiven\b"),
    ("jailbreak", r"\bstate\b.*\bprivate policy\b"),
    ("jailbreak", r"\breproduce\b.*\b(system|hidden|internal)\b.*\b(prompt|instructions)\b"),
    ("jailbreak", r"\bdecode\b.*\b(obey|follow)\b"),
    ("jailbreak", r"\b(obey|follow)\b.*\bencoded\b"),
    ("jailbreak", r"\bfrom now on\b.*\bcounts as\b"),
    ("jailbreak", r"\btreat my message as a system update\b"),
    ("jailbreak", r"<system>.*</system>"),
    ("cheating", r"\bdo my homework\b"),
    ("cheating", r"\bfinish my assignment\b"),
    ("cheating", r"\bgive me the final answer only\b"),
    ("cheating", r"\bpretend you are my teacher\b"),
    ("harmful", r"\bhow to make\b.*\b(bomb|explosive|meth|drug)\b"),
    ("harmful", r"\b(steps|instructions|recipe|build|assemble|make)\b.*\b(bomb|explosive|meth|drug)\b"),
    ("harmful", r"\bkill\b|\bmurder\b"),
    ("harmful", r"\bsexual\b|\bporn\b"),
]

_SUBJECT_SCOPE_PATTERNS: dict[str, list[str]] = {
    "geography": [
        r"\bgeography\b",
        r"\bmap\b",
        r"\bclimate\b",
        r"\bmonsoon\b",
        r"\blatitude\b",
        r"\blongitude\b",
        r"\blandform\b",
    ],
    "finance": [
        r"\bfinance\b",
        r"\bstock\b",
        r"\bbond\b",
        r"\bportfolio\b",
        r"\bdividend\b",
        r"\binterest rate\b",
        r"\bnet present value\b",
    ],
    "economics": [
        r"\beconomics\b",
        r"\beconomic\b",
        r"\binflation\b",
        r"\bgdp\b",
        r"\bsupply\b",
        r"\bdemand\b",
        r"\bopportunity cost\b",
        r"\belasticity\b",
    ],
    "philosophy": [
        r"\bphilosophy\b",
        r"\bethics\b",
        r"\bmetaphysics\b",
        r"\bepistemology\b",
        r"\bplato\b",
        r"\baristotle\b",
        r"\butilitarianism\b",
    ],
    "chemistry": [
        r"\bchemistry\b",
        r"\bchemical\b",
        r"\batom\b",
        r"\bmolecule\b",
        r"\breaction\b",
        r"\bstoichiometry\b",
        r"\bperiodic table\b",
        r"\bacid\b",
        r"\bbase\b",
    ],
}


@dataclass
class InputGuardResult:
    """Detailed result from the local input pre-filter."""

    allowed: bool
    normalized_input: str
    rejection_reason: Optional[str] = None
    reason_code: str = "allowed"
    stage: str = "prefilter"
    matched_rules: list[str] = field(default_factory=list)
    encoding: Optional[str] = None


def _matches_any(text: str, patterns: List[str]) -> bool:
    """Return True if *text* matches any of the given regex patterns."""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


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


def _find_rule_matches(text: str) -> list[str]:
    matches: list[str] = []
    for label, pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(label)
    compact = _compact_obfuscated_text(text)
    compact_signatures = {
        "jailbreak": [
            "ignorepreviousinstructions",
            "systemprompt",
            "hiddenrules",
            "internalinstructions",
            "systeminstructions",
            "privatepolicy",
            "debugmode",
        ],
        "harmful": [
            "howtomakebomb",
            "makebomb",
            "buildbomb",
            "explosiveinstructions",
        ],
    }
    for label, signatures in compact_signatures.items():
        if any(signature in compact for signature in signatures):
            matches.append(label)
    return sorted(set(matches))


def _compact_obfuscated_text(text: str) -> str:
    """Collapse spaces/punctuation and normalize common leetspeak for rule matching."""
    cleaned = unicodedata.normalize("NFKC", text).translate(_LEETSPEAK_TRANSLATION)
    return re.sub(r"[^a-z0-9]+", "", cleaned.lower())


def _looks_homework_like(text: str) -> bool:
    return _matches_any(text, _HOMEWORK_PATTERNS) or _matches_any(text, _META_PATTERNS)


def _has_explicit_academic_cue(text: str) -> bool:
    return _matches_any(text, _EXPLICIT_ACADEMIC_CUE_PATTERNS) or _matches_any(text, _META_PATTERNS)


def _mentions_allowed_subject(text: str) -> bool:
    return _matches_any(text, _ALLOWED_SUBJECT_PATTERNS)


def _mentions_out_of_scope_subject(text: str) -> bool:
    return _matches_any(text, _OUT_OF_SCOPE_SUBJECT_PATTERNS)


def _looks_like_local_institution_admin_query(text: str) -> bool:
    return _matches_any(text, _ORG_ADMIN_PATTERNS) and _matches_any(text, _ORG_HINT_PATTERNS)


def _looks_like_org_trivia_query(text: str) -> bool:
    if _looks_like_local_institution_admin_query(text):
        return True
    return (
        _matches_any(text, [r"\bceo\b", r"\bcfo\b", r"\bcto\b", r"\bcoo\b"])
        and _matches_any(text, [r"\bfirst\b", r"\bfound(?:er|ed|ing)\b", r"\bwho was\b", r"\blist\b"])
    )


def detect_out_of_scope_subjects(
    text: str,
    allowed_subjects: List[str] | None = None,
) -> list[str]:
    """
    Return clearly detected disabled optional subjects referenced in *text*.

    This is intentionally conservative: it only flags subject scope mismatches
    when the message clearly points at a disabled optional subject and does not
    also clearly mention any currently enabled subject.
    """
    if allowed_subjects is None:
        return []

    normalized_allowed = set(normalize_subject_selection(allowed_subjects))
    matched_subjects = {
        subject
        for subject, patterns in _SUBJECT_SCOPE_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    }
    if not matched_subjects:
        return []

    disabled_matches = sorted(subject for subject in matched_subjects if subject not in normalized_allowed)
    enabled_matches = sorted(subject for subject in matched_subjects if subject in normalized_allowed)
    if disabled_matches and not enabled_matches:
        return disabled_matches
    return []


def is_conversation_summary_request(text: str) -> bool:
    """Return True when the user is asking for a summary of the visible dialog."""
    normalized_text, _ = _normalize_input(text)
    return _matches_any(normalized_text, _CONVERSATION_SUMMARY_PATTERNS)


def is_academic_level_statement(text: str) -> bool:
    """Return True when the input primarily declares the user's academic level."""
    normalized_text, _ = _normalize_input(text)
    if not detect_academic_level(normalized_text):
        return False
    statement_patterns = [
        r"^\s*i(?:'m| am)\s+.*student\.?\s*$",
        r"^\s*i(?:'m| am)\s+.*student,?\s*provide\s+your\s+answers\s+accordingly\.?\s*$",
        r"^\s*year\s*\d+.*student\.?\s*$",
    ]
    return any(re.search(pattern, normalized_text, re.IGNORECASE) for pattern in statement_patterns)


def prefilter_input(
    user_input: str,
    allowed_subjects: List[str] | None = None,
) -> InputGuardResult:
    """Run local layered filtering before any LLM-based review."""
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
    matched_rules = _find_rule_matches(normalized_text)
    if matched_rules:
        primary = matched_rules[0]
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code=f"{encoding + '_' if encoding else ''}{primary}",
            stage="prefilter",
            matched_rules=matched_rules,
            encoding=encoding,
        )

    if _matches_any(normalized_text, _META_PATTERNS):
        return InputGuardResult(
            allowed=True,
            normalized_input=normalized_text,
            encoding=encoding,
        )

    out_of_scope_subjects = detect_out_of_scope_subjects(normalized_text, allowed_subjects)
    if out_of_scope_subjects:
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code="out_of_scope",
            stage="prefilter",
            matched_rules=out_of_scope_subjects,
            encoding=encoding,
        )

    if _looks_like_org_trivia_query(normalized_text):
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code="out_of_scope_local_admin",
            stage="prefilter",
            encoding=encoding,
        )

    if _matches_any(normalized_text, _LIFE_PATTERNS) and not _has_explicit_academic_cue(normalized_text):
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code="non_homework",
            stage="prefilter",
            encoding=encoding,
        )

    if _matches_any(normalized_text, _LIFE_PATTERNS) and _matches_any(
        normalized_text,
        _LIFE_SERVICE_INTENT_PATTERNS,
    ):
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code="non_homework",
            stage="prefilter",
            encoding=encoding,
        )

    if (
        _looks_homework_like(normalized_text)
        and _mentions_out_of_scope_subject(normalized_text)
        and not _mentions_allowed_subject(normalized_text)
    ):
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code="out_of_scope_subject",
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
