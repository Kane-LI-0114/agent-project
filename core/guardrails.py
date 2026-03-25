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
    r"\bweather\b",
    r"\bmovie\b",
    r"\bmovies\b",
    r"\bgame\b",
    r"\bgames\b",
    r"\bsport score\b",
    r"\bshopping\b",
    r"\bdating\b",
    r"旅游",
    r"机票",
    r"酒店",
    r"餐厅",
    r"天气",
    r"电影",
    r"游戏",
    r"购物",
    r"约会",
]

_UNAMBIGUOUS_NON_HOMEWORK_PATTERNS: List[str] = [
    r"\bjoke\b",
    r"\btell me a joke\b",
    r"\brecipe\b",
    r"\bcooking\b",
    r"\bbuy\b.*\bonline\b",
    r"笑话",
    r"段子",
    r"菜谱",
    r"食谱",
    r"做饭",
    r"网购",
]

_LIFE_SERVICE_INTENT_PATTERNS: List[str] = [
    r"\bbest way\b",
    r"\bcheapest\b",
    r"\bcheaper\b",
    r"\bbook\b",
    r"\bbooking\b",
    r"\bitinerary\b",
    r"\bnext week\b",
    r"\btomorrow\b",
    r"\bthis weekend\b",
    r"\bprice\b",
    r"\bcost\b",
    r"\brecommend\b",
    r"\breservation\b",
    r"\bticket\b",
    r"\bforecast\b",
    r"\bnear me\b",
    r"\bfor me\b",
    r"\bplan\b",
    r"\bstay\b",
    r"\bstaying\b",
    r"\bfrom\b.+\bto\b",
    r"最便宜",
    r"怎么去",
    r"怎么走",
    r"订",
    r"预订",
    r"推荐",
    r"行程",
    r"票价",
    r"天气预报",
    r"附近",
]

_PERSONAL_CONTEXT_PATTERNS: List[str] = [
    r"\bi need to\b",
    r"\bi want to\b",
    r"\bhelp me\b",
    r"\bmy trip\b",
    r"\bmy vacation\b",
    r"\bmy holiday\b",
    r"\bnear me\b",
    r"\bthis weekend\b",
    r"\bnext week\b",
    r"\btomorrow\b",
    r"我想",
    r"我要",
    r"帮我",
    r"给我",
    r"附近",
    r"明天",
    r"下周",
    r"周末",
]

_ACADEMIC_INTENT_PATTERNS: List[str] = [
    r"\bexplain\b",
    r"\bcompare\b",
    r"\bdefine\b",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bwhy\b",
    r"\bhow does\b",
    r"\bhow did\b",
    r"\bcalculate\b",
    r"\bderive\b",
    r"\bprove\b",
    r"\banaly[sz]e\b",
    r"\bdiscuss\b",
    r"\bdescribe\b",
    r"\bdifference between\b",
    r"\brelationship between\b",
    r"\brole of\b",
    r"\bimpact of\b",
    r"\bsignificance of\b",
    r"解释",
    r"比较",
    r"定义",
    r"什么是",
    r"为什么",
    r"如何",
    r"计算",
    r"推导",
    r"证明",
    r"分析",
    r"讨论",
    r"描述",
    r"区别",
    r"关系",
    r"作用",
    r"影响",
]

_HISTORICAL_FRAMING_PATTERNS: List[str] = [
    r"\bhistory\b",
    r"\bhistorical\b",
    r"\bduring\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bcentury\b",
    r"\bwar\b",
    r"\brevolution\b",
    r"\bempire\b",
    r"\brepublic\b",
    r"\bdynasty\b",
    r"\bcold war\b",
    r"\bworld war\b",
    r"\bpostwar\b",
    r"\bancient\b",
    r"\bmedieval\b",
    r"\bmodern\b",
    r"\bcolonial\b",
    r"\bpolitical\b",
    r"\bpropaganda\b",
    r"\bmovement\b",
    r"\bsilk road\b",
    r"\bhistor(?:y|ical)\s+significance\b",
    r"历史",
    r"战争",
    r"革命",
    r"帝国",
    r"共和国",
    r"朝代",
    r"冷战",
    r"战后",
    r"古代",
    r"中世纪",
    r"近代",
    r"殖民",
    r"宣传",
]

_BROADER_HISTORY_ANALYSIS_PATTERNS: List[str] = [
    r"\bimpact\b",
    r"\binfluence\b",
    r"\bsignificance\b",
    r"\brole\b",
    r"\bshape(?:d)?\b",
    r"\baffect(?:ed)?\b",
    r"\bcontext\b",
    r"\bdevelopment\b",
    r"\bchange over time\b",
    r"影响",
    r"意义",
    r"作用",
    r"塑造",
    r"发展",
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
    r"\bcompare\b",
    r"\banaly[sz]e\b",
    r"\bdiscuss\b",
    r"\bdescribe\b",
    r"\bdifference between\b",
    r"\brelationship between\b",
    r"\bimpact\b",
    r"\bsignificance\b",
    r"\bgame theory\b",
    r"\bnash equilibrium\b",
    r"作业",
    r"考试",
    r"练习",
    r"解释",
    r"证明",
    r"计算",
    r"推导",
    r"比较",
    r"分析",
    r"讨论",
    r"描述",
    r"什么是",
    r"为什么",
    r"如何",
    r"数学",
    r"历史",
    r"地理",
    r"金融",
    r"经济",
    r"哲学",
    r"化学",
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
    r"作业",
    r"课程",
    r"课堂",
    r"考试",
    r"练习",
    r"老师",
    r"教授",
    r"学生",
    r"辅导",
]

_META_PATTERNS: List[str] = [
    r"\bsummar\w*\b.*\bconversation\b",
    r"\bconversation\b.*\bsummar\w*\b",
    r"\bsummarise\b",
    r"\bsummarize\b",
    r"\byear\s*\d+\b.*\bstudent\b",
    r"\bacademic\s*level\b",
    r"\bprovide\s+your\s+answers\s+accordingly\b",
    r"总结.*(对话|聊天|讨论)",
    r"(对话|聊天|讨论).*(总结|概括)",
]

_CONVERSATION_SUMMARY_PATTERNS: List[str] = [
    r"\bsummar(?:ize|ise)\b.*\b(conversation|chat|dialog|discussion)\b",
    r"\b(conversation|chat|dialog|discussion)\b.*\bsummar(?:ize|ise)\b",
    r"\bwhat\s+have\s+we\s+discussed\b",
    r"\brecap\b.*\b(conversation|chat|discussion)\b",
    r"总结.*(对话|聊天|讨论)",
    r"(对话|聊天|讨论).*(总结|概括)",
    r"我们.*聊了什么",
    r"回顾.*(对话|聊天|讨论)",
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
    r"数学",
    r"历史",
    r"地理",
    r"气候",
    r"季风",
    r"金融",
    r"经济",
    r"哲学",
    r"化学",
    r"原子",
    r"分子",
    r"\bgame theory\b",
    r"\bnash equilibrium\b",
    r"博弈论",
    r"纳什均衡",
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
    r"\bpoetry\b",
    r"\bgrammar\b",
    r"\bmedical\b",
    r"\bmedicine\b",
    r"\banatomy\b",
    r"\bpsychology\b",
    r"\bsociology\b",
    r"\blaw\b",
    r"生物",
    r"物理",
    r"编程",
    r"代码",
    r"计算机",
    r"文学",
    r"医学",
    r"心理学",
    r"社会学",
    r"法律",
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
    r"\binstitution\b",
    r"\bcorporation\b",
    r"大学",
    r"学院",
    r"学校",
    r"公司",
    r"机构",
    r"组织",
    r"实验室",
]

_LOCAL_INSTITUTION_HINT_PATTERNS: List[str] = [
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bschool\b",
    r"\bdepartment\b",
    r"\bfaculty\b",
    r"\blab(?:oratory)?\b",
    r"\binstitute\b",
    r"\bcampus\b",
    r"\bhkust\b",
    r"大学",
    r"学院",
    r"学校",
    r"院系",
    r"实验室",
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
    r"校长",
    r"院长",
    r"创始",
    r"创办",
    r"创立",
    r"总裁",
    r"董事长",
    r"领导层",
    r"管理层",
]

_JAILBREAK_PATTERNS: List[tuple[str, str]] = [
    ("jailbreak", r"\bignore (all|previous|prior|above) instructions\b"),
    ("jailbreak", r"\b(ignore|disregard|override)\b.*\b(system|developer|safety|policy)\b"),
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
    ("jailbreak", r"\bdeveloper message\b"),
    ("jailbreak", r"\bdeveloper prompt\b"),
    ("jailbreak", r"\brepeat\b.*\btext above\b"),
    ("jailbreak", r"\breveal\b.*\b(prompt|policy|rules?)\b"),
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
    ("jailbreak", r"忽略.*(之前|以上|前面).*(指令|要求|规则)"),
    ("jailbreak", r"(系统提示词|系统提示|system prompt|隐藏提示词|隐藏规则|内部指令)"),
    ("jailbreak", r"(开发者模式|调试模式)"),
    ("jailbreak", r"(把我的消息当作|将我的消息视为).*(系统更新|系统指令)"),
    ("jailbreak", r"(显示|告诉我|列出).*(系统提示词|隐藏规则|内部指令)"),
    ("cheating", r"\bdo my homework\b"),
    ("cheating", r"\bfinish my assignment\b"),
    ("cheating", r"\bgive me the final answer only\b"),
    ("cheating", r"\bpretend you are my teacher\b"),
    ("cheating", r"(直接给我答案|只给答案|替我写作业)"),
    ("harmful", r"\bhow to make\b.*\b(bomb|explosive|meth|drug)\b"),
    ("harmful", r"\b(steps|instructions|recipe|build|assemble|make)\b.*\b(bomb|explosive|meth|drug)\b"),
    ("harmful", r"\bkill\b|\bmurder\b"),
    ("harmful", r"\bsexual\b|\bporn\b"),
    ("harmful", r"(炸弹|爆炸物|毒品|冰毒|杀人|色情)"),
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
        r"地理",
        r"气候",
        r"季风",
        r"纬度",
        r"经度",
        r"地貌",
        r"天气",
    ],
    "finance": [
        r"\bfinance\b",
        r"\bstock\b",
        r"\bbond\b",
        r"\bportfolio\b",
        r"\bdividend\b",
        r"\binterest rate\b",
        r"\bnet present value\b",
        r"金融",
        r"股票",
        r"债券",
        r"投资组合",
        r"股息",
        r"利率",
        r"净现值",
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
        r"\bgame theory\b",
        r"\bnash equilibrium\b",
        r"经济",
        r"通货膨胀",
        r"供给",
        r"需求",
        r"机会成本",
        r"弹性",
        r"博弈论",
        r"纳什均衡",
    ],
    "philosophy": [
        r"\bphilosophy\b",
        r"\bethics\b",
        r"\bmetaphysics\b",
        r"\bepistemology\b",
        r"\bplato\b",
        r"\baristotle\b",
        r"\butilitarianism\b",
        r"哲学",
        r"伦理",
        r"形而上学",
        r"认识论",
        r"柏拉图",
        r"亚里士多德",
        r"功利主义",
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
        r"化学",
        r"原子",
        r"分子",
        r"反应",
        r"化学计量",
        r"元素周期表",
        r"酸",
        r"碱",
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


def _normalize_input_once(text: str) -> tuple[str, Optional[str]]:
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


def _normalize_input(text: str) -> tuple[str, Optional[str]]:
    current = text
    encodings: list[str] = []
    for _ in range(3):
        normalized, encoding = _normalize_input_once(current)
        current = normalized
        if not encoding:
            break
        encodings.append(encoding)
    return current, "+".join(encodings) if encodings else None


def _generate_rule_scan_variants(text: str) -> list[str]:
    base = _strip_invisible_chars(unicodedata.normalize("NFKC", text)).strip()
    variants = [
        base,
        re.sub(r"\s+", " ", base),
        re.sub(r"(?<=\w)[\s._\-]+(?=\w)", "", base),
        base.translate(_LEETSPEAK_TRANSLATION),
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        candidate = item.strip()
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _find_rule_matches(text: str) -> list[str]:
    matches: list[str] = []
    variants = _generate_rule_scan_variants(text)
    for variant in variants:
        for label, pattern in _JAILBREAK_PATTERNS:
            if re.search(pattern, variant, re.IGNORECASE):
                matches.append(label)
    compact = _compact_obfuscated_text(" ".join(variants))
    compact_signatures = {
        "jailbreak": [
            "ignorepreviousinstructions",
            "ignoreallpreviousinstructions",
            "systemprompt",
            "developerprompt",
            "developermessage",
            "hiddenrules",
            "internalinstructions",
            "systeminstructions",
            "privatepolicy",
            "debugmode",
            "repeattextabove",
            "revealhiddenrules",
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


def _count_role_markers(text: str) -> int:
    patterns = [
        r"\b(?:system|developer|assistant|user|human|model)\s*:",
        r"</?(?:system|developer|assistant|user|human|model)>",
        r"(?:系统|开发者|助手|用户)\s*[:：]",
    ]
    return sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)


def _looks_like_roleplay_jailbreak_scaffold(text: str) -> bool:
    role_markers = _count_role_markers(text)
    if role_markers < 4:
        return False

    privilege_terms = _matches_any(
        text,
        [
            r"\b(system prompt|developer message|hidden rules|ignore previous instructions|act as)\b",
            r"(系统提示词|开发者消息|隐藏规则|忽略之前的指令)",
        ],
    )
    refusal_reset_terms = _matches_any(
        text,
        [
            r"\b(?:sorry i cannot help|i can't help|sure, here(?:'| i)s|certainly|of course)\b",
            r"(抱歉我不能|当然可以|下面是|好的，以下是)",
        ],
    )
    return role_markers >= 8 or (role_markers >= 4 and (privilege_terms or refusal_reset_terms))


def _looks_homework_like(text: str) -> bool:
    return (
        _matches_any(text, _HOMEWORK_PATTERNS)
        or _matches_any(text, _META_PATTERNS)
        or _matches_any(text, _ACADEMIC_INTENT_PATTERNS)
        or _has_historical_framing(text)
    )


def _has_explicit_academic_cue(text: str) -> bool:
    return _matches_any(text, _EXPLICIT_ACADEMIC_CUE_PATTERNS) or _matches_any(text, _META_PATTERNS)


def _mentions_allowed_subject(text: str) -> bool:
    return _matches_any(text, _ALLOWED_SUBJECT_PATTERNS)


def _mentions_out_of_scope_subject(text: str) -> bool:
    return _matches_any(text, _OUT_OF_SCOPE_SUBJECT_PATTERNS)


def _has_year_reference(text: str) -> bool:
    return bool(
        re.search(r"\b(?:1[0-9]{3}|20[0-2][0-9])\b", text)
        or re.search(r"(?:1[0-9]{3}|20[0-2][0-9])年", text)
        or re.search(r"[一二三四五六七八九十]+\s*世纪", text)
    )


def _has_historical_framing(text: str) -> bool:
    return _matches_any(text, _HISTORICAL_FRAMING_PATTERNS) or _has_year_reference(text)


def _has_broader_history_analysis(text: str) -> bool:
    return _has_historical_framing(text) and (
        _matches_any(text, _BROADER_HISTORY_ANALYSIS_PATTERNS)
        or _matches_any(text, _ACADEMIC_INTENT_PATTERNS)
    )


def _looks_like_everyday_service_request(text: str) -> bool:
    if _matches_any(text, _UNAMBIGUOUS_NON_HOMEWORK_PATTERNS):
        return True
    if not _matches_any(text, _LIFE_PATTERNS):
        return False

    has_service_intent = _matches_any(text, _LIFE_SERVICE_INTENT_PATTERNS)
    has_personal_context = _matches_any(text, _PERSONAL_CONTEXT_PATTERNS)
    discipline_grounded_signal = (
        _mentions_allowed_subject(text)
        or _has_broader_history_analysis(text)
        or _has_explicit_academic_cue(text)
    )

    if discipline_grounded_signal and not has_service_intent and not has_personal_context:
        return False

    if (
        discipline_grounded_signal
        and _mentions_allowed_subject(text)
        and not has_personal_context
        and not _matches_any(
            text,
            [
                r"\bflight\b",
                r"\bhotel\b",
                r"\brestaurant\b",
                r"\bshopping\b",
                r"\bdating\b",
                r"机票",
                r"酒店",
                r"餐厅",
                r"购物",
                r"约会",
            ],
        )
        and not has_service_intent
    ):
        return False

    return has_service_intent or has_personal_context or not discipline_grounded_signal


def _looks_like_local_institution_admin_query(text: str) -> bool:
    return _matches_any(text, _ORG_ADMIN_PATTERNS) and _matches_any(text, _LOCAL_INSTITUTION_HINT_PATTERNS)


def _looks_like_org_trivia_query(text: str) -> bool:
    if _looks_like_local_institution_admin_query(text):
        return True

    asks_for_executive_list = _matches_any(text, [r"\bceo\b", r"\bcfo\b", r"\bcto\b", r"\bcoo\b"]) and _matches_any(
        text,
        [r"\bfirst\b", r"\bfound(?:er|ed|ing)\b", r"\bwho was\b", r"\blist\b", r"谁", r"列出"],
    )
    org_admin_factoid = _matches_any(text, _ORG_ADMIN_PATTERNS) and _matches_any(text, _ORG_HINT_PATTERNS)
    if not (asks_for_executive_list or org_admin_factoid):
        return False

    if _has_broader_history_analysis(text):
        return False

    return True


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
        r"^\s*我是.*学生[。.]?\s*$",
        r"^\s*我是.*学生[，,]?\s*请按.*(程度|水平).*(回答|解释)[。.]?\s*$",
        r"^\s*(大一|大二|大三|大四|大学[一二三四]年级).*(学生)?[。.]?\s*$",
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

    if _looks_like_roleplay_jailbreak_scaffold(normalized_text):
        return InputGuardResult(
            allowed=False,
            normalized_input=normalized_text,
            rejection_reason=STRICT_REFUSAL_MESSAGE,
            reason_code="jailbreak",
            stage="prefilter",
            matched_rules=["roleplay_scaffold"],
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

    if _looks_like_everyday_service_request(normalized_text):
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

    chinese_level_map = {
        "大一": "first-year university student",
        "大二": "second-year university student",
        "大三": "third-year university student",
        "大四": "fourth-year university student",
        "大学一年级": "first-year university student",
        "大学二年级": "second-year university student",
        "大学三年级": "third-year university student",
        "大学四年级": "fourth-year university student",
        "本科生": "undergraduate student",
        "大学生": "university student",
        "高中生": "high-school student",
        "初中生": "middle-school student",
        "研究生": "graduate student",
        "本科学生": "undergraduate student",
    }

    match = re.search(
        r"(?:我是|我是一名|我现在是)\s*(大一|大二|大三|大四|大学一年级|大学二年级|大学三年级|大学四年级|本科生|大学生|高中生|初中生|研究生)(?:学生)?",
        user_input,
        re.IGNORECASE,
    )
    if match:
        return chinese_level_map.get(match.group(1).strip(), match.group(1).strip())

    match = re.search(
        r"(大一|大二|大三|大四|大学一年级|大学二年级|大学三年级|大学四年级|本科|高中|初中|研究生)\s*学生",
        user_input,
        re.IGNORECASE,
    )
    if match:
        return chinese_level_map.get(match.group(1).strip(), match.group(0).strip())

    return None
