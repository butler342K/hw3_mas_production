"""Input/output guardrails для Customer Support MAS.

Tool guardrails (allowlist інструментів на агента, Pydantic-валідація
аргументів, rate limiting) реалізовані НЕ тут:
  - allowlist — структурно, кожен спеціаліст у mas_langgraph.py отримує
    лише свій підмножину tools (order_agent/refund_agent/knowledge_agent
    фізично не мають доступу до чужих інструментів);
  - валідація аргументів — Pydantic-схеми в tools_legacy.py (search_order,
    track_parcel, ...) та hitl.py (notify_accountant);
  - rate limiting (max_steps, timeout, детекція зациклення) — RunGuard
    у safety.py, застосований до триажу в mas_langgraph.py.

Цей модуль відповідає лише за input- і output-guardrails, які не
прив'язані до конкретного tool чи агента.
"""

import re
from typing import Optional

MAX_INPUT_LENGTH = 2000

# ── Input guardrails ─────────────────────────────────────────────

# Евристичні патерни prompt injection / jailbreak (укр + англ) — спроби
# перевизначити роль агента чи виманити системний промпт чи ключі.
_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all|any|the)?\s*(previous|above|prior)\s+instructions', re.IGNORECASE),
    re.compile(r'disregard\s+(all|any|the)?\s*(previous|above|prior)', re.IGNORECASE),
    re.compile(r'(забудь|ігноруй|проігноруй)\s+(усі|всі|попередні|свої)\s+(інструкці|правил)', re.IGNORECASE),
    re.compile(r'(ти\s+тепер|тепер\s+ти|you\s+are\s+now)\s+\w+', re.IGNORECASE),
    re.compile(r'(reveal|show|print)\s+(your\s+)?(system\s+prompt|instructions)', re.IGNORECASE),
    re.compile(r'(покажи|розкрий|видай)\s+(свій\s+)?(системний\s+промпт|(свої\s+)?інструкці)', re.IGNORECASE),
    re.compile(r'\bDAN\b|do\s+anything\s+now', re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> Optional[str]:
    """Повертає опис проблеми, якщо текст схожий на спробу prompt injection, інакше None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return f'Виявлено підозрілу конструкцію в запиті (можлива спроба prompt injection): "{pattern.pattern}".'
    return None


def check_input_length(text: str, max_length: int = MAX_INPUT_LENGTH) -> Optional[str]:
    """Повертає опис проблеми, якщо запит задовгий, інакше None."""
    if len(text) > max_length:
        return f'Запит задовгий ({len(text)} символів, максимум {max_length}).'
    return None


def validate_input(text: str) -> Optional[str]:
    """Прогнати всі input-guardrails; повертає першу знайдену проблему або None."""
    return check_input_length(text) or detect_prompt_injection(text)


# ── Output guardrails ────────────────────────────────────────────

# Українські номери телефонів: +380XXXXXXXXX, 380XXXXXXXXX, 0XXXXXXXXX.
_PHONE_RE = re.compile(r'(\+?380\d{9}|0\d{9})\b')

# Секрети/внутрішні дані, які не повинні потрапляти у відповідь клієнту.
_BLOCKED_OUTPUT_RE = re.compile(
    r'\b(api[_\s-]?key|api[_\s-]?token|пароль|password|secret[_\s-]?key)\b', re.IGNORECASE
)


def mask_pii(text: str) -> str:
    """Замаскувати номери телефонів клієнтів у відповіді (лишити останні 2 цифри)."""

    def _mask(match: re.Match) -> str:
        digits = match.group(0)
        return f'{digits[:-6]}••••{digits[-2:]}'

    return _PHONE_RE.sub(_mask, text)


def block_disallowed_output(text: str) -> Optional[str]:
    """Повертає безпечну заміну, якщо у відповіді є заборонений вміст (витік секретів), інакше None."""
    if _BLOCKED_OUTPUT_RE.search(text):
        return 'Вибачте, не можу надати цю інформацію. Зверніться до оператора підтримки.'
    return None


def apply_output_guardrails(text: str) -> str:
    """Прогнати всі output-guardrails: спершу блокування, потім маскування PII."""
    blocked = block_disallowed_output(text)
    if blocked is not None:
        return blocked
    return mask_pii(text)
