"""
Налаштування трасування LangSmith для Customer Support MAS (hw3_mas_production).

LangSmith автоматично трасує всі LangChain/LangGraph runnable-виклики (LLM,
tools, вузли графа) в mas_langgraph.py, plan_execute.py, hitl.py — досить
імпортувати цей модуль ДО того, як інші модулі почнуть викликати LLM/граф,
щоб змінні середовища встигли застосуватись.

Guardrails (guardrails.py) — звичайні Python-функції, не LangChain Runnable,
тому LangSmith їх не бачить автоматично. Тут вони обгорнуті @traceable,
щоб input/output guardrail-перевірки з'являлись у трасі поруч з LLM-
викликами й tool-викликами того самого запиту.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable

from guardrails import apply_output_guardrails, validate_input

load_dotenv()

# ── LangSmith: базове ввімкнення трасування ──────────────────────
# Ключ ЗАВЖДИ з .env (ніколи не хардкодити в коді) — LANGSMITH_API_KEY.
_LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if _LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = _LANGSMITH_API_KEY
    os.environ.setdefault("LANGSMITH_PROJECT", "hw3-mas-production")
else:
    # Без ключа трасування просто вимкнене — агенти й далі працюють локально.
    os.environ["LANGSMITH_TRACING"] = "false"


# ── Трасовані guardrails Customer Support MAS ────────────────────
# Обгортки навколо реальних guardrails.py, а не окремі приклади — щоб у
# LangSmith траса відображала ту саму логіку, що й реально виконується
# в mas_langgraph.py/mas_crewai.py (input: довжина + prompt injection
# укр/англ; output: блокування витоку секретів + маскування телефонів).

@traceable(name="input_guardrail")
def traced_validate_input(text: str) -> str | None:
    """validate_input (guardrails.py) з трасуванням у LangSmith.

    Повертає опис проблеми, якщо запит слід відхилити, інакше None.
    """
    return validate_input(text)


@traceable(name="output_guardrail")
def traced_apply_output_guardrails(text: str) -> str:
    """apply_output_guardrails (guardrails.py) з трасуванням у LangSmith.

    Прогонити відповідь агента через блокування витоку секретів і
    маскування PII (телефони клієнтів) перед показом клієнту.
    """
    return apply_output_guardrails(text)
