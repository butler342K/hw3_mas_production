"""
ADK-агент, який використовує tools з локального MCP-сервера.

Запуск: `adk web` з кореня проєкту (hw3_mas_production) — ADK підхопить
цю теку як окремий застосунок "adk_agent".
"""

import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# mcp_server.py лежить у корені проєкту
# cwd задаємо явно, бо ADK може запускати процес з будь-якого каталогу.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# root_agent — стандартна точка входу для ADK-проєкту
root_agent = LlmAgent(
    model="gemini-3.6-flash",
    name="support_agent",
    instruction=(
        "Ти агент підтримки клієнтів інтернет-магазину. "
        "Використовуй доступні MCP-tools для пошуку замовлень, "
        "перевірки дати відправки, відстеження посилок Нової Пошти й "
        "Укрпошти та пошуку відповідей у базі знань (доставка, оплата, "
        "повернення, гарантія). "
        "Відповідай українською мовою, коротко і професійно."
    ),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=["mcp_server.py"],
                    cwd=str(_PROJECT_ROOT),
                ),
                timeout=30.0,
            ),
        )
    ],
)
