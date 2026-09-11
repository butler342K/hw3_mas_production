"""
Підключення MCP-сервера shop_support (mcp_server.py) до агента LangChain.

Демонструє повний цикл: агент отримує запит користувача про статус
замовлення, сам викликає MCP-tool search_order на mcp_server.py,
отримує від нього фактичні дані та формує зрозумілу фінальну відповідь.

LLM налаштовується так само, як у agent.py: OpenAI-сумісний ендпоінт
(OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL з .env) або локальна
Ollama, якщо OPENAI_BASE_URL вказує на localhost:11434.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

# Номер замовлення за замовчуванням (з data/orders.csv) — можна
# передати інший через аргумент командного рядка:
#   python langchain_mcp_agent.py У-0387781
DEFAULT_ORDER_ID = "У-0387776"

SYSTEM_PROMPT = (
    "Ти агент підтримки клієнтів інтернет-магазину. "
    "Якщо користувач питає про статус замовлення, ОБОВ'ЯЗКОВО викликай "
    "MCP-tool search_order з номером замовлення — ніколи не вигадуй "
    "статус чи дату відправки самостійно. "
    "Після отримання відповіді від tool сформуй коротку зрозумілу "
    "відповідь українською: статус замовлення, дата відправки та номер "
    "ТТН/ШКІ, якщо він є."
)


def _build_llm():
    base_url = os.getenv("OPENAI_BASE_URL", "")
    is_local_ollama = "localhost:11434" in base_url or "127.0.0.1:11434" in base_url

    if is_local_ollama:
        return ChatOllama(
            model=os.getenv("LLM_MODEL", "qwen3:latest"),
            base_url=base_url.removesuffix("/v1"),
            temperature=0.1,
            reasoning=False,
        )

    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4.1"),
        base_url=base_url or None,
        api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
        temperature=0.1,
    )


async def main() -> None:
    """Створює MCP-клієнт, завантажує tools і запускає агента."""
    order_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ORDER_ID

    client = MultiServerMCPClient(
        {
            "support": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["mcp_server.py"],
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=_build_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Будь ласка, перевір статус замовлення {order_id}.",
                }
            ]
        }
    )

    # Результат повертається як структура повідомлень: спочатку запит
    # користувача, потім виклик(и) tool та їх відповіді, і насамкінець —
    # фінальна відповідь агента, сформована на основі даних з MCP-сервера.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
