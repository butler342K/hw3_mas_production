"""
MCP-сервер для Customer Support MAS (підтримка інтернет-магазину).

Призначення:
    Обгортає в протокол MCP ті самі tools, якими користується
    LangGraph-агент (agent.py): пошук замовлень у data/orders.csv,
    відстеження посилок Нової Пошти й Укрпошти та пошук по базі знань
    (доставка, оплата, повернення, гарантія). Логіку не дублюємо —
    імпортуємо готові LangChain-tools із tools_legacy.py і knowledge.py
    та викликаємо їх через .invoke(), щоб MCP-сервер і агент завжди
    працювали з одним і тим самим кодом та даними.

Навіщо окремий MCP-сервер поруч з LangGraph-агентом:
    Дає змогу користуватись тими самими інструментами з будь-якого
    MCP-клієнта (MCP Inspector, Claude Desktop тощо) напряму, без
    запуску всього агентського графа — зручно для ручної перевірки
    даних і для дебагу окремого tool.

Запуск:
    python mcp_server.py

Тестування через Inspector:
    npx @modelcontextprotocol/inspector python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

from knowledge import knowledge_base
from knowledge import search_knowledge as _search_knowledge
from tools_legacy import _load_orders
from tools_legacy import get_order_ship_date as _get_order_ship_date
from tools_legacy import search_order as _search_order
from tools_legacy import track_parcel as _track_parcel
from tools_legacy import track_ukrposhta_parcel as _track_ukrposhta_parcel

# ── Ініціалізація MCP-сервера ────────────────────────────────────

mcp = FastMCP(
    name="shop_support",
    instructions=(
        "Сервер підтримки клієнтів інтернет-магазину. "
        "Надає пошук замовлень (номер або телефон клієнта), відстеження "
        "посилок Нової Пошти й Укрпошти та пошук довідкової інформації "
        "про доставку, оплату, повернення й гарантію."
    ),
)

# ── MCP tools ────────────────────────────────────────────────────
# Кожен tool — тонка обгортка над відповідним LangChain-tool із
# tools_legacy.py / knowledge.py: та сама Pydantic-валідація вхідних
# параметрів, той самий доступ до CSV і зовнішніх API.


@mcp.tool()
def search_order(query: str) -> str:
    """
    Знайти замовлення в базі за номером або телефоном клієнта.

    Args:
        query: Номер замовлення (напр. "У-0387776", можна частково)
            або телефон клієнта.

    Returns:
        Статус, дата відправки і ТТН/ШКІ знайдених замовлень.
    """
    return _search_order.invoke({"query": query})


@mcp.tool()
def get_order_ship_date(query: str) -> str:
    """
    Отримати дату відправки замовлення за номером або телефоном клієнта.

    Args:
        query: Номер замовлення (повністю або частково) або телефон клієнта.

    Returns:
        Дата відправки знайдених замовлень.
    """
    return _get_order_ship_date.invoke({"query": query})


@mcp.tool()
def track_parcel(ttn: str) -> str:
    """
    Відстежити посилку Нової Пошти за номером ТТН через офіційне API.

    Args:
        ttn: Номер ТТН (щонайменше 12 цифр).

    Returns:
        Поточний статус посилки, маршрут і дати доставки.
    """
    return _track_parcel.invoke({"ttn": ttn})


@mcp.tool()
def track_ukrposhta_parcel(barcode: str) -> str:
    """
    Відстежити відправлення Укрпошти за ШКІ через офіційне API.

    Args:
        barcode: Номер поштового відправлення (ШКІ), щонайменше 8 символів.

    Returns:
        Поточний статус відправлення.
    """
    return _track_ukrposhta_parcel.invoke({"barcode": barcode})


@mcp.tool()
def search_knowledge(query: str) -> str:
    """
    Пошук довідкової інформації в базі знань (доставка, оплата,
    повернення, гарантія, митні процедури тощо).

    Args:
        query: Текстовий пошуковий запит.

    Returns:
        Топ-3 релевантних документи з бази знань.
    """
    return _search_knowledge.invoke({"query": query})


# ── MCP resource ─────────────────────────────────────────────────


@mcp.resource("support://info")
def support_info() -> str:
    """Загальна інформація про сервіс підтримки: обсяг даних у базах."""
    return (
        f"Shop Support System\n"
        f"Orders in DB: {len(_load_orders())}\n"
        f"Knowledge base documents: {knowledge_base.count()}"
    )


# ── Точка входу ──────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
