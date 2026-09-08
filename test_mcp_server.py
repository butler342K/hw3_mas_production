"""
Unit tests для бізнес-логіки MCP-сервера (shop_support).

Покриває ті самі tools, які MCP-сервер (mcp_server.py) обгортає з
tools_legacy.py / knowledge.py: пошук замовлення, дату відправки,
відстеження посилок Нової Пошти/Укрпошти (з мокнутим API) та пошук
у базі знань.

Запуск:
    pytest test_mcp_server.py -v
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

import tools_legacy
from mcp_server import (
    get_order_ship_date,
    search_knowledge,
    search_order,
    track_parcel,
    track_ukrposhta_parcel,
)
from tools_legacy import TrackParcelInput

# Дані взяті з data/orders.csv (реальні тестові дані)
EXISTING_ORDER = "У-0387776"
EXISTING_ORDER_PHONE = "0971879823"
EXISTING_ORDER_TTN = "20400500577861"
CANCELLED_ORDER = "М-0096354"  # статус "Отменен" у CSV


# ── search_order ──────────────────────────────────────────────────

def test_search_order_by_number_found():
    result = search_order(EXISTING_ORDER)
    assert EXISTING_ORDER in result
    assert "доставлено" in result
    assert EXISTING_ORDER_TTN in result


def test_search_order_by_phone_found():
    result = search_order(EXISTING_ORDER_PHONE)
    assert EXISTING_ORDER in result


def test_search_order_not_found():
    result = search_order("У-9999999")
    assert "не знайдено" in result


def test_search_order_status_label_translated():
    """Статус з CSV ("Отменен") має перекладатись через STATUS_LABELS."""
    result = search_order(CANCELLED_ORDER)
    assert "скасовано" in result


def test_search_order_invalid_prefix_raises():
    with pytest.raises(ValidationError):
        search_order("X-0387776")


def test_search_order_too_short_raises():
    with pytest.raises(ValidationError):
        search_order("У-")


# ── get_order_ship_date ───────────────────────────────────────────

def test_get_order_ship_date_found():
    result = get_order_ship_date(EXISTING_ORDER)
    assert EXISTING_ORDER in result
    assert "дата відправки" in result


def test_get_order_ship_date_not_found():
    result = get_order_ship_date("У-9999999")
    assert "не знайдено" in result


# ── track_parcel (Нова Пошта) ──────────────────────────────────────

def test_track_parcel_ttn_too_short_raises():
    with pytest.raises(ValidationError):
        track_parcel("123")


def test_track_parcel_input_strips_non_digits():
    assert TrackParcelInput(ttn="2040-0500-577861").ttn == "20400500577861"


def test_track_parcel_missing_api_key(monkeypatch):
    monkeypatch.setattr(tools_legacy, "NOVA_POSHTA_API_KEY", "")
    result = track_parcel(EXISTING_ORDER_TTN)
    assert "NOVA_POSHTA_API_KEY" in result


@patch("tools_legacy.requests.post")
def test_track_parcel_success(mock_post, monkeypatch):
    monkeypatch.setattr(tools_legacy, "NOVA_POSHTA_API_KEY", "test-key")
    mock_post.return_value = Mock(
        raise_for_status=lambda: None,
        json=lambda: {
            "success": True,
            "data": [{
                "Status": "Прибув до відділення",
                "CitySender": "Київ",
                "CityRecipient": "Львів",
            }],
        },
    )
    result = track_parcel(EXISTING_ORDER_TTN)
    assert "Прибув до відділення" in result
    assert "Київ" in result


@patch("tools_legacy.requests.post")
def test_track_parcel_api_reports_error(mock_post, monkeypatch):
    monkeypatch.setattr(tools_legacy, "NOVA_POSHTA_API_KEY", "test-key")
    mock_post.return_value = Mock(
        raise_for_status=lambda: None,
        json=lambda: {"success": False, "errors": ["TTN not found"]},
    )
    result = track_parcel(EXISTING_ORDER_TTN)
    assert "не вдалося" in result.lower()


# ── track_ukrposhta_parcel ─────────────────────────────────────────

def test_track_ukrposhta_barcode_too_short_raises():
    with pytest.raises(ValidationError):
        track_ukrposhta_parcel("123")


def test_track_ukrposhta_missing_api_token(monkeypatch):
    monkeypatch.setattr(tools_legacy, "UKRPOSHTA_API_TOKEN", "")
    result = track_ukrposhta_parcel("0505676508680")
    assert "UKRPOSHTA_API_TOKEN" in result


@patch("tools_legacy.requests.get")
def test_track_ukrposhta_success(mock_get, monkeypatch):
    monkeypatch.setattr(tools_legacy, "UKRPOSHTA_API_TOKEN", "test-token")
    mock_get.return_value = Mock(
        raise_for_status=lambda: None,
        json=lambda: {
            "barcode": "0505676508680",
            "eventName": "Вручено",
            "name": "Відділення №1",
            "index": "01001",
            "date": "2026-01-05",
        },
    )
    result = track_ukrposhta_parcel("0505676508680")
    assert "Вручено" in result


@patch("tools_legacy.requests.get")
def test_track_ukrposhta_not_found(mock_get, monkeypatch):
    monkeypatch.setattr(tools_legacy, "UKRPOSHTA_API_TOKEN", "test-token")
    mock_get.return_value = Mock(raise_for_status=lambda: None, json=lambda: {})
    result = track_ukrposhta_parcel("0505676508680")
    assert "не знайдено" in result


# ── search_knowledge (RAG) ─────────────────────────────────────────
# Живий виклик до ChromaDB/OpenAI embeddings (потребує OPENAI_API_KEY) —
# так само, як і решта агента (agent.py, plan_execute.py) не мокає LLM.

def test_search_knowledge_finds_relevant_doc():
    result = search_knowledge("товар пошкоджений у відділенні Нової Пошти")
    assert "пошкодж" in result.lower()
