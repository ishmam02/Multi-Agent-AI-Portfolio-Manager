"""Integration tests for alpaca.py — makes real API calls using keys from .env."""

import os
import time
import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

from tradingagents.dataflows.alpaca import (
    _get_headers,
    get_account_info,
    get_open_positions,
    place_order,
)

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]


# ---------------------------------------------------------------------------
# _get_headers
# ---------------------------------------------------------------------------


def test_get_headers_keys():
    headers = _get_headers(API_KEY, SECRET_KEY)
    assert headers["APCA-API-KEY-ID"] == API_KEY
    assert headers["APCA-API-SECRET-KEY"] == SECRET_KEY
    assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# get_account_info
# ---------------------------------------------------------------------------


def test_get_account_info_returns_expected_fields():
    info = get_account_info(API_KEY, SECRET_KEY)
    assert isinstance(info, dict)
    for field in ("cash", "buying_power", "portfolio_value", "equity"):
        assert field in info, f"Missing field: {field}"
        assert isinstance(info[field], float)


def test_get_account_info_values_are_non_negative():
    info = get_account_info(API_KEY, SECRET_KEY)
    # Paper trading accounts always start with non-negative balances
    assert info["cash"] >= 0
    assert info["buying_power"] >= 0
    assert info["portfolio_value"] >= 0
    assert info["equity"] >= 0


def test_get_account_info_bad_credentials():
    import requests

    with pytest.raises(requests.HTTPError):
        get_account_info("bad_key", "bad_secret")


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------


def test_get_open_positions_returns_list():
    positions = get_open_positions(API_KEY, SECRET_KEY)
    assert isinstance(positions, list)


def test_get_open_positions_structure():
    positions = get_open_positions(API_KEY, SECRET_KEY)
    expected_fields = (
        "symbol",
        "qty",
        "avg_entry_price",
        "market_value",
        "cost_basis",
        "unrealized_pl",
        "unrealized_plpc",
        "current_price",
    )
    for pos in positions:
        for field in expected_fields:
            assert field in pos, f"Position missing field: {field}"
            if field != "symbol":
                assert isinstance(pos[field], float)


def test_get_open_positions_bad_credentials():
    import requests

    with pytest.raises(requests.HTTPError):
        get_open_positions("bad_key", "bad_secret")


# ---------------------------------------------------------------------------
# place_order — validation (no real order sent for these)
# ---------------------------------------------------------------------------


def test_place_order_missing_qty_and_notional():
    result = place_order(
        API_KEY,
        SECRET_KEY,
        {
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        },
    )
    assert result.startswith("Error:")
    assert "qty" in result or "notional" in result


def test_place_order_both_qty_and_notional():
    result = place_order(
        API_KEY,
        SECRET_KEY,
        {
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "qty": 1,
            "notional": 100,
        },
    )
    assert result.startswith("Error:")
    assert "qty" in result or "notional" in result


# ---------------------------------------------------------------------------
# place_order — real market order (1 share of SPY, day order)
# ---------------------------------------------------------------------------


def test_place_order_market_buy():
    result = place_order(
        API_KEY,
        SECRET_KEY,
        {
            "symbol": "SPY",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "qty": 1,
        },
    )
    assert isinstance(result, str)
    assert "Order placed:" in result
    assert "SPY" in result
    assert "buy" in result


def test_place_order_notional_buy():
    result = place_order(
        API_KEY,
        SECRET_KEY,
        {
            "symbol": "SPY",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "notional": 10,
        },
    )
    assert isinstance(result, str)
    assert "Order placed:" in result
    assert "SPY" in result
    assert "buy" in result


def test_place_order_market_sell():
    # Wait for the preceding orders to fill before attempting to sell.
    time.sleep(3)
    result = place_order(
        API_KEY,
        SECRET_KEY,
        {
            "symbol": "SPY",
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "qty": 1,
        },
    )
    assert isinstance(result, str)
    assert "Order placed:" in result
    assert "SPY" in result
    assert "sell" in result


def test_place_order_notional_sell():
    # Wait for the preceding orders to fill before attempting to sell.
    time.sleep(3)
    result = place_order(
        API_KEY,
        SECRET_KEY,
        {
            "symbol": "SPY",
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "notional": 10,
        },
    )
    assert isinstance(result, str)
    assert "Order placed:" in result
    assert "SPY" in result
    assert "sell" in result
