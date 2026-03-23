import requests
from langchain_core.tools import tool
from typing import Annotated, Optional

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


def _get_headers(api_key: str, secret_key: str) -> dict:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }


def get_account_info(api_key: str, secret_key: str) -> dict:
    """Fetch Alpaca account info (cash balance, buying power, etc.)."""
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/account", headers=_get_headers(api_key, secret_key)
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "cash": float(data.get("cash", 0)),
        "buying_power": float(data.get("buying_power", 0)),
        "portfolio_value": float(data.get("portfolio_value", 0)),
        "equity": float(data.get("equity", 0)),
    }


def get_open_positions(api_key: str, secret_key: str) -> list[dict]:
    """Fetch all open positions from Alpaca."""
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/positions", headers=_get_headers(api_key, secret_key)
    )
    resp.raise_for_status()
    positions = []
    for pos in resp.json():
        positions.append(
            {
                "symbol": pos.get("symbol", ""),
                "qty": float(pos.get("qty", 0)),
                "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                "market_value": float(pos.get("market_value", 0)),
                "cost_basis": float(pos.get("cost_basis", 0)),
                "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                "unrealized_plpc": float(pos.get("unrealized_plpc", 0)),
                "current_price": float(pos.get("current_price", 0)),
            }
        )
    return positions


def place_order(api_key: str, secret_key: str, order_params: dict) -> str:
    """Place an order on Alpaca Paper Trading from a flat order_params dict.

    Required keys: symbol, side, type, time_in_force, and either qty or notional.
    Optional keys: limit_price, stop_price, trail_price, trail_percent,
                   extended_hours, order_class, take_profit_limit_price,
                   stop_loss_stop_price, stop_loss_limit_price.
    """
    qty = order_params.get("qty")
    notional = order_params.get("notional")
    if qty is None and notional is None:
        return "Error: You must provide either 'qty' (number of shares) or 'notional' (dollar amount)."
    if qty is not None and notional is not None:
        return "Error: Provide either 'qty' or 'notional', not both."

    body = {
        "symbol": order_params["symbol"],
        "side": order_params["side"],
        "type": order_params["type"],
        "time_in_force": order_params["time_in_force"],
    }

    optional_direct = [
        "qty", "notional", "limit_price", "stop_price",
        "trail_price", "trail_percent", "extended_hours", "order_class",
    ]
    for key in optional_direct:
        if order_params.get(key) is not None:
            body[key] = order_params[key]

    if order_params.get("take_profit_limit_price") is not None:
        body["take_profit"] = {"limit_price": order_params["take_profit_limit_price"]}
    if order_params.get("stop_loss_stop_price") is not None:
        stop_loss = {"stop_price": order_params["stop_loss_stop_price"]}
        if order_params.get("stop_loss_limit_price") is not None:
            stop_loss["limit_price"] = order_params["stop_loss_limit_price"]
        body["stop_loss"] = stop_loss

    resp = requests.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=_get_headers(api_key, secret_key),
        json=body,
    )
    resp.raise_for_status()
    order = resp.json()
    qty_display = order.get("qty") or f"${order.get('notional', 'N/A')}"
    return (
        f"Order placed: {order['side']} {qty_display} {order['symbol']} "
        f"({order['type']}) — status: {order['status']}, id: {order['id']}"
    )
