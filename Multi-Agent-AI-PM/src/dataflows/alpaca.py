import os
import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Annotated, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv not installed or no .env file

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


def _get_headers(api_key: str, secret_key: str) -> dict:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }


def _get_alpaca_credentials() -> tuple[str | None, str | None]:
    """Read Alpaca API credentials from environment variables."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    return api_key, secret_key


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


def _fetch_alpaca_news(
    api_key: str,
    secret_key: str,
    symbols: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Fetch news from Alpaca News API with pagination support.

    Alpaca returns max 50 articles per request. Uses next_page_token to
    paginate until all articles in the date range are retrieved.

    Args:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        symbols: Comma-separated ticker symbols (e.g. "AAPL,TSLA"). If None,
                 returns broad market news.
        start: Start datetime in ISO 8601 format (e.g. "2023-01-01T00:00:00Z")
        end: End datetime in ISO 8601 format
        limit: Max articles per page (Alpaca caps at 50)

    Returns:
        List of news article dicts from Alpaca.
    """
    params: dict[str, str | int | bool] = {
        "limit": min(limit, 50),
        "include_content": False,
        "exclude_contentless": True,
    }
    if symbols:
        params["symbols"] = symbols
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    articles: list[dict] = []
    next_token: str | None = None
    max_pages = 50  # 50 × 50 = 2,500 articles max across full date range
    total_pages = 0

    for _ in range(max_pages):
        total_pages += 1
        if next_token:
            params["page_token"] = next_token
        elif "page_token" in params:
            del params["page_token"]

        resp = requests.get(
            f"{ALPACA_DATA_URL}/v1beta1/news",
            headers=_get_headers(api_key, secret_key),
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("news", [])
        articles.extend(batch)

        next_token = data.get("next_page_token")
        if not next_token or not batch:
            break

    if next_token and total_pages >= max_pages:
        print(
            f"[Alpaca News] WARNING: Truncated at {len(articles)} articles "
            f"({max_pages} pages). More news exists in date range but safety cap reached."
        )

    return articles


def _format_alpaca_news(
    articles: list[dict],
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Format Alpaca news articles as a CSV string.

    Columns: date, headline, summary
    """
    import csv
    import io

    if not articles:
        if ticker:
            return f"No news found for {ticker}"
        return "No global news found"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "headline", "summary"])

    for art in articles:
        headline = art.get("headline", "No title")
        summary = art.get("summary", "")

        # Parse created_at to YYYY-MM-DD
        created = art.get("created_at", "")
        article_date = "Unknown"
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                article_date = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                pass

        # Escape newlines in summary so CSV stays single-line per row
        summary_clean = summary.replace("\n", " ").replace("\r", " ") if summary else ""
        writer.writerow([article_date, headline, summary_clean])

    return buf.getvalue()


def get_alpaca_news(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    secret_key: str | None = None,
) -> str:
    """Retrieve company-specific news via Alpaca News API.

    Falls back to empty string if credentials are missing.
    """
    api_key = api_key or _get_alpaca_credentials()[0]
    secret_key = secret_key or _get_alpaca_credentials()[1]
    if not api_key or not secret_key:
        return f"Alpaca credentials not configured. Cannot fetch news for {ticker}."

    # Alpaca expects ISO 8601; convert yyyy-mm-dd to midnight UTC
    start_iso = f"{start_date}T00:00:00Z"
    end_iso = f"{end_date}T23:59:59Z"

    articles = _fetch_alpaca_news(
        api_key,
        secret_key,
        symbols=ticker,
        start=start_iso,
        end=end_iso,
        limit=50,
    )
    return _format_alpaca_news(articles, ticker=ticker, start_date=start_date, end_date=end_date)


def get_alpaca_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
    api_key: str | None = None,
    secret_key: str | None = None,
) -> str:
    """Retrieve broad market / macro news via Alpaca News API (no symbol filter).

    Falls back to empty string if credentials are missing.
    """
    api_key = api_key or _get_alpaca_credentials()[0]
    secret_key = secret_key or _get_alpaca_credentials()[1]
    if not api_key or not secret_key:
        return "Alpaca credentials not configured. Cannot fetch global news."

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)
    start_iso = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end_iso = end_dt.strftime("%Y-%m-%dT23:59:59Z")

    articles = _fetch_alpaca_news(
        api_key,
        secret_key,
        symbols=None,  # broad market news
        start=start_iso,
        end=end_iso,
        limit=50,
    )
    return _format_alpaca_news(
        articles,
        ticker=None,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=curr_date,
    )


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
