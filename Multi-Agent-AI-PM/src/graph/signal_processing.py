# TradingAgents/graph/signal_processing.py

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI

from src.dataflows.alpaca import (
    get_account_info,
    get_open_positions,
    place_order,
)

_ORDER_PARAMS_SCHEMA = """\
**Required parameters (must always be present):**
  - symbol (str): Ticker symbol, e.g. "AAPL", "TSLA".
  - side (str): Trade direction. Must be exactly "buy" or "sell". Use "sell" for both closing a long position AND opening a short position.
  - type (str): Order type. Must be one of: "market", "limit", "stop", "stop_limit", "trailing_stop".
  - time_in_force (str): How long the order stays active. Must be one of: "day" (expires end of day), "gtc" (good til canceled), "opg" (market on open), "cls" (market on close), "ioc" (immediate or cancel), "fok" (fill or kill).

**Quantity (exactly one of these is required):**
  - qty (str): Number of shares to trade, e.g. "10", "100". Provide qty OR notional, never both.
  - notional (str): Dollar amount to trade, e.g. "5000.00". Provide notional OR qty, never both.

**Conditional parameters (include only when the order type requires them):**
  - limit_price (str): The limit price. REQUIRED for "limit" and "stop_limit" order types.
  - stop_price (str): The stop trigger price. REQUIRED for "stop" and "stop_limit" order types.
  - trail_price (str): Trailing amount in dollars. For "trailing_stop" orders only. Provide trail_price OR trail_percent, not both.
  - trail_percent (str): Trailing amount as a percentage. For "trailing_stop" orders only. Provide trail_percent OR trail_price, not both.

**Advanced parameters (include only when needed):**
  - extended_hours (bool): Set to true to allow execution during pre-market and after-hours sessions.
  - order_class (str): Enables multi-leg orders. Must be one of: "simple" (default single order), "bracket" (entry + take-profit + stop-loss), "oco" (one-cancels-other: two exit legs, no entry), "oto" (one-triggers-other: entry triggers a second order).
  - take_profit_limit_price (str): The take-profit limit price. Used with order_class "bracket" or "oto".
  - stop_loss_stop_price (str): The stop-loss trigger price. Used with order_class "bracket" or "oco".
  - stop_loss_limit_price (str): Optional stop-loss limit price (makes the stop-loss a stop-limit instead of a stop-market). Used alongside stop_loss_stop_price."""


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content

    def propose_trade(
        self, full_signal: str, ticker: str, api_key: str, secret_key: str
    ) -> Dict[str, Any]:
        """
        Propose a trade on Alpaca based on the full analysis signal.

        Fetches account info and open positions, then asks the LLM to
        produce order parameters as JSON. The order is NOT placed —
        the caller decides whether to execute.

        Returns:
            Dict with keys:
              - reasoning (str): The LLM's explanation
              - order_params (dict | None): Proposed order params, or None if no trade
        """
        account = get_account_info(api_key, secret_key)
        positions = get_open_positions(api_key, secret_key)

        positions_str = "No open positions."
        if positions:
            positions_str = json.dumps(positions, indent=2)

        messages = [
            (
                "system",
                "You are a trade planning agent. Based on the analysis signal and the investor's current Alpaca account state, propose the appropriate STOCK order.\n\n"
                "IMPORTANT RULES:\n"
                "- Only propose stock equity trades. If the analysis mentions options, puts, calls, or any derivatives strategy, IGNORE those parts and translate the directional intent into a stock order instead.\n"
                '- Short selling is allowed. If the analysis is bearish and recommends selling a stock the investor does NOT own, you may propose a short sell by using side "sell" with the desired qty. Alpaca paper trading supports short selling.\n'
                "- Use market orders with 'day' time_in_force unless the analysis specifically recommends limit entries, stop-losses, or bracket strategies.\n"
                "- If the analysis specifies price targets and stop-loss levels, use a bracket order to capture both.\n"
                "- If the decision is HOLD and the investor has no position, output `null` for the order and explain why.\n\n"
                "You must respond with EXACTLY two sections separated by a line containing only '---':\n\n"
                "1. **REASONING** — your explanation of why this order is appropriate, including how you translated the analysis into order parameters.\n"
                "2. **ORDER JSON** — a single JSON object with the order parameters, or the exact word `null` if no order should be placed.\n\n"
                f"**Order Parameter Reference:**\n{_ORDER_PARAMS_SCHEMA}\n\n"
                "**Example responses:**\n\n"
                "Example 1 — Simple market buy:\n"
                "The analysis strongly recommends buying AAPL. No specific price levels mentioned, so a straightforward market order is appropriate. Allocating 50 shares based on available buying power.\n"
                "---\n"
                '{"symbol": "AAPL", "side": "buy", "type": "market", "time_in_force": "day", "qty": "50"}\n\n'
                "Example 2 — Bracket order with take-profit and stop-loss:\n"
                "The analysis recommends buying TSLA with a price target of $260 and a stop-loss at $230. Using a bracket order to automatically manage both exit levels.\n"
                "---\n"
                '{"symbol": "TSLA", "side": "buy", "type": "market", "time_in_force": "day", "qty": "20", "order_class": "bracket", "take_profit_limit_price": "260.00", "stop_loss_stop_price": "230.00"}\n\n'
                "Example 3 — Limit buy with trailing stop:\n"
                "The analysis suggests accumulating NVDA below $450. Using a limit order for entry. Since no fixed stop is given, a trailing stop OTO will protect gains.\n"
                "---\n"
                '{"symbol": "NVDA", "side": "buy", "type": "limit", "time_in_force": "gtc", "qty": "15", "limit_price": "450.00", "order_class": "oto", "stop_loss_stop_price": "430.00"}\n\n'
                "Example 4 — Stop-limit sell to exit a position:\n"
                "The analysis recommends selling if the stock breaks below support at $150. Using a stop-limit sell to exit the existing 100-share position.\n"
                "---\n"
                '{"symbol": "AAPL", "side": "sell", "type": "stop_limit", "time_in_force": "gtc", "qty": "100", "stop_price": "150.00", "limit_price": "148.00"}\n\n'
                "Example 5 — Short sell (bearish, no existing position):\n"
                "The analysis is bearish on META and the investor has no position. Opening a short position with a bracket order: short sell at market with a stop-loss (buy-to-cover) at $540 and take-profit at $480.\n"
                "---\n"
                '{"symbol": "META", "side": "sell", "type": "market", "time_in_force": "day", "qty": "25", "order_class": "bracket", "take_profit_limit_price": "480.00", "stop_loss_stop_price": "540.00"}\n\n'
                "Example 6 — HOLD with no position:\n"
                "The analysis recommends holding. The investor has no existing position in MSFT, so no order is needed.\n"
                "---\n"
                "null",
            ),
            (
                "human",
                f"**Analysis Signal for {ticker}:**\n{full_signal}\n\n"
                f"**Alpaca Account:**\n"
                f"- Cash: ${account['cash']:,.2f}\n"
                f"- Buying Power: ${account['buying_power']:,.2f}\n"
                f"- Portfolio Value: ${account['portfolio_value']:,.2f}\n"
                f"- Equity: ${account['equity']:,.2f}\n\n"
                f"**Open Positions:**\n{positions_str}\n\n"
                f"Based on the above, propose a stock trade for {ticker}.",
            ),
        ]

        response = self.quick_thinking_llm.invoke(messages).content
        reasoning, order_params = self._parse_proposal(response)
        return {"reasoning": reasoning, "order_params": order_params}

    @staticmethod
    def _parse_proposal(response: str) -> tuple[str, Optional[dict]]:
        """Parse the LLM's two-section response into reasoning and order params."""
        if "---" in response:
            parts = response.split("---", 1)
            reasoning = parts[0].strip()
            json_part = parts[1].strip()
        else:
            reasoning = response
            json_part = ""

        order_params = None
        if json_part and json_part.lower() != "null":
            cleaned = json_part
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            try:
                order_params = json.loads(cleaned)
            except json.JSONDecodeError:
                reasoning += f"\n\n(Failed to parse order JSON: {json_part})"
                order_params = None

        return reasoning, order_params

    def propose_portfolio_trades(
        self,
        ticker_signals: Dict[str, str],
        api_key: str,
        secret_key: str,
    ) -> Dict[str, Any]:
        """Portfolio Manager: evaluate ALL stock decisions holistically.

        Receives every ticker's trader_investment_plan at once and produces
        portfolio-level allocation with fact-based reasoning per stock.

        Args:
            ticker_signals: {ticker: trader_investment_plan_text}
            api_key: Alpaca API key
            secret_key: Alpaca secret key

        Returns:
            {
                "portfolio_reasoning": str,
                "orders": [
                    {
                        "ticker": str,
                        "decision": str,      # BUY / SELL / HOLD
                        "reasoning": str,      # fact-based per-stock reasoning
                        "order_params": dict | None
                    }, ...
                ]
            }
        """
        account = get_account_info(api_key, secret_key)
        positions = get_open_positions(api_key, secret_key)

        positions_str = "No open positions."
        if positions:
            positions_str = json.dumps(positions, indent=2)

        # Build per-ticker signal block
        signal_blocks = []
        for ticker, signal in ticker_signals.items():
            signal_blocks.append(f"### {ticker}\n{signal}")
        all_signals = "\n\n".join(signal_blocks)

        ticker_list = list(ticker_signals.keys())

        # Determine market-hours status for extended_hours guidance
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = now.weekday() < 5 and market_open <= now <= market_close
        hours_note = (
            "The market is currently OPEN (regular trading hours)."
            if is_market_hours
            else "The market is currently CLOSED (outside regular hours 9:30 AM – 4:00 PM ET, Mon–Fri). "
            "For any orders you want filled now, set extended_hours=true and time_in_force='day'."
        )

        messages = [
            (
                "system",
                "You are a Portfolio Manager at a hedge fund. You are reviewing "
                "analysis signals for MULTIPLE stocks simultaneously and must decide "
                "how to allocate capital across them as a cohesive portfolio.\n\n"
                "IMPORTANT RULES:\n"
                "- Only propose stock equity trades. If the analysis mentions options, puts, calls, "
                "or any derivatives strategy, IGNORE those parts and translate the directional intent "
                "into a stock order instead.\n"
                "- Short selling is allowed on Alpaca paper trading. If the analysis is bearish and "
                "recommends selling a stock the investor does NOT own, you may propose a short sell "
                'by using side "sell" with the desired qty.\n'
                "- **MANDATORY RISK MANAGEMENT**: Any trade with theoretically infinite risk MUST include "
                "a stop-loss. Specifically:\n"
                '  - Every SHORT SELL must use order_class "bracket" with a stop_loss_stop_price '
                "(buy-to-cover level) to cap downside. No exceptions.\n"
                "  - For long positions, bracket orders with stop-loss are strongly recommended when "
                "the analysis provides price targets and risk levels.\n"
                "- Consider POSITION SIZING: allocate more capital to higher-conviction trades. "
                "Never allocate more than 30% of buying power to a single position.\n"
                "- Consider CORRELATION: avoid over-concentrating in one sector unless justified.\n"
                "- Consider TOTAL EXPOSURE: the sum of all new positions must not exceed available buying power.\n"
                "- Each reasoning MUST cite specific facts from the analysis (P/E ratios, growth rates, "
                "analyst targets, risk assessments, technical levels) — not vague opinions.\n"
                "- Use the FULL range of order types and parameters when the analysis supports it:\n"
                "  - Use bracket orders when analysis gives both a price target and stop-loss level.\n"
                "  - Use limit orders when analysis recommends accumulating below a certain price.\n"
                "  - Use trailing_stop (via trail_percent or trail_price) when the analysis suggests "
                "letting winners run with a trailing exit.\n"
                "  - Use stop or stop_limit orders for conditional entries or exits.\n"
                "- Use market orders with 'day' time_in_force only when no specific price levels are given.\n"
                "- If a stock's decision is HOLD and no position exists, set order_params to null.\n\n"
                f"**Current Time:** {current_time_str}\n"
                f"**Market Status:** {hours_note}\n\n"
                "RESPONSE FORMAT — you MUST respond with exactly two sections separated by a line "
                "containing only '---':\n\n"
                "1. **PORTFOLIO STRATEGY** — overall reasoning about how these trades work together, "
                "risk/reward balance, sector exposure, capital allocation rationale.\n\n"
                "2. **ORDERS JSON** — a JSON array with one object per ticker (see examples below for format).\n\n"
                f"**Order Parameter Reference:**\n{_ORDER_PARAMS_SCHEMA}\n\n"
                "**Example order_params for each order style:**\n\n"
                "Example 1 — Simple market buy (no specific price levels in analysis):\n"
                '{"symbol": "AAPL", "side": "buy", "type": "market", "time_in_force": "day", "qty": "50"}\n\n'
                "Example 2 — Bracket buy with take-profit and stop-loss (analysis gives target $260 and stop $230):\n"
                '{"symbol": "TSLA", "side": "buy", "type": "market", "time_in_force": "day", "qty": "20", '
                '"order_class": "bracket", "take_profit_limit_price": "260.00", "stop_loss_stop_price": "230.00"}\n\n'
                "Example 3 — Bracket buy with stop-limit stop-loss (tighter exit control):\n"
                '{"symbol": "AMZN", "side": "buy", "type": "market", "time_in_force": "day", "qty": "30", '
                '"order_class": "bracket", "take_profit_limit_price": "210.00", '
                '"stop_loss_stop_price": "180.00", "stop_loss_limit_price": "178.00"}\n\n'
                "Example 4 — Limit buy with OTO trailing stop (accumulate below $450, trail 5% on fill):\n"
                '{"symbol": "NVDA", "side": "buy", "type": "limit", "time_in_force": "gtc", "qty": "15", '
                '"limit_price": "450.00", "order_class": "oto", "stop_loss_stop_price": "430.00"}\n\n'
                "Example 5 — Trailing-stop sell to protect gains on existing long (trail by $5):\n"
                '{"symbol": "MSFT", "side": "sell", "type": "trailing_stop", "time_in_force": "gtc", '
                '"qty": "40", "trail_price": "5.00"}\n\n'
                "Example 6 — Trailing-stop sell by percentage (trail 4%):\n"
                '{"symbol": "GOOG", "side": "sell", "type": "trailing_stop", "time_in_force": "gtc", '
                '"qty": "25", "trail_percent": "4.0"}\n\n'
                "Example 7 — Stop-limit sell to exit at support breakdown ($150 trigger, $148 limit):\n"
                '{"symbol": "AAPL", "side": "sell", "type": "stop_limit", "time_in_force": "gtc", '
                '"qty": "100", "stop_price": "150.00", "limit_price": "148.00"}\n\n'
                "Example 8 — Stop buy for breakout entry (trigger at $320, then market fill):\n"
                '{"symbol": "CRM", "side": "buy", "type": "stop", "time_in_force": "gtc", '
                '"qty": "35", "stop_price": "320.00"}\n\n'
                "Example 9 — Short sell with mandatory bracket (bearish, stop-loss at $540, take-profit at $480):\n"
                '{"symbol": "META", "side": "sell", "type": "market", "time_in_force": "day", '
                '"qty": "25", "order_class": "bracket", "take_profit_limit_price": "480.00", '
                '"stop_loss_stop_price": "540.00"}\n\n'
                "Example 10 — Notional buy (invest $5000 worth, market hours only):\n"
                '{"symbol": "SPY", "side": "buy", "type": "market", "time_in_force": "day", '
                '"notional": "5000.00"}\n\n'
                "Example 11 — Extended-hours limit buy (outside market hours):\n"
                '{"symbol": "AAPL", "side": "buy", "type": "limit", "time_in_force": "day", '
                '"qty": "50", "limit_price": "178.00", "extended_hours": true}\n\n'
                "Example 12 — OCO exit (take-profit at $200 OR stop-loss at $160 on existing position):\n"
                '{"symbol": "DIS", "side": "sell", "type": "limit", "time_in_force": "gtc", '
                '"qty": "60", "limit_price": "200.00", "order_class": "oco", '
                '"stop_loss_stop_price": "160.00"}\n\n'
                "Example 13 — HOLD with no position:\n"
                "null\n\n"
                "**Full response example:**\n"
                "Portfolio strategy reasoning here...\n"
                "---\n"
                "```json\n"
                "[\n"
                '  {"ticker": "AAPL", "decision": "BUY", "reasoning": "Strong fundamentals...", '
                '"order_params": {"symbol": "AAPL", "side": "buy", "type": "market", '
                '"time_in_force": "day", "qty": "50", "order_class": "bracket", '
                '"take_profit_limit_price": "195.00", "stop_loss_stop_price": "170.00"}},\n'
                '  {"ticker": "TSLA", "decision": "HOLD", "reasoning": "Neutral outlook...", '
                '"order_params": null}\n'
                "]\n"
                "```\n\n"
                f"You must produce exactly {len(ticker_list)} entries, one per ticker: "
                f"{', '.join(ticker_list)}.",
            ),
            (
                "human",
                f"**Alpaca Account:**\n"
                f"- Cash: ${account['cash']:,.2f}\n"
                f"- Buying Power: ${account['buying_power']:,.2f}\n"
                f"- Portfolio Value: ${account['portfolio_value']:,.2f}\n"
                f"- Equity: ${account['equity']:,.2f}\n\n"
                f"**Open Positions:**\n{positions_str}\n\n"
                f"**Analysis Signals:**\n\n{all_signals}\n\n"
                f"Based on the above, propose portfolio-level trades for all {len(ticker_list)} stocks.",
            ),
        ]

        response = self.quick_thinking_llm.invoke(messages).content
        return self._parse_portfolio_proposal(response, ticker_list)

    def _parse_portfolio_proposal(
        self, response: str, expected_tickers: List[str]
    ) -> Dict[str, Any]:
        """Parse the portfolio manager's response into structured output."""
        portfolio_reasoning = ""
        orders = []

        if "---" in response:
            parts = response.split("---", 1)
            portfolio_reasoning = parts[0].strip()
            json_part = parts[1].strip()
        else:
            portfolio_reasoning = response
            json_part = ""

        if json_part:
            # Extract JSON array
            cleaned = json_part
            if "```" in cleaned:
                # Find JSON block between backticks
                segments = cleaned.split("```")
                for seg in segments[1:]:
                    if seg.strip().startswith("json"):
                        seg = seg.strip()[4:]
                    seg = seg.strip()
                    if seg.startswith("["):
                        cleaned = seg
                        break

            # Try to find array in the cleaned text
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    orders = parsed
            except json.JSONDecodeError:
                portfolio_reasoning += (
                    f"\n\n(Failed to parse orders JSON from response)"
                )

        # Validate and normalize orders
        validated_orders = []
        for order in orders:
            if not isinstance(order, dict):
                continue
            validated_orders.append(
                {
                    "ticker": order.get("ticker", ""),
                    "decision": order.get("decision", "HOLD"),
                    "reasoning": order.get("reasoning", ""),
                    "order_params": order.get("order_params"),
                }
            )

        return {
            "portfolio_reasoning": portfolio_reasoning,
            "orders": validated_orders,
        }

    @staticmethod
    def execute_portfolio_orders(
        orders: List[Dict[str, Any]], api_key: str, secret_key: str
    ) -> List[Dict[str, str]]:
        """Execute all non-null orders from a portfolio proposal.

        Returns list of {ticker, result} dicts.
        """
        results = []
        for order in orders:
            ticker = order.get("ticker", "")
            params = order.get("order_params")
            if params is None:
                results.append({"ticker": ticker, "result": "No order (HOLD)"})
                continue
            try:
                result = place_order(api_key, secret_key, params)
                results.append({"ticker": ticker, "result": result})
            except Exception as e:
                results.append({"ticker": ticker, "result": f"Error: {str(e)}"})
        return results

    @staticmethod
    def execute_order(order_params: dict, api_key: str, secret_key: str) -> str:
        """
        Execute a previously proposed order on Alpaca.

        Args:
            order_params: The order parameters dict from propose_trade
            api_key: Alpaca API key
            secret_key: Alpaca secret key

        Returns:
            Order execution result string
        """
        return place_order(api_key, secret_key, order_params)
