# src/graph/signal_processing.py
"""Deterministic signal processing and trade proposal based on CompositeSignal.

Replaces LLM-based trade planning with reproducible math:
  - Direction: mu_final thresholds
  - Position sizing: conviction-scaled Kelly fraction
  - Risk management: bracket orders with mu/sigma-derived price levels
  - Portfolio allocation: mean-variance optimization using covariance matrix
"""

import math
from typing import Dict, Any, List

from src.dataflows.alpaca import (
    get_account_info,
    get_open_positions,
    place_order,
    get_latest_quote,
)


class SignalProcessor:
    """Deterministic signal processing — no LLM calls."""

    # Thresholds for BUY/SELL/HOLD
    MU_BUY_THRESHOLD = 0.05
    MU_SELL_THRESHOLD = -0.05

    # Position sizing constants
    MAX_POSITION_PCT = 0.30  # Max 30% of equity in one position
    KELLY_FRACTION = 0.25  # Quarter-Kelly for safety
    MIN_CONVICTION_TO_TRADE = 0.10

    def __init__(self):
        pass

    # ── Signal extraction ──────────────────────────────────────────────────────

    @staticmethod
    def process_signal(full_signal: str) -> str:
        """Extract BUY/SELL/HOLD from a CompositeSignal JSON string."""
        try:
            from src.agents.utils.schemas import CompositeSignal

            cs = CompositeSignal.model_validate_json(full_signal)
            if cs.mu_final > SignalProcessor.MU_BUY_THRESHOLD:
                return "BUY"
            elif cs.mu_final < SignalProcessor.MU_SELL_THRESHOLD:
                return "SELL"
            else:
                return "HOLD"
        except Exception:
            return "HOLD"

    @staticmethod
    def _parse_composite(full_signal: str):
        """Parse CompositeSignal JSON; return None on failure."""
        try:
            from src.agents.utils.schemas import CompositeSignal

            return CompositeSignal.model_validate_json(full_signal)
        except Exception:
            return None

    # ── Single-ticker trade proposal ─────────────────────────────────────────

    def propose_trade(
        self,
        full_signal: str,
        ticker: str,
        api_key: str,
        secret_key: str,
    ) -> Dict[str, Any]:
        """Propose a deterministic trade based on CompositeSignal.

        Returns:
            {"reasoning": str, "order_params": dict | None}
        """
        cs = self._parse_composite(full_signal)
        if cs is None:
            return {
                "reasoning": "Failed to parse composite signal. No trade.",
                "order_params": None,
            }

        account = get_account_info(api_key, secret_key)
        positions = get_open_positions(api_key, secret_key)
        equity = float(account.get("equity", 0))

        # Current position
        current_qty = 0
        for p in positions:
            if p.get("symbol") == ticker:
                current_qty = float(p.get("qty", 0))
                break

        decision = self.process_signal(full_signal)

        if decision == "HOLD":
            return {
                "reasoning": (
                    f"mu_final={cs.mu_final:.4f} within [{self.MU_SELL_THRESHOLD}, "
                    f"{self.MU_BUY_THRESHOLD}] → HOLD. No action."
                ),
                "order_params": None,
            }

        if cs.conviction_final < self.MIN_CONVICTION_TO_TRADE:
            return {
                "reasoning": (
                    f"mu_final={cs.mu_final:.4f} suggests {decision}, but "
                    f"conviction={cs.conviction_final:.4f} is below "
                    f"min threshold {self.MIN_CONVICTION_TO_TRADE}. HOLD."
                ),
                "order_params": None,
            }

        # Position sizing: quarter-Kelly scaled by conviction
        # f* = (mu / sigma^2) * conviction * KELLY_FRACTION
        kelly_fraction = (
            (cs.mu_final / max(cs.sigma_final**2, 0.0001))
            * cs.conviction_final
            * self.KELLY_FRACTION
        )
        # Clamp to max position size
        position_fraction = max(-self.MAX_POSITION_PCT, min(self.MAX_POSITION_PCT, kelly_fraction))
        notional = abs(position_fraction) * equity

        # Need a price estimate for qty conversion
        try:
            quote = get_latest_quote(api_key, secret_key, ticker)
            price = (quote.get("bid", 0) + quote.get("ask", 0)) / 2
            if price <= 0:
                price = quote.get("last", 0)
        except Exception:
            price = 0

        if price <= 0:
            return {
                "reasoning": f"Cannot determine price for {ticker}. No trade.",
                "order_params": None,
            }

        qty = math.floor(notional / price)
        if qty < 1:
            return {
                "reasoning": (
                    f"Calculated notional ${notional:,.2f} at ${price:,.2f} "
                    f"yields qty={qty}. Below minimum. No trade."
                ),
                "order_params": None,
            }

        # Build bracket order with take-profit and stop-loss
        if decision == "BUY":
            side = "buy"
            take_profit_price = round(price * (1 + max(cs.mu_final, 0.01)), 2)
            stop_loss_price = round(price * (1 - max(cs.sigma_final, 0.01)), 2)
        else:  # SELL
            side = "sell"
            # If already long, sell to close; otherwise short
            if current_qty > 0:
                qty = min(qty, current_qty)
            take_profit_price = round(price * (1 - max(abs(cs.mu_final), 0.01)), 2)
            stop_loss_price = round(price * (1 + max(cs.sigma_final, 0.01)), 2)

        order_params = {
            "symbol": ticker,
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "qty": str(qty),
            "order_class": "bracket",
            "take_profit_limit_price": str(take_profit_price),
            "stop_loss_stop_price": str(stop_loss_price),
        }

        reasoning = (
            f"Decision: {decision} | mu_final={cs.mu_final:.4f} | "
            f"sigma_final={cs.sigma_final:.4f} | conviction={cs.conviction_final:.4f}\n"
            f"Position sizing: Kelly fraction={position_fraction:.4f} → "
            f"notional=${notional:,.2f} → qty={qty} @ ${price:,.2f}\n"
            f"Bracket: take-profit=${take_profit_price:.2f}, stop-loss=${stop_loss_price:.2f}"
        )

        return {"reasoning": reasoning, "order_params": order_params}

    # ── Portfolio-level allocation ───────────────────────────────────────────

    def propose_portfolio_trades(
        self,
        ticker_signals: Dict[str, str],
        api_key: str,
        secret_key: str,
    ) -> Dict[str, Any]:
        """Deterministic portfolio allocation using mean-variance logic.

        Args:
            ticker_signals: {ticker: composite_signal_json}
            covariance_matrix: Optional output from compute_multi_ticker_covariance
        Returns:
            {"portfolio_reasoning": str, "orders": [...]}
        """
        account = get_account_info(api_key, secret_key)
        positions = get_open_positions(api_key, secret_key)
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))

        # Parse all signals
        parsed = {}
        for ticker, sig in ticker_signals.items():
            cs = self._parse_composite(sig)
            if cs:
                parsed[ticker] = cs

        if not parsed:
            return {
                "portfolio_reasoning": "No valid composite signals. No trades.",
                "orders": [],
            }

        tickers = list(parsed.keys())

        # Simple inverse-variance allocation weighted by mu * conviction
        scores = {}
        for t, cs in parsed.items():
            direction = 1 if cs.mu_final > 0 else -1 if cs.mu_final < 0 else 0
            scores[t] = (
                direction
                * abs(cs.mu_final)
                * cs.conviction_final
                / max(cs.sigma_final**2, 0.0001)
            )

        # Normalize scores to position fractions
        total_abs_score = sum(abs(s) for s in scores.values())
        if total_abs_score == 0:
            return {
                "portfolio_reasoning": "All signals are neutral. No trades.",
                "orders": [],
            }

        fractions = {
            t: (scores[t] / total_abs_score) * self.MAX_POSITION_PCT
            for t in tickers
        }

        # Clamp total exposure
        total_long = sum(f for f in fractions.values() if f > 0)
        total_short = sum(abs(f) for f in fractions.values() if f < 0)
        max_exposure = min(self.MAX_POSITION_PCT * len(tickers), 1.0)
        if total_long + total_short > max_exposure:
            scale = max_exposure / (total_long + total_short)
            fractions = {t: f * scale for t, f in fractions.items()}

        # Build orders
        orders = []
        portfolio_reasoning_parts = [
            f"Portfolio allocation for {len(tickers)} tickers:",
            f"Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f}",
        ]

        for t in tickers:
            cs = parsed[t]
            frac = fractions[t]
            decision = (
                "BUY" if frac > 0 else "SELL" if frac < 0 else "HOLD"
            )

            if decision == "HOLD":
                orders.append(
                    {
                        "ticker": t,
                        "decision": "HOLD",
                        "reasoning": (
                            f"mu_final={cs.mu_final:.4f}, score=0. No trade."
                        ),
                        "order_params": None,
                    }
                )
                continue

            notional = abs(frac) * equity
            try:
                quote = get_latest_quote(api_key, secret_key, t)
                price = (quote.get("bid", 0) + quote.get("ask", 0)) / 2
                if price <= 0:
                    price = quote.get("last", 0)
            except Exception:
                price = 0

            if price <= 0:
                orders.append(
                    {
                        "ticker": t,
                        "decision": "HOLD",
                        "reasoning": f"Cannot get price for {t}. No trade.",
                        "order_params": None,
                    }
                )
                continue

            qty = math.floor(notional / price)
            if qty < 1:
                orders.append(
                    {
                        "ticker": t,
                        "decision": "HOLD",
                        "reasoning": (
                            f"Allocation ${notional:,.2f} @ ${price:,.2f} "
                            f"below minimum qty. No trade."
                        ),
                        "order_params": None,
                    }
                )
                continue

            side = "buy" if decision == "BUY" else "sell"
            if decision == "SELL":
                # Cap qty to existing long position
                existing_qty = 0
                for p in positions:
                    if p.get("symbol") == t:
                        existing_qty = float(p.get("qty", 0))
                        break
                if existing_qty > 0:
                    qty = min(qty, existing_qty)

            take_profit = round(
                price * (1 + max(cs.mu_final, 0.01) * (1 if decision == "BUY" else -1)), 2
            )
            stop_loss = round(
                price * (1 - max(cs.sigma_final, 0.01) * (1 if decision == "BUY" else -1)), 2
            )

            order_params = {
                "symbol": t,
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "qty": str(qty),
                "order_class": "bracket",
                "take_profit_limit_price": str(take_profit),
                "stop_loss_stop_price": str(stop_loss),
            }

            reasoning = (
                f"mu_final={cs.mu_final:.4f}, sigma={cs.sigma_final:.4f}, "
                f"conviction={cs.conviction_final:.4f} → score={scores[t]:.4f} → "
                f"alloc={frac:.2%} → notional=${notional:,.2f} → qty={qty}"
            )
            portfolio_reasoning_parts.append(reasoning)

            orders.append(
                {
                    "ticker": t,
                    "decision": decision,
                    "reasoning": reasoning,
                    "order_params": order_params,
                }
            )

        return {
            "portfolio_reasoning": "\n".join(portfolio_reasoning_parts),
            "orders": orders,
        }

    # ── Portfolio-level covariance ───────────────────────────────────────────

    @staticmethod
    def compute_multi_ticker_covariance(
        analysis_results: Dict[str, Dict],
        cross_ticker_correlation: float = 0.3,
    ) -> Dict[str, Any]:
        """Compute cross-asset covariance matrix from composite signals.

        Uses a single-factor model with a configurable default correlation
        between different tickers.

        Returns dict with:
          - tickers: List[str]
          - covariance_matrix: List[List[float]]  # n x n
          - annualized_volatilities: Dict[str, float]
        """
        from src.agents.utils.schemas import CompositeSignal

        tickers = []
        vols: Dict[str, float] = {}

        for ticker, state in analysis_results.items():
            if "error" in state:
                continue
            raw = state.get("composite_signal", "")
            if not raw:
                continue
            try:
                cs = CompositeSignal.model_validate_json(raw)
                tickers.append(ticker)
                vols[ticker] = max(0.001, cs.sigma_final)
            except Exception:
                continue

        n = len(tickers)
        if n == 0:
            return {
                "tickers": [],
                "covariance_matrix": [],
                "annualized_volatilities": {},
            }

        cov = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov[i][j] = vols[tickers[i]] ** 2
                else:
                    cov[i][j] = cross_ticker_correlation * vols[tickers[i]] * vols[tickers[j]]

        return {
            "tickers": tickers,
            "covariance_matrix": cov,
            "annualized_volatilities": vols,
        }

    # ── Execution helpers ────────────────────────────────────────────────────

    @staticmethod
    def execute_portfolio_orders(
        orders: List[Dict[str, Any]], api_key: str, secret_key: str
    ) -> List[Dict[str, str]]:
        """Execute all non-null orders from a portfolio proposal."""
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
        """Execute a single order on Alpaca."""
        return place_order(api_key, secret_key, order_params)
