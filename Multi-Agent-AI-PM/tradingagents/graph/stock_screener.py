"""Hedge fund-style stock screening pipeline.

Three-stage process:
  Stage 1 (Quantitative): Multi-strategy screen via yfinance (value, growth,
      momentum, quality, dividend, turnaround) + sector/industry leaders.
  Stage 2 (Scoring & Ranking): Weighted composite scoring across 6 factor
      families (valuation, profitability, growth, momentum, financial health,
      analyst sentiment) with sector-diversity and market-cap tier bonuses.
  Stage 3 (LLM Selection): Portfolio strategist picks final set, personalized
      by the user's risk profile.

Portfolio holdings are ALWAYS included in the final research list — the
screener only discovers *new* tickers to add alongside them.
"""

import csv
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import yfinance as yf

from tradingagents.dataflows.y_finance import (
    screen_stocks,
    get_analyst_recommendations,
    get_growth_estimates,
)


# ---------------------------------------------------------------------------
# US-exchange allow-list (Yahoo Finance exchange codes)
# ---------------------------------------------------------------------------
_US_EXCHANGES = frozenset(
    {
        "NMS",  # NASDAQ Global Select
        "NGM",  # NASDAQ Global Market
        "NCM",  # NASDAQ Capital Market
        "NYQ",  # NYSE
        "ASE",  # NYSE American (AMEX)
        "PCX",  # NYSE Arca
        "BTS",  # BATS
    }
)


class StockScreener:
    """Discovers new stocks to research using a hybrid quant + LLM pipeline."""

    def __init__(self, config: Dict[str, Any], llm=None):
        self.config = config
        self.llm = llm  # quick_thinking_llm for Stage 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen_universe(
        self,
        portfolio_tickers: List[str],
        trade_date: str,
        num_picks: int = 5,
        risk_profile: Optional[Dict] = None,
        criteria: Optional[Dict] = None,
        on_progress=None,
    ) -> List[str]:
        """Run the full 3-stage screening pipeline.

        Portfolio tickers are ALWAYS included in the returned list.  The
        screener discovers ``num_picks`` *new* tickers on top of those.

        Args:
            portfolio_tickers: Tickers already in the portfolio.
            trade_date: Current trading date (YYYY-MM-DD).
            num_picks: How many **new** stocks to discover (portfolio tickers
                are added automatically on top of this number).
            risk_profile: User risk profile dict for personalized LLM selection.
            criteria: Optional overrides for quantitative filters.
            on_progress: Optional callback ``on_progress(stage, data)`` called
                after each pipeline stage with:
                - stage 1: ("quantitative", list_of_ticker_strings)
                - stage 2: ("scored", list_of_scored_dicts)  # top 40
                - stage 3: ("llm_picks", list_of_ticker_strings)
                - final:   ("final", list_of_ticker_strings)

        Returns:
            List of ticker symbols to research (portfolio + new discoveries).
        """
        portfolio_set = {t.upper() for t in portfolio_tickers}

        def _notify(stage, data):
            if on_progress:
                try:
                    on_progress(stage, data)
                except Exception:
                    pass

        # Stage 1: broad quantitative filter
        candidates = self._quantitative_screen(portfolio_set, criteria)
        _notify("quantitative", candidates)

        if not candidates:
            # Even if screening fails, always research portfolio holdings
            final = list(portfolio_set) if portfolio_set else []
            _notify("final", final)
            return final

        # Stage 2: score and rank
        ranked = self._score_and_rank(candidates, portfolio_set)
        shortlist_size = min(100, len(ranked))
        shortlist = ranked[:shortlist_size]
        _notify("scored", shortlist)

        if not ranked:
            final = list(portfolio_set) if portfolio_set else []
            _notify("final", final)
            return final

        # Stage 3: LLM selection (or deterministic top-N fallback)
        if self.llm is not None:
            picks = self._llm_select(
                shortlist, portfolio_tickers, num_picks, risk_profile
            )
        else:
            picks = [entry["ticker"] for entry in shortlist[:num_picks]]
        _notify("llm_picks", picks)

        # Always prepend portfolio tickers so they get researched too
        final = list(portfolio_set)
        for t in picks:
            if t.upper() not in portfolio_set:
                final.append(t)

        _notify("final", final)
        return final

    # ------------------------------------------------------------------
    # Stage 1: Multi-Strategy Quantitative Screen
    # ------------------------------------------------------------------

    def _quantitative_screen(
        self, portfolio_set: set, criteria: Optional[Dict] = None
    ) -> List[str]:
        """Cast a wide net using multiple hedge fund strategies.

        Runs predefined Yahoo screens, custom EquityQuery factor screens,
        and sector-leader scans. Merges, deduplicates, and filters to
        US-listed equities only that exist within the Russell 3000.
        """
        all_tickers: Dict[str, dict] = {}  # ticker -> best quote dict

        # Fetch Russell 3000 universe
        r3000_universe = _fetch_russell_3000()

        # --- 1a. Predefined Yahoo screens (broad coverage) ---
        predefined_screens = [
            "undervalued_growth_stocks",
            "growth_technology_stocks",
            "undervalued_large_caps",
            "day_gainers",  # performing well / momentum
            "day_losers",  # contrarian / bearish / turnaround
            "most_actives",
        ]
        for idx, name in enumerate(predefined_screens):
            try:
                if idx > 0:
                    time.sleep(1.0)  # pace requests to avoid rate limits
                result = screen_stocks(
                    query=name, size=250, sort_field="intradaymarketcap", sort_asc=False
                )
                for q in result.get("data", []):
                    sym = q.get("symbol", "")
                    if sym and sym not in all_tickers:
                        all_tickers[sym] = q
            except Exception as e:
                print(f"Predefined screen '{name}' failed: {e}")

        # --- 1b. Custom factor screens via EquityQuery ---
        # Sorting by varying metrics to extract top 250 segments from different angles
        # per highly-constrained custom query without randomly truncating by ticker name.
        sort_variations = [
            ("intradaymarketcap", False),  # 250 Largest in category
            ("intradaymarketcap", True),  # 250 Smallest in category
            ("percentchange", False),  # 250 Biggest short-term gainers in category
        ]

        for idx, query in enumerate(self._build_custom_queries(criteria)):
            for sort_field, sort_asc in sort_variations:
                try:
                    if all_tickers:
                        time.sleep(1.0)  # pace requests to avoid rate limits
                    result = screen_stocks(
                        query=query, size=250, sort_field=sort_field, sort_asc=sort_asc
                    )
                    for q in result.get("data", []):
                        sym = q.get("symbol", "")
                        if sym and sym not in all_tickers:
                            all_tickers[sym] = q
                except Exception as e:
                    print(f"Custom screen failed: {e}")

        # --- Filter: US-listed only and in Russell 3000 ---
        us_tickers = set()
        for sym, quote in all_tickers.items():
            # Use exchange from quote data if available
            exchange = quote.get("exchange", "") if quote else ""
            if exchange and exchange in _US_EXCHANGES:
                us_tickers.add(sym)
            elif not exchange and "." not in sym:
                # No exchange data but no foreign suffix — assume US
                us_tickers.add(sym)
            # Tickers with "." (e.g. WX8.F) or known non-US exchange → skip

        # Intersect with the Russell 3000 universe to fulfill the user requirement
        if r3000_universe:
            us_tickers = us_tickers.intersection(r3000_universe)

        # Remove portfolio holdings (they'll be added back at the end)
        us_tickers -= portfolio_set

        return sorted(us_tickers)

    def _build_custom_queries(self, criteria: Optional[Dict] = None) -> list:
        """Build EquityQuery objects covering multiple hedge fund strategies.

        Each query enforces US exchange listing and minimum investability
        thresholds (market cap, price, volume).
        """
        queries = []

        # Shared investability floor for all custom screens
        base = [
            yf.EquityQuery("GT", ["intradaymarketcap", 1_000_000_000]),
            yf.EquityQuery("GT", ["eodprice", 5]),
        ]

        # 1. Deep Value — low P/E + high ROE (Buffett-style)
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    *base,
                    yf.EquityQuery("GT", ["peratio.lasttwelvemonths", 0]),
                    yf.EquityQuery("LT", ["peratio.lasttwelvemonths", 15]),
                    yf.EquityQuery("GT", ["returnonequity.lasttwelvemonths", 15]),
                ],
            ),
        )

        # 2. GARP (Growth at Reasonable Price) — moderate P/E + revenue growth
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    *base,
                    yf.EquityQuery("GT", ["peratio.lasttwelvemonths", 0]),
                    yf.EquityQuery("LT", ["peratio.lasttwelvemonths", 25]),
                    yf.EquityQuery("GT", ["quarterlyrevenuegrowth.quarterly", 10]),
                ],
            ),
        )

        # 3. High Growth — explosive revenue with positive margins
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    *base,
                    yf.EquityQuery("GT", ["quarterlyrevenuegrowth.quarterly", 25]),
                    yf.EquityQuery("GT", ["grossprofitmargin.lasttwelvemonths", 20]),
                ],
            ),
        )

        # 4. Quality Compounders — high ROE + high margins (wide moat)
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    *base,
                    yf.EquityQuery("GT", ["returnonequity.lasttwelvemonths", 20]),
                    yf.EquityQuery("GT", ["grossprofitmargin.lasttwelvemonths", 30]),
                ],
            ),
        )

        # 5. Large-Cap Momentum — big companies with positive earnings
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    yf.EquityQuery("GT", ["intradaymarketcap", 10_000_000_000]),
                    yf.EquityQuery("GT", ["eodprice", 10]),
                    yf.EquityQuery("GT", ["peratio.lasttwelvemonths", 0]),
                    yf.EquityQuery("GT", ["returnonequity.lasttwelvemonths", 10]),
                ],
            ),
        )

        # 6. Mid-Cap Opportunities — $2B-$20B sweet spot
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    yf.EquityQuery("GT", ["intradaymarketcap", 2_000_000_000]),
                    yf.EquityQuery("LT", ["intradaymarketcap", 20_000_000_000]),
                    yf.EquityQuery("GT", ["eodprice", 5]),
                    yf.EquityQuery("GT", ["quarterlyrevenuegrowth.quarterly", 10]),
                ],
            ),
        )

        # 7. Bearish / Turnaround / Oversold — High Free Cashflow, dropped in past year
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    *base,
                    yf.EquityQuery("GT", ["unleveredfreecashflow.lasttwelvemonths", 0]),
                    yf.EquityQuery("LT", ["fiftytwowkpercentchange", -10]),
                ],
            ),
        )

        # 8. High Yield Dividend
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    *base,
                    yf.EquityQuery("GT", ["forward_dividend_yield", 4]),
                    yf.EquityQuery("GT", ["returnonequity.lasttwelvemonths", 5]),
                ],
            ),
        )

        # 9. Small-Cap Aggressive Growth
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    yf.EquityQuery("GT", ["intradaymarketcap", 300_000_000]),
                    yf.EquityQuery("LT", ["intradaymarketcap", 2_000_000_000]),
                    yf.EquityQuery("GT", ["eodprice", 2]),
                    yf.EquityQuery("GT", ["quarterlyrevenuegrowth.quarterly", 15]),
                    yf.EquityQuery("LT", ["peratio.lasttwelvemonths", 40]),
                ],
            ),
        )

        # 10. Core Portfolio Anchors (Large, Highly Profitable)
        _try_add(
            queries,
            lambda: yf.EquityQuery(
                "AND",
                [
                    yf.EquityQuery("GT", ["intradaymarketcap", 50_000_000_000]),
                    yf.EquityQuery("GT", ["returnonequity.lasttwelvemonths", 20]),
                    yf.EquityQuery("GT", ["grossprofitmargin.lasttwelvemonths", 40]),
                    yf.EquityQuery("GT", ["ebitdamargin.lasttwelvemonths", 15]),
                ],
            ),
        )

        return queries

    # ------------------------------------------------------------------
    # Stage 2: Multi-Factor Scoring & Ranking
    # ------------------------------------------------------------------

    def _score_and_rank(
        self, candidates: List[str], portfolio_set: set
    ) -> List[Dict[str, Any]]:
        """Score each candidate on a composite of 6 factor families.

        Uses ThreadPoolExecutor to fetch ticker info in parallel to avoid
        sequential API round-trips for hundreds of candidates.

        Returns list of dicts sorted by score (descending):
        [{"ticker": str, "score": float, "info": dict}, ...]
        """
        portfolio_sectors = self._get_portfolio_sectors(portfolio_set)

        def _fetch_and_score(ticker: str):
            try:
                # Suppress noisy HTTP 404/401 errors from yfinance internals
                _yf_logger = logging.getLogger("yfinance")
                _prev_level = _yf_logger.level
                _yf_logger.setLevel(logging.CRITICAL)
                try:
                    info = yf.Ticker(ticker).info
                except Exception as inner_e:
                    err_msg = str(inner_e)
                    if "401" in err_msg or "Crumb" in err_msg:
                        # Crumb expired — wait and retry once
                        time.sleep(2)
                        info = yf.Ticker(ticker).info
                    else:
                        raise
                finally:
                    _yf_logger.setLevel(_prev_level)
                if not info or not info.get("marketCap"):
                    return None

                # Hard filters — reject unsuitable stocks
                market_cap = info.get("marketCap", 0)
                price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                quote_type = info.get("quoteType", "")
                exchange = info.get("exchange", "")

                if market_cap < 500_000_000:
                    return None
                if price < 5:
                    return None
                if quote_type != "EQUITY":
                    return None
                # Double-check US exchange at info level
                if exchange and exchange not in _US_EXCHANGES:
                    return None

                # Enrich with analyst data and growth estimates
                analyst_data = _fetch_analyst_data(ticker)
                growth_data = _fetch_growth_data(ticker)

                score = self._compute_composite_score(
                    info, portfolio_sectors, analyst_data, growth_data
                )
                extracted = _extract_info(info, ticker)
                extracted["analyst_data"] = analyst_data
                extracted["growth_data"] = growth_data
                return {
                    "ticker": ticker,
                    "score": score,
                    "info": extracted,
                }
            except Exception:
                return None

        scored = []
        # Cap workers to reduce rate-limiting and crumb-invalidation risk
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_and_score, t): t for t in candidates}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    scored.append(result)

        scored.sort(key=lambda x: (-x["score"], x["ticker"]))
        return scored

    def _compute_composite_score(
        self,
        info: dict,
        portfolio_sectors: Dict[str, int],
        analyst_data: Optional[Dict] = None,
        growth_data: Optional[Dict] = None,
    ) -> float:
        """Compute a composite score across 7 factor families.

        Factor weights (120 points total):
          Valuation:          20 pts   (P/E, P/B, PEG)
          Profitability:      20 pts   (ROE, profit margin, operating margin)
          Growth:             15 pts   (revenue growth, earnings growth)
          Forward Growth:     10 pts   (from growth estimates — next Qtr/Yr)
          Momentum:           15 pts   (52-week return)
          Financial Health:   10 pts   (current ratio, debt-to-equity)
          Analyst Sentiment:  10 pts   (recommendation mean + price target upside)
          Sector Diversity:   15 pts   (bonus for under-represented sectors)
          Market Cap Tier:     5 pts   (tiebreaker)
        """
        score = 0.0

        # --- Valuation (20 pts) ---
        pe = info.get("trailingPE")
        if pe and 0 < pe < 100:
            score += max(0, 35 - pe) / 35 * 10  # up to 10 pts

        pb = info.get("priceToBook")
        if pb and 0 < pb < 50:
            score += max(0, 8 - pb) / 8 * 5  # up to 5 pts

        peg = info.get("pegRatio")
        if peg and 0 < peg < 5:
            score += max(0, 2 - peg) / 2 * 5  # up to 5 pts

        # --- Profitability (20 pts) ---
        roe = info.get("returnOnEquity")
        if roe and roe > 0:
            score += min(roe * 100, 40) / 40 * 8  # up to 8 pts

        profit_margin = info.get("profitMargins")
        if profit_margin and profit_margin > 0:
            score += min(profit_margin * 100, 40) / 40 * 6  # up to 6 pts

        op_margin = info.get("operatingMargins")
        if op_margin and op_margin > 0:
            score += min(op_margin * 100, 40) / 40 * 6  # up to 6 pts

        # --- Growth (15 pts) ---
        rev_growth = info.get("revenueGrowth")
        if rev_growth and rev_growth > 0:
            score += min(rev_growth * 100, 50) / 50 * 8  # up to 8 pts

        earnings_growth = info.get("earningsGrowth")
        if earnings_growth and earnings_growth > 0:
            score += min(earnings_growth * 100, 50) / 50 * 7  # up to 7 pts

        # --- Forward Growth from growth_estimates (10 pts) ---
        if growth_data:
            fwd_growth = growth_data.get("next_quarter") or growth_data.get("next_year")
            if fwd_growth is not None and fwd_growth > 0:
                score += min(fwd_growth * 100, 50) / 50 * 6  # up to 6 pts
            # Bonus if stock growth beats its sector/industry estimate
            stock_growth = growth_data.get("stock_current_quarter")
            industry_growth = growth_data.get("industry_current_quarter")
            if (
                stock_growth is not None
                and industry_growth is not None
                and industry_growth > 0
                and stock_growth > industry_growth
            ):
                score += 4  # outperforming industry peers

        # --- Momentum (15 pts) ---
        change_52w = info.get("52WeekChange")
        if change_52w is not None:
            clamped = max(min(change_52w * 100, 80), -30)
            score += (clamped + 30) / 110 * 15  # -30%→0pts, +80%→15pts

        # --- Financial Health (10 pts) ---
        current_ratio = info.get("currentRatio")
        if current_ratio and current_ratio > 0:
            if current_ratio < 1.0:
                score += 1
            elif current_ratio <= 3.0:
                score += 5
            else:
                score += 3

        dte = info.get("debtToEquity")
        if dte is not None and dte >= 0:
            if dte < 50:
                score += 5
            elif dte < 100:
                score += 3
            elif dte < 200:
                score += 1

        # --- Analyst Sentiment (10 pts) ---
        rec = info.get("recommendationMean")
        if rec and 1 <= rec <= 5:
            score += max(0, 4 - rec) / 3 * 5  # up to 5 pts

        # Price target upside from analyst data (up to 5 pts)
        if analyst_data:
            upside = analyst_data.get("upside_pct")
            if upside is not None and upside > 0:
                # 30%+ upside → full 5 pts, scaled linearly
                score += min(upside, 30) / 30 * 5

            # Bonus for strong buy consensus (buy_count >> sell_count)
            buy_count = analyst_data.get("buy_count", 0)
            sell_count = analyst_data.get("sell_count", 0)
            total = buy_count + sell_count
            if total > 0 and buy_count / total > 0.7:
                score += 2  # strong buy consensus bonus

        # --- Sector Diversity Bonus (15 pts) ---
        sector = info.get("sector", "Unknown")
        sector_count = portfolio_sectors.get(sector, 0)
        if sector_count == 0:
            score += 15
        elif sector_count == 1:
            score += 7
        elif sector_count == 2:
            score += 2

        # --- Market Cap Tier Bonus (5 pts) ---
        mcap = info.get("marketCap", 0)
        if mcap >= 200_000_000_000:
            score += 5  # mega-cap
        elif mcap >= 50_000_000_000:
            score += 4  # large-cap
        elif mcap >= 10_000_000_000:
            score += 3  # mid-large
        elif mcap >= 2_000_000_000:
            score += 2  # mid-cap
        elif mcap >= 500_000_000:
            score += 1  # small-cap

        return score

    def _get_portfolio_sectors(self, portfolio_set: set) -> Dict[str, int]:
        """Count how many portfolio holdings are in each sector."""
        sector_counts: Dict[str, int] = {}
        for ticker in portfolio_set:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            except Exception:
                continue
        return sector_counts

    # ------------------------------------------------------------------
    # Stage 3: LLM Selection (Risk-Profile Aware)
    # ------------------------------------------------------------------

    def _llm_select(
        self,
        shortlist: List[Dict[str, Any]],
        portfolio_tickers: List[str],
        num_picks: int,
        risk_profile: Optional[Dict] = None,
    ) -> List[str]:
        """Use LLM to select final picks, personalized by risk profile."""

        # Format candidates with enriched data
        candidate_lines = []
        for entry in shortlist:
            info = entry["info"]
            analyst = info.get("analyst_data") or {}
            growth = info.get("growth_data") or {}

            # Build analyst string
            analyst_parts = []
            if info.get("recommendationMean"):
                analyst_parts.append(f"Rec: {_fmt_num(info['recommendationMean'])}")
            if analyst.get("upside_pct") is not None:
                analyst_parts.append(f"Upside: {analyst['upside_pct']:.0f}%")
            if analyst.get("buy_count"):
                analyst_parts.append(
                    f"Buy/Sell: {analyst['buy_count']}/{analyst.get('sell_count', 0)}"
                )
            analyst_str = ", ".join(analyst_parts) if analyst_parts else "N/A"

            # Build forward growth string
            fwd_parts = []
            if growth.get("next_quarter") is not None:
                fwd_parts.append(f"NextQ: {growth['next_quarter'] * 100:.0f}%")
            if growth.get("next_year") is not None:
                fwd_parts.append(f"NextY: {growth['next_year'] * 100:.0f}%")
            fwd_str = ", ".join(fwd_parts) if fwd_parts else "N/A"

            line = (
                f"- {entry['ticker']} ({info.get('name', 'N/A')}) | "
                f"Sector: {info.get('sector', 'N/A')} | "
                f"Industry: {info.get('industry', 'N/A')} | "
                f"Mkt Cap: {_fmt_mcap(info.get('marketCap'))} | "
                f"P/E: {_fmt_num(info.get('trailingPE'))} | "
                f"FwdP/E: {_fmt_num(info.get('forwardPE'))} | "
                f"PEG: {_fmt_num(info.get('pegRatio'))} | "
                f"ROE: {_fmt_pct(info.get('returnOnEquity'))} | "
                f"Margin: {_fmt_pct(info.get('profitMargins'))} | "
                f"Rev Growth: {_fmt_pct(info.get('revenueGrowth'))} | "
                f"52w: {_fmt_pct(info.get('fiftyTwoWeekChangePercent'))} | "
                f"Analyst: [{analyst_str}] | "
                f"Fwd Growth: [{fwd_str}] | "
                f"D/E: {_fmt_num(info.get('debtToEquity'))} | "
                f"Score: {entry['score']:.1f}"
            )
            candidate_lines.append(line)

        portfolio_str = (
            ", ".join(portfolio_tickers)
            if portfolio_tickers
            else "Empty portfolio (no current holdings)"
        )

        # Build risk-profile context for the LLM
        risk_context = ""
        if risk_profile:
            risk_context = (
                "\n**Investor Risk Profile:**\n"
                f"- Experience: {risk_profile.get('experience', 'N/A')}\n"
                f"- Income: {risk_profile.get('income', 'N/A')}\n"
                f"- Net Worth: {risk_profile.get('net_worth', 'N/A')}\n"
                f"- Goal: {risk_profile.get('goal', 'N/A')}\n"
                f"- Risk Tolerance: {risk_profile.get('risk', 'N/A')}\n"
                f"- Investment Horizon: {risk_profile.get('period', 'N/A')}\n\n"
                "IMPORTANT: Tailor your selections to match this investor's profile:\n"
                "- Conservative/short-horizon → favor large-cap, low-volatility, "
                "dividend-paying, value stocks with strong balance sheets.\n"
                "- Aggressive/long-horizon → can include mid/small-caps, high-growth, "
                "momentum plays, and turnaround stories.\n"
                "- Balanced → mix of stable compounders and moderate growth.\n"
            )

        messages = [
            (
                "system",
                "You are a Portfolio Strategist at a top-tier hedge fund. Your job is to "
                f"select exactly {num_picks} NEW stocks from the screened candidates for "
                "deep multi-agent research.\n\n"
                "Think like a portfolio manager building a diversified book:\n"
                "1. **Sector balance** — fill gaps in the current portfolio; avoid over-concentration\n"
                "2. **Factor diversity** — mix value, growth, quality, and momentum names\n"
                "3. **Market cap spread** — blend mega/large-caps (stability) with mid-caps (alpha)\n"
                "4. **Risk/reward** — favor the best composite scores but override when "
                "diversification or the investor's risk profile demands it\n"
                "5. **Catalyst awareness** — prefer stocks where the metrics suggest "
                "an inflection point (accelerating growth, improving margins, attractive PEG)\n"
                f"{risk_context}\n"
                f"You MUST respond with ONLY a JSON array of exactly {num_picks} ticker symbols.\n"
                'Example: ["AAPL", "TSLA", "NVDA", "UNH", "XOM"]\n'
                "No explanation, no markdown, just the JSON array.",
            ),
            (
                "human",
                f"**Current Portfolio:** {portfolio_str}\n\n"
                f"**Screened Candidates (ranked by composite score, top {len(shortlist)}):**\n"
                + "\n".join(candidate_lines)
                + f"\n\nSelect exactly {num_picks} NEW tickers (not already in the portfolio).",
            ),
        ]

        try:
            response = self.llm.invoke(messages).content.strip()
            cleaned = response
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            picks = json.loads(cleaned)
            if isinstance(picks, list) and all(isinstance(t, str) for t in picks):
                return picks[:num_picks]
        except Exception as e:
            print(f"LLM selection failed ({e}), falling back to top-N by score")

        # Fallback: top N by score
        return [entry["ticker"] for entry in shortlist[:num_picks]]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

import os


def _fetch_russell_3000() -> set:
    """Fetch the latest Russell 3000 constituents from iShares IWV ETF."""
    import csv
    import urllib.request
    from datetime import datetime, timedelta

    tickers = set()
    cache_file = os.path.join(os.path.dirname(__file__), "russell_3000_cache.txt")

    # Check cache first (expire after 24 hours)
    if os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mtime < timedelta(hours=24):
            with open(cache_file, "r") as f:
                return set(line.strip() for line in f if line.strip())

    url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode("utf-8")
            lines = content.split("\n")

            # Find the header row that starts with 'Ticker'
            start_idx = -1
            for i, line in enumerate(lines):
                if line.startswith("Ticker") or line.startswith('"Ticker"'):
                    start_idx = i
                    break

            if start_idx != -1:
                # Basic CSV parse to handle quotes
                reader = csv.reader(lines[start_idx:])
                header = next(reader)

                # locate ticker column (usually 0)
                ticker_idx = 0
                for i, col in enumerate(header):
                    if col.strip().lower() == "ticker":
                        ticker_idx = i
                        break

                for row in reader:
                    if len(row) > ticker_idx:
                        sym = row[ticker_idx].strip()
                        # Avoid cash, bonds or missing
                        if sym and sym != "-" and not sym.startswith("BLK"):
                            # Filter obvious currency lines
                            if sym not in ("USD", "CAD", "EUR", "M", "PRO TWD"):
                                tickers.add(sym)

        # Update cache
        if tickers:
            with open(cache_file, "w") as f:
                for t in sorted(tickers):
                    f.write(t + "\n")

    except Exception as e:
        print(f"Failed to fetch Russell 3000 tickers: {e}")
        # If fetch fails, try to load stale cache if it exists
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return set(line.strip() for line in f if line.strip())

    return tickers


def _fetch_analyst_data(ticker: str) -> Optional[Dict]:
    """Fetch analyst recommendations and price targets for scoring.

    Returns a compact dict with buy/sell counts and upside percentage,
    or None if data is unavailable.
    """
    try:
        raw = get_analyst_recommendations(ticker)
        if not isinstance(raw, str) or "Error" in raw:
            return None

        result = {}

        # Parse buy/sell/hold counts from recommendations text
        buy_count = 0
        sell_count = 0
        hold_count = 0
        for line in raw.split("\n"):
            lower = line.lower()
            if "'strongbuy'" in lower or "'buy'" in lower:
                # Extract number from recommendation dict lines
                for part in line.split(","):
                    part = part.strip()
                    if "'strongbuy'" in part.lower() or "'buy'" in part.lower():
                        try:
                            val = int(
                                "".join(c for c in part.split(":")[-1] if c.isdigit())
                            )
                            buy_count += val
                        except (ValueError, IndexError):
                            pass
            if "'sell'" in lower or "'strongsell'" in lower:
                for part in line.split(","):
                    part = part.strip()
                    if "'sell'" in part.lower() or "'strongsell'" in part.lower():
                        try:
                            val = int(
                                "".join(c for c in part.split(":")[-1] if c.isdigit())
                            )
                            sell_count += val
                        except (ValueError, IndexError):
                            pass
            if "'hold'" in lower:
                for part in line.split(","):
                    part = part.strip()
                    if "'hold'" in part.lower():
                        try:
                            val = int(
                                "".join(c for c in part.split(":")[-1] if c.isdigit())
                            )
                            hold_count += val
                        except (ValueError, IndexError):
                            pass

        result["buy_count"] = buy_count
        result["sell_count"] = sell_count
        result["hold_count"] = hold_count

        # Parse price targets for upside calculation
        for line in raw.split("\n"):
            if "mean" in line.lower() or "current" in line.lower():
                try:
                    val = float(
                        "".join(
                            c
                            for c in line.split(":")[-1].strip()
                            if c.isdigit() or c == "."
                        )
                    )
                    if "mean" in line.lower():
                        result["target_mean"] = val
                    elif "current" in line.lower():
                        result["current_price"] = val
                except (ValueError, IndexError):
                    pass

        # Compute upside %
        if (
            "target_mean" in result
            and "current_price" in result
            and result["current_price"] > 0
        ):
            result["upside_pct"] = (
                (result["target_mean"] - result["current_price"])
                / result["current_price"]
                * 100
            )

        return result if result else None
    except Exception:
        return None


def _fetch_growth_data(ticker: str) -> Optional[Dict]:
    """Fetch forward growth estimates for scoring.

    Returns a compact dict with next-quarter, next-year, and
    stock-vs-industry growth comparisons, or None if unavailable.
    """
    try:
        raw = get_growth_estimates(ticker)
        if not isinstance(raw, str) or "Error" in raw:
            return None

        result = {}

        # Parse CSV-style growth estimates output
        # Format: rows like "stock,0.15,0.20,0.18,0.25"
        # Columns are typically: current_qtr, next_qtr, current_year, next_year
        lines = raw.strip().split("\n")
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            row_label = parts[0].lower()

            def _parse_val(idx):
                try:
                    return float(parts[idx])
                except (ValueError, IndexError):
                    return None

            if "stock" in row_label or ticker.lower() in row_label:
                result["stock_current_quarter"] = _parse_val(1)
                result["next_quarter"] = _parse_val(2)
                result["stock_current_year"] = _parse_val(3)
                result["next_year"] = _parse_val(4)
            elif "industry" in row_label:
                result["industry_current_quarter"] = _parse_val(1)
            elif "sector" in row_label:
                result["sector_current_quarter"] = _parse_val(1)

        return result if result else None
    except Exception:
        return None


def _try_add(queries: list, builder):
    """Try to build a query and append it; silently skip on failure."""
    try:
        queries.append(builder())
    except Exception as e:
        print(f"Custom query build failed: {e}")


def _extract_info(info: dict, ticker: str) -> dict:
    """Extract the subset of info fields we carry through scoring."""
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "priceToBook": info.get("priceToBook"),
        "returnOnEquity": info.get("returnOnEquity"),
        "profitMargins": info.get("profitMargins"),
        "operatingMargins": info.get("operatingMargins"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
        "fiftyTwoWeekChangePercent": info.get("52WeekChange"),
        "currentPrice": info.get("currentPrice"),
        "recommendationMean": info.get("recommendationMean"),
        "debtToEquity": info.get("debtToEquity"),
        "currentRatio": info.get("currentRatio"),
        "dividendYield": info.get("dividendYield"),
    }


def _fmt_mcap(val) -> str:
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.1f}T"
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    if val >= 1e6:
        return f"${val / 1e6:.0f}M"
    return f"${val:,.0f}"


def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.1f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(val)
