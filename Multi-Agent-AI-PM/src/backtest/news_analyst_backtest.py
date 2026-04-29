"""
News Analyst backtest — regime-aware, parallelised directional accuracy test.

Samples historical dates for a basket of large-cap stocks across market regimes,
runs the news analyst in parallel, and scores predicted mu.long_term sign
against realised 252-day forward returns.

Usage:
    python -m src.backtest.news_analyst_backtest --workers 4 --depth shallow
"""

from __future__ import annotations

import argparse
import bisect
import pathlib
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

# ── Path setup ─────────────────────────────────────────────────────────────────
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.llm_clients import create_llm_client
from src.agents.analysts.news_analyst import create_news_analyst
from src.agents.utils.schemas import ResearchReport

# ── Default config ───────────────────────────────────────────────────────────────
DEFAULT_REGIMES = [
    ("Post-GFC recovery", "2010-01-01", "2013-12-31", 2),
    ("Mid bull", "2014-01-01", "2017-12-31", 2),
    ("Volatile / trade war", "2018-01-01", "2019-12-31", 2),
    ("COVID crash", "2020-02-15", "2020-03-31", 1),
    ("Recovery bull", "2020-04-01", "2021-12-31", 2),
    ("2022 bear", "2022-01-01", "2022-10-31", 2),
    ("Post-bear bull", "2022-11-01", "2024-12-01", 2),
]

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "WMT"]

# ── Thread-local analyst nodes ─────────────────────────────────────────────────
_thread_local = threading.local()


def _get_node(depth: str):
    """Return a thread-local news-analyst node."""
    key = f"news_node_{depth}"
    if not hasattr(_thread_local, key):
        llm = create_llm_client(
            provider="ollama",
            model="minimax-m2.7:cloud",
            base_url="http://localhost:11434/v1",
            max_retries=10,
            reasoning_effort="high",
            temperature=0,
            seed=42,
        ).get_llm()
        setattr(
            _thread_local, key, create_news_analyst(llm, depth, horizons=("long_term",))
        )
    return getattr(_thread_local, key)


# ── Data sampling ──────────────────────────────────────────────────────────────


def sample_dates(
    tickers: list[str],
    regimes,
    forward_days: int = 252,
    seed: int = 42,
):
    """Download price history for each ticker and sample trade dates per regime."""
    random.seed(seed)

    all_test_dates = []
    for ticker in tickers:
        data = yf.download(
            ticker,
            start="2010-01-01",
            end="2024-12-31",
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            print(f"  WARN: no data for {ticker}, skipping")
            continue

        prices = data["Close"].values.flatten()
        dates_list = data.index.strftime("%Y-%m-%d").tolist()

        for label, r_start, r_end, n in regimes:
            lo = bisect.bisect_left(dates_list, r_start)
            hi = bisect.bisect_right(dates_list, r_end) - 1
            valid = [i for i in range(lo, hi + 1) if i + forward_days < len(dates_list)]
            if not valid:
                continue
            chosen = random.sample(valid, min(n, len(valid)))
            for i in chosen:
                all_test_dates.append(
                    (
                        ticker,
                        dates_list[i],
                        float(prices[i]),
                        float(prices[i + forward_days]),
                        label,
                    )
                )

    return all_test_dates


# ── Single run ─────────────────────────────────────────────────────────────────


def _run_once(args: tuple, depth: str) -> dict:
    ticker, d, sp, ep, reg = args
    actual = (ep - sp) / sp
    try:
        node = _get_node(depth)
        r = node({"company_of_interest": ticker, "trade_date": d})
        rep = ResearchReport.model_validate_json(r.get("news_report", "{}"))
        pred = rep.mu.long_term
        sigma = rep.sigma_contribution.long_term
        conv = rep.conviction.long_term
        correct = (pred > 0) == (actual > 0)
        return {
            "ticker": ticker,
            "date": d,
            "regime": reg,
            "pred": pred,
            "actual": actual,
            "sigma": sigma,
            "conviction": conv,
            "correct": correct,
            "ok": True,
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "date": d,
            "regime": reg,
            "error": str(exc),
            "ok": False,
        }


# ── Reporting ──────────────────────────────────────────────────────────────────


def _running_summary(results: list[dict]) -> str:
    n = len(results)
    acc = sum(r["correct"] for r in results) / n
    mae = statistics.mean(abs(r["pred"] - r["actual"]) for r in results)
    return f"n={n}  acc={acc:.1%}  mae={mae:.4f}"


def _print_aggregate(results: list[dict]):
    if not results:
        print("No successful results.")
        return

    print(f"\n{'=' * 60}\n  AGGREGATE (n={len(results)})\n{'=' * 60}")
    batches = [("Overall", results)] + [
        (reg, [r for r in results if r["regime"] == reg])
        for reg in sorted({r["regime"] for r in results})
    ]
    for label, rows in batches:
        if not rows:
            continue
        acc = sum(r["correct"] for r in rows) / len(rows)
        mae = statistics.mean(abs(r["pred"] - r["actual"]) for r in rows)
        rmse = statistics.mean((r["pred"] - r["actual"]) ** 2 for r in rows) ** 0.5
        avg_conv = statistics.mean(r["conviction"] for r in rows)
        print(
            f"  {label:30s}  n={len(rows):2d}  acc={acc:.1%}  "
            f"mae={mae:.4f}  rmse={rmse:.4f}  avg_conv={avg_conv:.2f}"
        )
    print(f"{'=' * 60}\nDone.")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="News analyst backtest")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS, help="Stock tickers"
    )
    parser.add_argument(
        "--forward-days", type=int, default=252, help="Forward return horizon"
    )
    parser.add_argument(
        "--depth", default="shallow", choices=["shallow", "medium", "deep"]
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent workers")
    parser.add_argument("--quick", action="store_true", help="Halve samples per regime")
    args = parser.parse_args()

    regimes = DEFAULT_REGIMES.copy()
    if args.quick:
        regimes = [(l, s, e, max(1, n // 2)) for l, s, e, n in regimes]

    print(
        f"NEWS ANALYST BACKTEST — {len(args.tickers)} tickers  "
        f"depth={args.depth}  workers={args.workers}\n"
    )
    test_dates = sample_dates(args.tickers, regimes, args.forward_days, args.seed)
    print(f"Sampled {len(test_dates)} dates across {len(regimes)} regimes\n")

    results: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_once, td, args.depth): td for td in test_dates}
        for fut in as_completed(futures):
            res = fut.result()
            if res["ok"]:
                results.append(res)
                print(
                    f"  [{res['regime']}] {res['ticker']} {res['date']}  "
                    f"act={res['actual']:+.4f}  pred={res['pred']:+.4f}  "
                    f"sigma={res['sigma']:.4f}  conv={res['conviction']:.2f}  "
                    f"correct={res['correct']}  |  {_running_summary(results)}"
                )
            else:
                errors += 1
                print(
                    f"  [{res['regime']}] {res['ticker']} {res['date']}  ERROR: {res['error']}"
                )

    if errors:
        print(f"\n  Errors: {errors}/{len(test_dates)}")
    if not results:
        raise SystemExit(1)

    _print_aggregate(results)


if __name__ == "__main__":
    main()
