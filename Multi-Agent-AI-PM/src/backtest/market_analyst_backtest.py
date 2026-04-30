"""
Market Analyst backtest — regime-aware, parallelised directional accuracy test.

Samples historical dates for a basket of large-cap stocks across market regimes,
runs the market analyst in parallel, and scores predicted mu.long_term sign
against realised 252-day forward returns.

Usage:
    python -m src.backtest.market_analyst_backtest --workers 4 --depth shallow
"""

from __future__ import annotations

import argparse
import bisect
import pathlib
import random
import statistics
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

import yfinance as yf

# ── Path setup ─────────────────────────────────────────────────────────────────
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.llm_clients import create_llm_client
from src.agents.code_agent.code_agent import CodeValidationAgent
from src.agents.analysts.market_analyst import create_market_analyst
from src.agents.utils.schemas import ResearchReport

# ── Default config ───────────────────────────────────────────────────────────────
DEFAULT_REGIMES = [
    ("Post-GFC recovery", "2010-01-01", "2013-12-31", 3),
    ("Mid bull", "2014-01-01", "2017-12-31", 3),
    ("Volatile / trade war", "2018-01-01", "2019-12-31", 2),
    ("COVID crash", "2020-02-15", "2020-03-31", 2),
    ("Recovery bull", "2020-04-01", "2021-12-31", 3),
    ("2022 bear", "2022-01-01", "2022-10-31", 2),
    ("Post-bear bull", "2022-11-01", "2024-12-01", 1),
]

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "WMT"]

# ── Thread-local analyst nodes (avoid shared-state in CodeValidationAgent) ─────
_thread_local = threading.local()


def _get_node(depth: str):
    """Return a thread-local market-analyst node."""
    key = f"market_node_{depth}"
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
        ca = CodeValidationAgent(
            model="minimax-m2.7:cloud",
            timeout=300,
            max_iterations=8,
            analyst_type="market",
            project_root=_project_root,
            verbose=False,
        )
        setattr(
            _thread_local,
            key,
            create_market_analyst(
                llm,
                ca,
                depth,
                active_horizons=("long_term", "medium_term", "short_term"),
            ),
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
        rep = ResearchReport.model_validate_json(r.get("market_report", "{}"))
        pred_l = rep.mu.long_term
        pred_m = rep.mu.medium_term
        pred_s = rep.mu.short_term
        sigma_l = rep.sigma_contribution.long_term
        sigma_m = rep.sigma_contribution.medium_term
        sigma_s = rep.sigma_contribution.short_term
        conv_l = rep.conviction.long_term
        conv_m = rep.conviction.medium_term
        conv_s = rep.conviction.short_term
        correct = (pred_l > 0) == (actual > 0)
        return {
            "ticker": ticker,
            "date": d,
            "regime": reg,
            "pred": pred_l,
            "pred_l": pred_l,
            "pred_m": pred_m,
            "pred_s": pred_s,
            "actual": actual,
            "sigma_l": sigma_l,
            "sigma_m": sigma_m,
            "sigma_s": sigma_s,
            "conviction_l": conv_l,
            "conviction_m": conv_m,
            "conviction_s": conv_s,
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
        avg_conv_l = statistics.mean(r["conviction_l"] for r in rows)
        avg_conv_m = statistics.mean(r["conviction_m"] for r in rows)
        avg_conv_s = statistics.mean(r["conviction_s"] for r in rows)
        avg_mu_l = statistics.mean(r["pred_l"] for r in rows)
        avg_mu_m = statistics.mean(r["pred_m"] for r in rows)
        avg_mu_s = statistics.mean(r["pred_s"] for r in rows)
        print(
            f"  {label:30s}  n={len(rows):2d}  acc={acc:.1%}  "
            f"mae={mae:.4f}  rmse={rmse:.4f}\n"
            f"    mu=[L={avg_mu_l:+.4f} M={avg_mu_m:+.4f} S={avg_mu_s:+.4f}]  "
            f"conv=[L={avg_conv_l:.2f} M={avg_conv_m:.2f} S={avg_conv_s:.2f}]"
        )
    print(f"{'=' * 60}\nDone.")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Market analyst backtest")
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
    parser.add_argument(
        "--output",
        default="experiment_results/backtest_market.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    regimes = DEFAULT_REGIMES.copy()
    if args.quick:
        regimes = [(l, s, e, max(1, n // 2)) for l, s, e, n in regimes]

    print(
        f"MARKET ANALYST BACKTEST — {len(args.tickers)} tickers  "
        f"depth={args.depth}  workers={args.workers}\n"
    )
    test_dates = sample_dates(args.tickers, regimes, args.forward_days, args.seed)
    print(f"Sampled {len(test_dates)} dates across {len(regimes)} regimes\n")

    results: list[dict] = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_once, td, args.depth): td for td in test_dates}
        for fut in as_completed(futures):
            res = fut.result()
            if res["ok"]:
                results.append(res)
                print(
                    f"  [{res['regime']}] {res['ticker']} {res['date']}  "
                    f"act={res['actual']:+.4f}  pred={res['pred']:+.4f}  "
                    f"sigma={res['sigma_l']:.4f}  conv={res['conviction_l']:.2f}  "
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

    # ── Save structured results ────────────────────────────────────────────
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "test_type": "backtest",
        "analyst": "market",
        "tickers": args.tickers,
        "forward_days": args.forward_days,
        "depth": args.depth,
        "seed": args.seed,
        "workers": args.workers,
        "quick": args.quick,
        "dates_tested": len(test_dates),
        "dates_successful": len(results),
        "errors": errors,
        "regimes": [r[0] for r in regimes],
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
