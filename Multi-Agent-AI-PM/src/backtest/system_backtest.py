"""
System-level regime-aware backtest for the multi-agent trading graph.

Runs the full multi-agent graph on historical dates sampled across market regimes,
then compares predicted mu / conviction against realized forward returns at
short (~10d), medium (~63d), and long (~252d) horizons.

Usage:
    python src/backtest/system_backtest.py --ticker SPY --workers 2 --research-depth shallow
"""

import argparse
import bisect
import json
import os
import pathlib
import random
import statistics
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

# ── Path setup ─────────────────────────────────────────────────────────────────
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)

from src.agents.utils.schemas import CompositeSignal, ResearchReport  # noqa: E402
from src.backtest._proc_utils import (  # noqa: E402
    init_worker_process_group,
    install_executor_cleanup,
)
from src.graph.trading_graph import TradingGraph  # noqa: E402


# ── Regime definitions (label, start, end, samples) ──────────────────────────
DEFAULT_REGIMES: List[Tuple[str, str, str, int]] = [
    ("Post-GFC recovery", "2010-01-01", "2013-12-31", 4),
    ("Mid bull", "2014-01-01", "2017-12-31", 4),
    ("Volatile / trade war", "2018-01-01", "2019-12-31", 4),
    ("COVID crash", "2020-02-15", "2020-03-31", 4),
    ("Recovery bull", "2020-04-01", "2021-12-31", 4),
    ("2022 bear", "2022-01-01", "2022-10-31", 4),
    ("Post-bear bull", "2022-11-01", "2024-12-01", 4),
]

# Forward horizons in trading days (must line up with graph horizons)
HORIZON_DAYS = {
    "short": 10,   # ~2 weeks
    "medium": 63,  # ~3 months
    "long": 252,   # ~1 year
}


@dataclass
class BacktestResult:
    """Single backtest observation."""

    date: str
    regime: str
    ticker: str

    # Composite signal prediction
    pred_mu: float = 0.0
    pred_sigma: float = 0.0
    pred_conviction: float = 0.0
    decision: str = "HOLD"

    # Actual forward returns per horizon
    actual_ret_short: Optional[float] = None
    actual_ret_medium: Optional[float] = None
    actual_ret_long: Optional[float] = None

    # Per-analyst predictions (when reports parse successfully)
    market_mu_long: Optional[float] = None
    market_mu_medium: Optional[float] = None
    market_mu_short: Optional[float] = None
    fundamentals_mu_long: Optional[float] = None
    fundamentals_mu_medium: Optional[float] = None
    fundamentals_mu_short: Optional[float] = None
    news_mu_long: Optional[float] = None
    news_mu_medium: Optional[float] = None
    news_mu_short: Optional[float] = None

    # Error tracking
    error: Optional[str] = None


def sample_dates(
    ticker: str,
    regimes: List[Tuple[str, str, str, int]],
    horizons: List[str],
    random_seed: int = 42,
) -> List[Tuple[str, float, Dict[str, Optional[float]], str]]:
    """
    Sample trade dates from each regime.

    Returns:
        List of (date, start_px, {horizon: end_px}, regime)
    """
    random.seed(random_seed)
    max_forward = max(HORIZON_DAYS[h] for h in horizons)

    print(f"Fetching {ticker} price history ...")
    data = yf.download(
        ticker,
        start="2010-01-01",
        end="2025-01-01",
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError(f"No price data for {ticker}")

    prices = data["Close"].values.flatten()
    dates = data.index.strftime("%Y-%m-%d").tolist()

    results: List[Tuple[str, float, Dict[str, Optional[float]], str]] = []
    for label, r_start, r_end, n_samples in regimes:
        r_start_idx = bisect.bisect_left(dates, r_start)
        r_end_idx = bisect.bisect_right(dates, r_end) - 1
        if r_start_idx > r_end_idx:
            continue

        # Need enough forward history for the longest horizon
        valid_in_regime = [
            i
            for i in range(r_start_idx, r_end_idx + 1)
            if i + max_forward < len(dates)
        ]
        if not valid_in_regime:
            continue

        chosen = random.sample(valid_in_regime, min(n_samples, len(valid_in_regime)))
        for i in chosen:
            start_px = prices[i]
            end_prices = {h: prices[i + HORIZON_DAYS[h]] for h in horizons}
            results.append((dates[i], start_px, end_prices, label))

    print(f"  {len(results)} dates selected across {len(regimes)} regimes\n")
    return results


def _pct_ret(start_px: float, end_px: Optional[float]) -> Optional[float]:
    if end_px is None:
        return None
    return (end_px - start_px) / start_px


def run_single(
    ticker: str,
    trade_date: str,
    start_px: float,
    end_prices: Dict[str, Optional[float]],
    regime: str,
    config: Dict,
) -> Tuple[BacktestResult, Optional[Dict[str, Any]]]:
    """Run the full multi-agent graph for one date.

    Returns:
        (BacktestResult, final_state_dict | None)
    """

    actual_short = _pct_ret(start_px, end_prices.get("short"))
    actual_medium = _pct_ret(start_px, end_prices.get("medium"))
    actual_long = _pct_ret(start_px, end_prices.get("long"))

    try:
        # Each thread gets its own graph instance to avoid shared-state issues
        graph = TradingGraph(
            selected_analysts=["market", "fundamentals", "news"],
            config=config,
            debug=False,
        )
        final_state, decision = graph.propagate(ticker, trade_date)
    except Exception as exc:
        return (
            BacktestResult(
                date=trade_date,
                regime=regime,
                ticker=ticker,
                actual_ret_short=actual_short,
                actual_ret_medium=actual_medium,
                actual_ret_long=actual_long,
                error=str(exc),
            ),
            None,
        )

    # ── Extract composite signal ──
    pred_mu = 0.0
    pred_sigma = 0.0
    pred_conviction = 0.0

    composite_json = final_state.get("composite_signal", "")
    if composite_json:
        try:
            cs = CompositeSignal.model_validate_json(composite_json)
            pred_mu = cs.mu_final
            pred_sigma = cs.sigma_final
            pred_conviction = cs.conviction_final
        except Exception:
            pass  # Keep defaults if parsing fails

    # ── Extract per-analyst predictions ──
    market_mu_long = None
    market_mu_medium = None
    market_mu_short = None
    fundamentals_mu_long = None
    fundamentals_mu_medium = None
    fundamentals_mu_short = None

    market_json = final_state.get("market_report", "")
    if market_json:
        try:
            mr = ResearchReport.model_validate_json(market_json)
            market_mu_long = mr.mu.long_term
            market_mu_medium = mr.mu.medium_term
            market_mu_short = mr.mu.short_term
        except Exception:
            pass

    fund_json = final_state.get("fundamentals_report", "")
    if fund_json:
        try:
            fr = ResearchReport.model_validate_json(fund_json)
            fundamentals_mu_long = fr.mu.long_term
            fundamentals_mu_medium = fr.mu.medium_term
            fundamentals_mu_short = fr.mu.short_term
        except Exception:
            pass

    news_mu_long = None
    news_mu_medium = None
    news_mu_short = None
    news_json = final_state.get("news_report", "")
    if news_json:
        try:
            nr = ResearchReport.model_validate_json(news_json)
            news_mu_long = nr.mu.long_term
            news_mu_medium = nr.mu.medium_term
            news_mu_short = nr.mu.short_term
        except Exception:
            pass

    result = BacktestResult(
        date=trade_date,
        regime=regime,
        ticker=ticker,
        pred_mu=pred_mu,
        pred_sigma=pred_sigma,
        pred_conviction=pred_conviction,
        decision=decision,
        actual_ret_short=actual_short,
        actual_ret_medium=actual_medium,
        actual_ret_long=actual_long,
        market_mu_long=market_mu_long,
        market_mu_medium=market_mu_medium,
        market_mu_short=market_mu_short,
        fundamentals_mu_long=fundamentals_mu_long,
        fundamentals_mu_medium=fundamentals_mu_medium,
        fundamentals_mu_short=fundamentals_mu_short,
        news_mu_long=news_mu_long,
        news_mu_medium=news_mu_medium,
        news_mu_short=news_mu_short,
    )
    return result, final_state


def _run_one_job(args_tuple, config_dict):
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    ticker, trade_date, start_px, end_prices, regime = args_tuple
    try:
        res, final_state = run_single(
            ticker, trade_date, start_px, end_prices, regime, config_dict
        )
    except Exception as exc:
        traceback.print_exc()
        res = BacktestResult(
            date=trade_date,
            regime=regime,
            ticker=ticker,
            error=str(exc),
        )
        final_state = None
    return res, final_state


def _report_batch(label: str, rows: List[BacktestResult], horizon: str = "long"):
    """Print metrics for a batch of results at a given horizon."""
    actual_key = f"actual_ret_{horizon}"
    actuals = [
        getattr(r, actual_key) for r in rows if getattr(r, actual_key) is not None
    ]
    if not actuals:
        return

    n = len(actuals)
    dir_correct = sum(
        1
        for r in rows
        if getattr(r, actual_key) is not None
        and (r.pred_mu > 0) == (getattr(r, actual_key) > 0)
    ) / n

    mae = statistics.mean(
        abs(r.pred_mu - getattr(r, actual_key))
        for r in rows
        if getattr(r, actual_key) is not None
    )
    mse = statistics.mean(
        (r.pred_mu - getattr(r, actual_key)) ** 2
        for r in rows
        if getattr(r, actual_key) is not None
    )
    avg_conv = statistics.mean(r.pred_conviction for r in rows)
    avg_sigma = statistics.mean(r.pred_sigma for r in rows)

    # Direction accuracy by signal type
    buy_rows = [
        r for r in rows if r.decision == "BUY" and getattr(r, actual_key) is not None
    ]
    sell_rows = [
        r
        for r in rows
        if r.decision == "SELL" and getattr(r, actual_key) is not None
    ]
    hold_rows = [
        r
        for r in rows
        if r.decision == "HOLD" and getattr(r, actual_key) is not None
    ]

    buy_acc = (
        sum(1 for r in buy_rows if (r.pred_mu > 0) == (getattr(r, actual_key) > 0))
        / len(buy_rows)
        if buy_rows
        else 0.0
    )
    sell_acc = (
        sum(1 for r in sell_rows if (r.pred_mu > 0) == (getattr(r, actual_key) > 0))
        / len(sell_rows)
        if sell_rows
        else 0.0
    )
    # "Flatness" for HOLD: actual return within +/- 5%
    hold_flat = (
        sum(1 for r in hold_rows if abs(getattr(r, actual_key)) < 0.05) / len(hold_rows)
        if hold_rows
        else 0.0
    )

    print(f"  {label} (n={n}, horizon={horizon})")
    print(f"    directional accuracy   = {dir_correct:.2%}")
    print(f"    MAE                    = {mae:.6f}")
    print(f"    MSE                    = {mse:.6f}")
    print(f"    RMSE                   = {mse**0.5:.6f}")
    print(f"    avg sigma              = {avg_sigma:.6f}")
    print(f"    avg conviction         = {avg_conv:.4f}")
    print(f"    BUY  accuracy ({len(buy_rows)})  = {buy_acc:.2%}")
    print(f"    SELL accuracy ({len(sell_rows)}) = {sell_acc:.2%}")
    print(f"    HOLD flatness ({len(hold_rows)}) = {hold_flat:.2%}")
    print()


def report(results: List[BacktestResult]):
    """Print aggregate backtest report."""
    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]

    print(f"\n{'=' * 70}")
    print(f"  SYSTEM BACKTEST REPORT  ({len(successes)}/{len(results)} successful)")
    print(f"{'=' * 70}\n")

    if failures:
        print(f"  Failures ({len(failures)}):")
        for f in failures[:5]:
            print(f"    {f.date} — {f.error[:80]}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")
        print()

    if not successes:
        print("  No successful results to report.")
        return

    # ── Overall ──
    for h in ["long", "medium", "short"]:
        _report_batch("Overall", successes, h)

    # ── Per-regime ──
    regimes = sorted({r.regime for r in successes})
    for reg in regimes:
        subset = [r for r in successes if r.regime == reg]
        for h in ["long", "medium", "short"]:
            _report_batch(reg, subset, h)

    # ── Per-analyst accuracy (long horizon) ──
    print("  PER-ANALYST LONG-HORIZON ACCURACY")
    print("  " + "-" * 66)
    for analyst_label, mu_attr in [
        ("Market", "market_mu_long"),
        ("Fundamentals", "fundamentals_mu_long"),
        ("News", "news_mu_long"),
    ]:
        valid = [
            r for r in successes
            if getattr(r, mu_attr) is not None and r.actual_ret_long is not None
        ]
        if not valid:
            continue
        acc = sum(
            1 for r in valid if (getattr(r, mu_attr) > 0) == (r.actual_ret_long > 0)
        ) / len(valid)
        mae = statistics.mean(
            abs(getattr(r, mu_attr) - r.actual_ret_long) for r in valid
        )
        print(
            f"    {analyst_label:14s}  dir_acc={acc:.2%}  MAE={mae:.6f}  n={len(valid)}"
        )
    print()

    # ── Conviction split (long horizon) ──
    sorted_by_conv = sorted(successes, key=lambda r: r.pred_conviction, reverse=True)
    mid = len(sorted_by_conv) // 2
    if mid > 0:
        _report_batch("High conviction", sorted_by_conv[:mid], "long")
        _report_batch("Low conviction", sorted_by_conv[mid:], "long")

    # ── Decision distribution ──
    print("  DECISION DISTRIBUTION")
    print("  " + "-" * 66)
    for dec in ["BUY", "SELL", "HOLD"]:
        subset = [r for r in successes if r.decision == dec and r.actual_ret_long is not None]
        if not subset:
            continue
        avg_actual = statistics.mean(r.actual_ret_long for r in subset)
        print(f"    {dec:4s}  n={len(subset):3d}  avg_actual_ret={avg_actual:+.6f}")
    print()

    print(f"{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


def _save_state(state: Dict[str, Any], path: str) -> None:
    """Serialize a graph final_state to JSON."""
    serializable = {}
    for k, v in state.items():
        if k == "messages":
            serializable[k] = [
                {"type": m.type, "content": m.content}
                if hasattr(m, "type") and hasattr(m, "content")
                else str(m)
                for m in v
            ]
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def _save_csv(results: List[BacktestResult], path: str) -> None:
    """Write a CSV summary of backtest results."""
    import csv

    fieldnames = [
        "date", "regime", "ticker", "decision",
        "pred_mu", "pred_sigma", "pred_conviction",
        "actual_ret_short", "actual_ret_medium", "actual_ret_long",
        "market_mu_long", "market_mu_medium", "market_mu_short",
        "fundamentals_mu_long", "fundamentals_mu_medium", "fundamentals_mu_short",
        "news_mu_long", "news_mu_medium", "news_mu_short",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def _capture_report_text(results: List[BacktestResult]) -> str:
    """Capture the printed report as a string for saving."""
    import io

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        report(results)
    finally:
        text = sys.stdout.getvalue()
        sys.stdout = old_stdout
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Backtest the full multi-agent trading graph across market regimes."
    )
    parser.add_argument("--ticker", default="SPY", help="Comma-separated tickers to backtest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers", type=int, default=1, help="Max concurrent graph runs"
    )
    parser.add_argument(
        "--research-depth",
        default="shallow",
        choices=["shallow", "medium", "deep"],
        help="Research depth passed to each analyst",
    )
    parser.add_argument(
        "--llm-provider", default="ollama", help="LLM provider (e.g. ollama, openai)"
    )
    parser.add_argument("--llm-model", default=None, help="Override both deep/quick LLMs")
    parser.add_argument(
        "--code-agent-model", default=None, help="Override code-agent model"
    )
    parser.add_argument(
        "--results-dir", default="./backtest_results", help="Output directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce samples per regime by half for a fast smoke test",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["short", "medium", "long"],
        choices=["short", "medium", "long"],
        help="Which forward-return horizons to evaluate (default: all)",
    )
    args = parser.parse_args()

    # ── Build config (merge overrides on top of DEFAULT_CONFIG) ──
    from src.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["research_depth"] = args.research_depth
    config["rate_limit_rpm"] = None  # Local backtest — disable rate limiting
    if args.llm_model:
        config["deep_think_llm"] = args.llm_model
        config["quick_think_llm"] = args.llm_model
    if args.code_agent_model:
        config["code_agent_model"] = args.code_agent_model

    print(f"Backtest config: {json.dumps(config, indent=2)}\n")

    # ── Adjust regimes for quick mode ──
    regimes = DEFAULT_REGIMES.copy()
    if args.quick:
        regimes = [(label, s, e, max(1, n // 2)) for label, s, e, n in regimes]
        print("QUICK MODE: sample counts halved\n")

    # ── Parse tickers ──
    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    print(f"Tickers: {tickers}\n")

    # ── Sample dates ──
    all_dates: List[Tuple[str, str, float, Dict[str, Optional[float]], str]] = []
    for ticker in tickers:
        dates = sample_dates(ticker, regimes, args.horizons, random_seed=args.seed)
        for trade_date, start_px, end_prices, regime in dates:
            all_dates.append((ticker, trade_date, start_px, end_prices, regime))

    if not all_dates:
        print("No dates sampled. Exiting.")
        return

    print(f"Total dates to run: {len(all_dates)}\n")

    # ── Prepare run directory (creation deferred until first write) ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tickers_tag = "_".join(tickers)
    run_dir = os.path.join(args.results_dir, f"run_{tickers_tag}_{timestamp}")
    states_dir = os.path.join(run_dir, "states")

    # ── Run backtest ──
    results: List[BacktestResult] = []
    print(f"Running backtest with max {args.workers} concurrent workers ...")
    futures: List[Any] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker_process_group,
    ) as executor:
        install_executor_cleanup(executor)
        for d in all_dates:
            futures.append(executor.submit(_run_one_job, d, config))

    for fut in futures:
        res, final_state = fut.result()
        # Persist full graph state when available
        if final_state is not None:
            if not os.path.isdir(states_dir):
                os.makedirs(states_dir, exist_ok=True)
            state_path = os.path.join(states_dir, f"state_{res.ticker}_{res.date}.json")
            try:
                _save_state(final_state, state_path)
            except Exception as e:
                print(f"    WARN: could not save state for {res.ticker} {res.date}: {e}")
        results.append(res)
        status = "OK" if res.error is None else f"ERR: {res.error[:40]}"
        print(
            f"  [{res.regime}] {res.ticker} {res.date}  "
            f"mu={res.pred_mu:+.4f}  dec={res.decision}  {status}"
        )

    # ── Report ──
    report(results)

    # ── Save all artifacts ──
    os.makedirs(run_dir, exist_ok=True)

    # 1. JSON summary (also save flat to results_dir for parse_results.py)
    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(args.results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, default=str)
    # Also save inside run_dir for record-keeping
    json_path_run = os.path.join(run_dir, "summary.json")
    with open(json_path_run, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2, default=str)

    # 2. CSV summary
    csv_path = os.path.join(run_dir, "summary.csv")
    _save_csv(results, csv_path)

    # 3. Text report
    report_path = os.path.join(run_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(_capture_report_text(results))

    print(f"\nAll artifacts saved to: {run_dir}")
    print("  summary.json   – structured results")
    print("  summary.csv    – spreadsheet-friendly")
    print("  report.txt     – human-readable report")
    print("  states/        – full graph final_state per date")


if __name__ == "__main__":
    main()
