"""
Trader (Synthesis) deviation test — run the three analysts once for real,
then reuse their reports across N synthesis runs and report statistics.

This isolates synthesis-agent non-determinism from analyst non-determinism.

Usage:
    python -m src.backtest.trader_deviation_test --ticker AAPL --date 2025-03-14 --runs 20 --workers 4 --seed 42
    python -m src.backtest.trader_deviation_test --ticker SPY --date 2025-03-14 --horizons long,medium --runs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ── Ensure project root is on path ────────────────────────────────────────────
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.llm_clients import create_llm_client
from src.agents.trader.trader import create_synthesis_agent
from src.agents.utils.schemas import (
    AgentType,
    Catalyst,
    Citation,
    CompositeSignal,
    ComputedMetric,
    HorizonThesis,
    HorizonTraceIds,
    HorizonValues,
    ResearchReport,
    Risk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("trader_deviation_test")

_ALL_HORIZONS = ("long_term", "medium_term", "short_term")


def _parse_horizons(arg: str | None) -> tuple[str, ...]:
    """Parse comma-separated horizon string into a canonical tuple."""
    if not arg:
        return _ALL_HORIZONS
    mapping = {
        "long": "long_term",
        "medium": "medium_term",
        "short": "short_term",
    }
    parts = [p.strip().lower() for p in arg.split(",")]
    parsed = []
    for p in parts:
        canonical = mapping.get(p, p)
        if canonical not in _ALL_HORIZONS:
            raise ValueError(f"Invalid horizon: {p!r}")
        parsed.append(canonical)
    return tuple(parsed)


def _build_mock_report(
    agent_type: AgentType,
    ticker: str,
    trade_date: str,
    mu: dict[str, float],
    sigma: dict[str, float],
    conviction: dict[str, float],
    thesis: dict[str, str],
) -> ResearchReport:
    """Build a deterministic mock ResearchReport as fallback."""
    return ResearchReport(
        ticker=ticker,
        agent_type=agent_type,
        timestamp=datetime.strptime(trade_date, "%Y-%m-%d"),
        metrics_selected=[],
        mu=HorizonValues(
            long_term=mu.get("long_term", 0.0),
            medium_term=mu.get("medium_term", 0.0),
            short_term=mu.get("short_term", 0.0),
        ),
        mu_trace_id=HorizonTraceIds(
            long_term=f"mu-trace-{agent_type.value}-lt",
            medium_term=f"mu-trace-{agent_type.value}-mt",
            short_term=f"mu-trace-{agent_type.value}-st",
        ),
        sigma_contribution=HorizonValues(
            long_term=sigma.get("long_term", 0.01),
            medium_term=sigma.get("medium_term", 0.01),
            short_term=sigma.get("short_term", 0.01),
        ),
        sigma_trace_id=HorizonTraceIds(
            long_term=f"sigma-trace-{agent_type.value}-lt",
            medium_term=f"sigma-trace-{agent_type.value}-mt",
            short_term=f"sigma-trace-{agent_type.value}-st",
        ),
        computed_metrics=[
            ComputedMetric(
                metric_name="mock_metric",
                term="long_term",
                value=0.0,
                metric_interpretation="Mock metric for deviation testing",
                value_interpretation="Neutral",
                computation_trace_id=f"mock-trace-{agent_type.value}",
            )
        ],
        investment_thesis=HorizonThesis(
            long_term=thesis.get("long_term", f"{agent_type.value} long thesis."),
            medium_term=thesis.get("medium_term", f"{agent_type.value} medium thesis."),
            short_term=thesis.get("short_term", f"{agent_type.value} short thesis."),
        ),
        conviction=HorizonValues(
            long_term=conviction.get("long_term", 0.5),
            medium_term=conviction.get("medium_term", 0.5),
            short_term=conviction.get("short_term", 0.5),
        ),
        key_catalysts=[
            Catalyst(
                catalyst=f"{agent_type.value} catalyst",
                term="long_term",
                metric_name="mock_metric",
                computation_trace_id=f"mock-trace-{agent_type.value}",
                claim="Mock claim",
                source_uuid=f"mock-{agent_type.value}-uuid",
            )
        ],
        key_risks=[
            Risk(
                risk=f"{agent_type.value} risk",
                term="long_term",
                metric_name="mock_metric",
                computation_trace_id=f"mock-trace-{agent_type.value}",
                claim="Mock risk claim",
                source_uuid=f"mock-{agent_type.value}-uuid",
            )
        ],
        source_uuids=[f"mock-{agent_type.value}-uuid"],
        computation_traces=[],
        citation_chain=[
            Citation(
                claim="Mock citation",
                source="Mock source",
                source_uuid=f"mock-{agent_type.value}-uuid",
            )
        ],
        contributing_factors=[],
    )


def _build_mock_state(ticker: str, trade_date: str) -> dict:
    """Fallback state with deterministic mock reports (all three analysts)."""
    market = _build_mock_report(
        AgentType.MARKET,
        ticker,
        trade_date,
        mu={"long_term": 0.3147, "medium_term": 0.0916, "short_term": 0.0200},
        sigma={"long_term": 0.3099, "medium_term": 0.3099, "short_term": 0.3164},
        conviction={"long_term": 0.94, "medium_term": 0.96, "short_term": 0.99},
        thesis={
            "long_term": "AAPL presents a uniformly bullish long-term outlook anchored by mu=0.3147, with signal_concordance=1.0 across all 8 empirical signals confirming historical precedent. The ema_cross_signal=0.472 and sma_cross_signal=0.366 (50-SMA/200-SMA golden cross regime) suggest sustained structural uptrend. XLK momentum=+8.73% provides sector corroboration, while VIX=16.89 reflects benign macro backdrop. Risks include sigma=0.31 (30.99% vol) amplified by Beta=1.109 and P/B=45.24 suggesting valuation premium. The 255-signal k-NN framework provides robust statistical grounding, though signal_dispersion=0.071 (low) warrants monitoring for regime shifts.",
            "medium_term": "AAPL's medium-term outlook is cautiously bullish with mu=0.0916 (9.16% annualised expected return) derived from unanimous signal_concordance=1.0. Signal strength is moderate (range 0.024-0.154) with ema_cross_signal=0.154 and price_50sma_signal=0.144 strongest. The sma_cross_signal=0.024 (weakest) and macdh_signal=0.066 suggest momentum is softening versus long-term view. Earnings at Jan 2026 (10 months away) eliminates near-term event risk. With signal_dispersion=0.042, signals are tightly clustered, confirming consensus but with less conviction than long-term.",
            "short_term": "AAPL's short-term outlook (12-day horizon) shows mu=0.020 (2.0% annualised return) with signal_concordance=1.0 confirming unanimous bullish signal. However, absolute signal magnitudes are minimal (0.006-0.028), indicating near-neutral short-term expectation. The price_50sma_signal=0.028 and ema_cross_signal=0.020 provide marginal support. With sigma=0.316 (31.6% annualised), short-term risk/reward is unfavorable. signal_dispersion=0.0075 (tightest across all horizons) confirms consensus but reflects low conviction. AAPL near 52-week high (288.62 vs ~230 current) suggests limited immediate upside. VIX=16.89 confirms low volatility backdrop favorable for stable prices.",
        },
    )

    fundamental = _build_mock_report(
        AgentType.FUNDAMENTAL,
        ticker,
        trade_date,
        mu={"long_term": -0.2410, "medium_term": -0.1807, "short_term": -0.1205},
        sigma={"long_term": 1.2858, "medium_term": 1.0929, "short_term": 0.9000},
        conviction={"long_term": 0.87, "medium_term": 0.17, "short_term": 0.13},
        thesis={
            "long_term": "[LONG TERM] Apple Inc. presents a challenging long-term outlook driven by strong model disagreement on valuation direction but consensus on negative near-term returns. The composite mu of -24.1% reflects deeply negative expected returns across 255 signal combinations. The multiples_implied_price of $290.41 suggests 7% upside potential, while the residual_income_value of $11.06 indicates 96% overvaluation on residual income basis. Signal concordance of 0.86 shows high model agreement on negative direction despite 0.73 dispersion. Quality_score of 0.5 is neutral while financial_health_score of 0.96 reflects excellent balance sheet. Growth_score of -0.612 signals secular deceleration. As a high-beta tech name (Beta 1.109) with modest 0.38% dividend yield and $3.98T market cap, Apple's wide model dispersion is expected. The medium_term mu of -18.1% and short_term mu of -12.0% show negative expected returns across all horizons, with decreasing severity at shorter terms.",
            "medium_term": "[MEDIUM TERM] Apple Inc. presents a challenging long-term outlook driven by strong model disagreement on valuation direction but consensus on negative near-term returns. The composite mu of -24.1% reflects deeply negative expected returns across 255 signal combinations. The multiples_implied_price of $290.41 suggests 7% upside potential, while the residual_income_value of $11.06 indicates 96% overvaluation on residual income basis. Signal concordance of 0.86 shows high model agreement on negative direction despite 0.73 dispersion. Quality_score of 0.5 is neutral while financial_health_score of 0.96 reflects excellent balance sheet. Growth_score of -0.612 signals secular deceleration. As a high-beta tech name (Beta 1.109) with modest 0.38% dividend yield and $3.98T market cap, Apple's wide model dispersion is expected. The medium_term mu of -18.1% and short_term mu of -12.0% show negative expected returns across all horizons, with decreasing severity at shorter terms.",
            "short_term": "[SHORT TERM] Apple Inc. presents a challenging long-term outlook driven by strong model disagreement on valuation direction but consensus on negative near-term returns. The composite mu of -24.1% reflects deeply negative expected returns across 255 signal combinations. The multiples_implied_price of $290.41 suggests 7% upside potential, while the residual_income_value of $11.06 indicates 96% overvaluation on residual income basis. Signal concordance of 0.86 shows high model agreement on negative direction despite 0.73 dispersion. Quality_score of 0.5 is neutral while financial_health_score of 0.96 reflects excellent balance sheet. Growth_score of -0.612 signals secular deceleration. As a high-beta tech name (Beta 1.109) with modest 0.38% dividend yield and $3.98T market cap, Apple's wide model dispersion is expected. The medium_term mu of -18.1% and short_term mu of -12.0% show negative expected returns across all horizons, with decreasing severity at shorter terms.",
        },
    )

    news = _build_mock_report(
        AgentType.NEWS,
        ticker,
        trade_date,
        mu={"long_term": 0.10, "medium_term": -0.05, "short_term": -0.15},
        sigma={"long_term": 0.20, "medium_term": 0.25, "short_term": 0.30},
        conviction={"long_term": 0.45, "medium_term": 0.40, "short_term": 0.35},
        thesis={
            "long_term": "Apple's long-term narrative remains intact but faces structural headwinds: iPhone dependency persists, China competition from Huawei intensifies, and Warren Buffett has significantly reduced Berkshire's Apple stake, signaling institutional uncertainty about growth sustainability. However, the company's $500B U.S. investment pledge and services growth provide structural support for eventual recovery.",
            "medium_term": "Apple trades at elevated risk over the next 3-12 months due to multiple overlapping catalysts: tariff-driven iPhone price increases of ~9%, a 9-10% stock decline from February highs amid market selloff, and DeepSeek AI concerns creating sector-wide pressure. The Siri delay signals competitive vulnerability in AI. Insider selling by Tim Cook at $255-260 precedes this decline, suggesting management awareness of headwinds.",
            "short_term": "Immediate momentum is sharply bearish following Apple's worst start to a year since 2008. The stock crashed from $238 to $212 (-11%) in just 2 weeks (March 5-14, 2025), with accelerating selling pressure and negative technical signals including MACD and RSI. Heavy volume confirms institutional distribution. Stagflation fears and Magnificent 7 selloff create headwinds for risk assets.",
        },
    )

    return {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "market_report": market.model_dump_json(),
        "fundamentals_report": fundamental.model_dump_json(),
        "news_report": news.model_dump_json(),
    }


def _run_analysts_once(
    llm,
    ticker: str,
    trade_date: str,
    depth: str,
    horizons: tuple[str, ...],
) -> dict:
    """Return deterministic mock reports for synthesis deviation testing.

    Real analyst runs are skipped so the test isolates synthesis-agent
    non-determinism from analyst non-determinism.
    """
    _log.info("Using MOCK analyst reports for %s @ %s", ticker, trade_date)
    return _build_mock_state(ticker, trade_date)


def _run_once(
    ticker: str,
    trade_date: str,
    depth: str,
    horizons: tuple[str, ...],
    seed: int,
    run_idx: int,
    state: dict,
) -> dict | None:
    """Single synthesis run. Creates LLM + node inside worker for pickle-safety.
    Returns dict with composite metrics or error info.
    """
    try:
        llm = create_llm_client(
            provider="ollama",
            model="minimax-m2.7:cloud",
            base_url="http://localhost:11434/v1",
            max_retries=10,
            reasoning_effort="high",
            temperature=0,
            seed=seed,
        ).get_llm()
        node = create_synthesis_agent(llm, horizons=horizons)
        result = node(state)
        raw_json = result.get("composite_signal", "{}")
        composite = CompositeSignal.model_validate_json(raw_json)
        return {
            "idx": run_idx,
            "ok": True,
            "mu_final": composite.mu_final,
            "sigma_final": composite.sigma_final,
            "conviction_final": composite.conviction_final,
            "mu_l": composite.mu_composite.long_term,
            "mu_m": composite.mu_composite.medium_term,
            "mu_s": composite.mu_composite.short_term,
            "sigma_l": composite.sigma_composite.long_term,
            "sigma_m": composite.sigma_composite.medium_term,
            "sigma_s": composite.sigma_composite.short_term,
            "wt_f_l": composite.analyst_weights.long_term.fundamental,
            "wt_m_l": composite.analyst_weights.long_term.market,
            "wt_n_l": composite.analyst_weights.long_term.news,
            "wt_f_m": composite.analyst_weights.medium_term.fundamental,
            "wt_m_m": composite.analyst_weights.medium_term.market,
            "wt_n_m": composite.analyst_weights.medium_term.news,
            "wt_f_s": composite.analyst_weights.short_term.fundamental,
            "wt_m_s": composite.analyst_weights.short_term.market,
            "wt_n_s": composite.analyst_weights.short_term.news,
            "blend_l": composite.horizon_blend_weights.long_term,
            "blend_m": composite.horizon_blend_weights.medium_term,
            "blend_s": composite.horizon_blend_weights.short_term,
            "unresolved_penalty": composite.unresolved_penalty,
            "conflicts": len(composite.cross_signal_conflicts),
            "composite": composite,
        }
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return {"idx": run_idx, "ok": False, "error": str(exc), "traceback": tb}


def _print_summary(results: list[dict], num_runs: int, horizons: tuple[str, ...]):
    """Print deviation summary for composite signal outputs."""
    print(f"\n{'=' * 60}")
    print(f"  SYNTHESIS DEVIATION SUMMARY  ({len(results)}/{num_runs} OK)")
    print(f"  horizons: {', '.join(horizons)}")
    print(f"{'=' * 60}\n")

    # Always print final unified signal stats
    _print_metric_block("mu_final", [r["mu_final"] for r in results])
    _print_metric_block("sigma_final", [r["sigma_final"] for r in results])
    _print_metric_block("conviction_final", [r["conviction_final"] for r in results])

    # Per-horizon composite metrics (filtered)
    if "long_term" in horizons:
        print()
        _print_metric_block("mu_composite (long)", [r["mu_l"] for r in results])
        _print_metric_block("sigma_composite (long)", [r["sigma_l"] for r in results])
    if "medium_term" in horizons:
        print()
        _print_metric_block("mu_composite (medium)", [r["mu_m"] for r in results])
        _print_metric_block("sigma_composite (medium)", [r["sigma_m"] for r in results])
    if "short_term" in horizons:
        print()
        _print_metric_block("mu_composite (short)", [r["mu_s"] for r in results])
        _print_metric_block("sigma_composite (short)", [r["sigma_s"] for r in results])

    # Analyst weights (filtered by horizon)
    print(f"\n{'-' * 60}")
    print("  ANALYST WEIGHTS")
    print(f"{'-' * 60}")
    if "long_term" in horizons:
        _print_metric_block("  wt fundamental (L)", [r["wt_f_l"] for r in results])
        _print_metric_block("  wt market      (L)", [r["wt_m_l"] for r in results])
        _print_metric_block("  wt news        (L)", [r["wt_n_l"] for r in results])
    if "medium_term" in horizons:
        print()
        _print_metric_block("  wt fundamental (M)", [r["wt_f_m"] for r in results])
        _print_metric_block("  wt market      (M)", [r["wt_m_m"] for r in results])
        _print_metric_block("  wt news        (M)", [r["wt_n_m"] for r in results])
    if "short_term" in horizons:
        print()
        _print_metric_block("  wt fundamental (S)", [r["wt_f_s"] for r in results])
        _print_metric_block("  wt market      (S)", [r["wt_m_s"] for r in results])
        _print_metric_block("  wt news        (S)", [r["wt_n_s"] for r in results])

    # Horizon blend weights
    print(f"\n{'-' * 60}")
    print("  HORIZON BLEND WEIGHTS")
    print(f"{'-' * 60}")
    if "long_term" in horizons:
        _print_metric_block("  blend (L)", [r["blend_l"] for r in results])
    if "medium_term" in horizons:
        _print_metric_block("  blend (M)", [r["blend_m"] for r in results])
    if "short_term" in horizons:
        _print_metric_block("  blend (S)", [r["blend_s"] for r in results])

    # Penalty / conflicts
    print(f"\n{'-' * 60}")
    print("  CONFLICTS & PENALTIES")
    print(f"{'-' * 60}")
    _print_metric_block(
        "unresolved_penalty", [r["unresolved_penalty"] for r in results]
    )
    _print_metric_block("conflict_count", [float(r["conflicts"]) for r in results])

    # ── Pretty-print first successful composite ──────────────────────────
    if not results:
        return
    comp: CompositeSignal = results[0]["composite"]
    print(f"\n{'=' * 60}")
    print("  SAMPLE COMPOSITE (run 1)")
    print(f"{'=' * 60}")
    print(f"  ticker           : {comp.ticker}")
    print(f"  mu_final         : {comp.mu_final:+.4f}")
    print(f"  sigma_final      : {comp.sigma_final:.4f}")
    print(f"  conviction_final : {comp.conviction_final:.4f}")

    if "long_term" in horizons:
        print(
            f"  mu_composite (L) : {comp.mu_composite.long_term:+.4f}  "
            f"sigma={comp.sigma_composite.long_term:.4f}  "
            f"blend={comp.horizon_blend_weights.long_term:.2f}"
        )
        print(
            f"  weights (L)      : F={comp.analyst_weights.long_term.fundamental:.2f}  "
            f"M={comp.analyst_weights.long_term.market:.2f}  "
            f"N={comp.analyst_weights.long_term.news:.2f}"
        )
    if "medium_term" in horizons:
        print(
            f"  mu_composite (M) : {comp.mu_composite.medium_term:+.4f}  "
            f"sigma={comp.sigma_composite.medium_term:.4f}  "
            f"blend={comp.horizon_blend_weights.medium_term:.2f}"
        )
        print(
            f"  weights (M)      : F={comp.analyst_weights.medium_term.fundamental:.2f}  "
            f"M={comp.analyst_weights.medium_term.market:.2f}  "
            f"N={comp.analyst_weights.medium_term.news:.2f}"
        )
    if "short_term" in horizons:
        print(
            f"  mu_composite (S) : {comp.mu_composite.short_term:+.4f}  "
            f"sigma={comp.sigma_composite.short_term:.4f}  "
            f"blend={comp.horizon_blend_weights.short_term:.2f}"
        )
        print(
            f"  weights (S)      : F={comp.analyst_weights.short_term.fundamental:.2f}  "
            f"M={comp.analyst_weights.short_term.market:.2f}  "
            f"N={comp.analyst_weights.short_term.news:.2f}"
        )

    print(f"  conflicts        : {len(comp.cross_signal_conflicts)}")
    for c in comp.cross_signal_conflicts:
        print(
            f"    • [{c.horizon}] {c.analyst_a.value} vs {c.analyst_b.value}: {c.conflict_description[:60]}..."
        )
    print(f"  rationale        : {comp.weighting_rationale[:120]}...")
    print(f"{'=' * 60}")
    print("\nFull JSON of run 1:")
    print(json.dumps(comp.model_dump(mode="json"), indent=2))


def _print_metric_block(label: str, vals: list[float]):
    """Print mean/stdev/spread for a numeric list."""
    if len(vals) < 2:
        print(f"  {label:22s}: n={len(vals)}  (insufficient for stats)")
        return
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals)
    spread = max(vals) - min(vals)
    print(
        f"  {label:22s}: n={len(vals):2d}  "
        f"mean={mean:+.6f}  stdev={stdev:.6f}  spread={spread:.6f}  "
        f"range=[{min(vals):+.6f}, {max(vals):+.6f}]"
    )


def _format_run_line(
    res: dict, run_idx: int, total: int, horizons: tuple[str, ...]
) -> str:
    """Format a single run result line, showing only selected horizons."""
    parts = [f"  Run {run_idx:02d}/{total}  OK"]
    mu_parts = []
    if "long_term" in horizons:
        mu_parts.append(f"L={res['mu_l']:+.4f}")
    if "medium_term" in horizons:
        mu_parts.append(f"M={res['mu_m']:+.4f}")
    if "short_term" in horizons:
        mu_parts.append(f"S={res['mu_s']:+.4f}")
    parts.append(f"mu=[{','.join(mu_parts)}]")
    parts.append(f"final={res['mu_final']:+.4f}")
    parts.append(f"conv={res['conviction_final']:.2f}")
    parts.append(f"conflicts={int(res['conflicts'])}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Deviation test for synthesis (trader) agent"
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    parser.add_argument("--date", default="2025-03-14", help="Trade date (yyyy-mm-dd)")
    parser.add_argument(
        "--depth", default="shallow", choices=["shallow", "medium", "deep"]
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of synthesis runs")
    parser.add_argument(
        "--workers", type=int, default=4, help="Max parallel synthesis workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LLM")
    parser.add_argument(
        "--output",
        default="experiment_results/deviation_synthesis.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--horizons",
        default="long,medium,short",
        help="Comma-separated horizons to evaluate (long,medium,short)",
    )
    args = parser.parse_args()

    horizons = _parse_horizons(args.horizons)
    _log.info("Active horizons: %s", horizons)

    _log.info("Creating LLM client ...")
    llm = create_llm_client(
        provider="ollama",
        model="minimax-m2.7:cloud",
        base_url="http://localhost:11434/v1",
        max_retries=10,
        reasoning_effort="high",
        temperature=0,
        seed=args.seed,
    ).get_llm()

    # ── Phase 1: Run all three analysts once for real ──────────────────────
    _log.info(
        "Running analysts once (depth=%s) for %s @ %s ...",
        args.depth,
        args.ticker,
        args.date,
    )
    synthesis_state = _run_analysts_once(
        llm, args.ticker, args.date, args.depth, horizons
    )

    # ── Phase 2: Run synthesis N times (node built inside each worker) ────
    _log.info(
        "Running %d syntheses in parallel (max_workers=%d) ...",
        args.runs,
        args.workers,
    )
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_once,
                args.ticker,
                args.date,
                args.depth,
                horizons,
                args.seed,
                i,
                synthesis_state,
            ): i
            for i in range(1, args.runs + 1)
        }
        for fut in as_completed(futures):
            res = fut.result()
            idx = res["idx"]
            if res["ok"]:
                print(_format_run_line(res, idx, args.runs, horizons))
                results.append(res)
            else:
                print(f"  Run {idx:02d}/{args.runs}  FAIL  {res['error']}")
                if res.get("traceback"):
                    print(f"    TRACEBACK:\n{res['traceback'][:800]}")

    _print_summary(results, args.runs, horizons)
    print("\nDone.")

    # ── Save structured results ────────────────────────────────────────────
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    serializable_results = []
    for r in results:
        d = {k: v for k, v in r.items() if k != "composite"}
        serializable_results.append(d)
    summary = {
        "test_type": "deviation",
        "analyst": "synthesis",
        "ticker": args.ticker,
        "date": args.date,
        "depth": args.depth,
        "runs_requested": args.runs,
        "runs_successful": len(results),
        "horizons": list(horizons),
        "results": serializable_results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
