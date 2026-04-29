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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Ensure project root is on path ────────────────────────────────────────────
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.llm_clients import create_llm_client
from src.agents.code_agent.code_agent import CodeValidationAgent
from src.agents.trader.trader import create_synthesis_agent
from src.agents.utils.schemas import (
    AgentType,
    AnalystWeights,
    Catalyst,
    Citation,
    CompositeSignal,
    ComputedMetric,
    HorizonThesis,
    HorizonTraceIds,
    HorizonValues,
    HorizonWeights,
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
        mu={"long_term": 0.08, "medium_term": 0.04, "short_term": -0.02},
        sigma={"long_term": 0.18, "medium_term": 0.14, "short_term": 0.22},
        conviction={"long_term": 0.72, "medium_term": 0.55, "short_term": 0.40},
        thesis={
            "long_term": "Technical trend bullish with 200-day SMA support.",
            "medium_term": "RSI neutral; consolidation expected.",
            "short_term": "Overbought near-term; pullback likely.",
        },
    )

    fundamental = _build_mock_report(
        AgentType.FUNDAMENTAL,
        ticker,
        trade_date,
        mu={"long_term": 0.12, "medium_term": 0.06, "short_term": 0.01},
        sigma={"long_term": 0.15, "medium_term": 0.12, "short_term": 0.10},
        conviction={"long_term": 0.80, "medium_term": 0.60, "short_term": 0.30},
        thesis={
            "long_term": "DCF implies 15% upside; strong free cash flow.",
            "medium_term": "Earnings growth stable; margins improving.",
            "short_term": "Valuation fair; no near-term catalyst.",
        },
    )

    news = _build_mock_report(
        AgentType.NEWS,
        ticker,
        trade_date,
        mu={"long_term": 0.05, "medium_term": 0.02, "short_term": -0.05},
        sigma={"long_term": 0.20, "medium_term": 0.18, "short_term": 0.25},
        conviction={"long_term": 0.50, "medium_term": 0.45, "short_term": 0.35},
        thesis={
            "long_term": "Sentiment mildly positive; no major regulatory risks.",
            "medium_term": "Sector rotation headwinds noted.",
            "short_term": "Negative sentiment spike on supply-chain rumour.",
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
    workers: int = 3,
) -> dict:
    """Run all three analysts once in parallel and return a state dict with real reports.

    Falls back to mock reports for any analyst that fails.
    """
    from src.agents.analysts.market_analyst import create_market_analyst
    from src.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
    from src.agents.analysts.news_analyst import create_news_analyst

    # Build code agents for market & fundamentals
    market_ca = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        timeout=60,
        max_iterations=5,
        analyst_type="market",
        project_root=_project_root,
        verbose=False,
    )
    fundamentals_ca = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        timeout=60,
        max_iterations=5,
        analyst_type="fundamental",
        project_root=_project_root,
        verbose=False,
    )

    # Build nodes
    market_node = create_market_analyst(
        llm, market_ca, depth, active_horizons=horizons
    )
    fundamentals_node = create_fundamentals_analyst(
        llm, fundamentals_ca, depth, active_horizons=horizons
    )
    news_node = create_news_analyst(llm, depth, horizons=horizons)

    init_state = {"company_of_interest": ticker, "trade_date": trade_date}

    def _run_market():
        try:
            result = market_node(init_state)
            report_json = result.get("market_report", "{}")
            ResearchReport.model_validate_json(report_json)  # sanity check
            return "market", report_json, None
        except Exception as exc:
            return "market", None, exc

    def _run_fundamentals():
        try:
            result = fundamentals_node(init_state)
            report_json = result.get("fundamentals_report", "{}")
            ResearchReport.model_validate_json(report_json)
            return "fundamentals", report_json, None
        except Exception as exc:
            return "fundamentals", None, exc

    def _run_news():
        try:
            result = news_node(init_state)
            report_json = result.get("news_report", "{}")
            ResearchReport.model_validate_json(report_json)
            return "news", report_json, None
        except Exception as exc:
            return "news", None, exc

    reports: dict[str, str] = {}
    errors: dict[str, Exception] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_market): "market",
            pool.submit(_run_fundamentals): "fundamentals",
            pool.submit(_run_news): "news",
        }
        for fut in as_completed(futures):
            analyst_key, report_json, err = fut.result()
            if err:
                errors[analyst_key] = err
                _log.warning("%s analyst failed: %s", analyst_key, err)
            else:
                reports[analyst_key] = report_json

    # Build final state — use real reports where available, mock fallback otherwise
    state = {"company_of_interest": ticker, "trade_date": trade_date}
    mock_fallback = _build_mock_state(ticker, trade_date)

    report_keys = {
        "market": "market_report",
        "fundamentals": "fundamentals_report",
        "news": "news_report",
    }

    for key in ("market", "fundamentals", "news"):
        if key in reports:
            state[report_keys[key]] = reports[key]
            _log.info("Using REAL %s report", key)
        else:
            state[report_keys[key]] = mock_fallback[report_keys[key]]
            _log.warning("Using MOCK %s report (real run failed)", key)

    return state


def _run_once(node, state: dict, run_idx: int, horizons: tuple[str, ...]) -> dict | None:
    """Single synthesis run. Returns dict with composite metrics or error info."""
    try:
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
        return {"idx": run_idx, "ok": False, "error": str(exc)}


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
    _print_metric_block("unresolved_penalty", [r["unresolved_penalty"] for r in results])
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
        print(f"    • [{c.horizon}] {c.analyst_a.value} vs {c.analyst_b.value}: {c.conflict_description[:60]}...")
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


def _format_run_line(res: dict, run_idx: int, total: int, horizons: tuple[str, ...]) -> str:
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
    parser.add_argument("--workers", type=int, default=4, help="Max parallel synthesis workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LLM")
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
        args.depth, args.ticker, args.date,
    )
    synthesis_state = _run_analysts_once(
        llm, args.ticker, args.date, args.depth, horizons, workers=3
    )

    # ── Phase 2: Build synthesis node and run N times ──────────────────────
    _log.info("Building synthesis (trader) node ...")
    node = create_synthesis_agent(llm, horizons=horizons)

    _log.info(
        "Running %d syntheses in parallel (max_workers=%d) ...",
        args.runs,
        args.workers,
    )
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_once, node, synthesis_state, i, horizons): i
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

    _print_summary(results, args.runs, horizons)
    print("\nDone.")


if __name__ == "__main__":
    main()
