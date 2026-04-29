"""
Deviation test — run any analyst N times in parallel and report statistics.

Usage:
    python -m src.backtest.deviation_test --analyst news --ticker AAPL --date 2025-03-14 --runs 30 --workers 10
    python -m src.backtest.deviation_test --analyst fundamental --ticker AAPL --date 2025-03-14 --runs 10
    python -m src.backtest.deviation_test --analyst market --ticker SPY --date 2025-03-14 --runs 10
    python -m src.backtest.deviation_test --analyst market --horizons long,short --runs 10
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Ensure project root is on path ────────────────────────────────────────────
_project_root = str(pathlib.Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.llm_clients import create_llm_client
from src.agents.utils.schemas import ResearchReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("deviation_test")


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


def _build_node(analyst: str, llm, depth: str, active_horizons: tuple[str, ...]):
    """Build the appropriate analyst node."""
    if analyst == "news":
        from src.agents.analysts.news_analyst import create_news_analyst

        return create_news_analyst(llm, depth, horizons=active_horizons)

    from src.agents.code_agent.code_agent import CodeValidationAgent

    ca = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        timeout=60,
        max_iterations=5,
        analyst_type=analyst,
        project_root=_project_root,
        verbose=False,
    )
    if analyst == "fundamental":
        from src.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

        return create_fundamentals_analyst(
            llm, ca, depth, active_horizons=active_horizons
        )
    if analyst == "market":
        from src.agents.analysts.market_analyst import create_market_analyst

        return create_market_analyst(llm, ca, depth, active_horizons=active_horizons)
    raise ValueError(f"Unknown analyst: {analyst}")


def _report_key(analyst: str) -> str:
    """JSON key that holds the report for this analyst type."""
    return {
        "news": "news_report",
        "fundamental": "fundamentals_report",
        "market": "market_report",
    }[analyst]


def _run_once(node, state: dict, run_idx: int, analyst: str) -> dict | None:
    """Single analysis run.  Returns dict with metrics or error info."""
    try:
        result = node(state)
        raw_json = result.get(_report_key(analyst), "{}")
        report = ResearchReport.model_validate_json(raw_json)
        return {
            "idx": run_idx,
            "ok": True,
            "mu_l": report.mu.long_term,
            "mu_m": report.mu.medium_term,
            "mu_s": report.mu.short_term,
            "sigma_l": report.sigma_contribution.long_term,
            "sigma_m": report.sigma_contribution.medium_term,
            "sigma_s": report.sigma_contribution.short_term,
            "conv_l": report.conviction.long_term,
            "conv_m": report.conviction.medium_term,
            "conv_s": report.conviction.short_term,
            "report": report,
            "extra": {
                k: v for k, v in result.items() if k not in (_report_key(analyst),)
            },
        }
    except Exception as exc:
        return {"idx": run_idx, "ok": False, "error": str(exc)}


def _print_summary(
    results: list[dict],
    ticker: str,
    trade_date: str,
    num_runs: int,
    horizons: tuple[str, ...],
):
    """Print deviation summary from parallel run results."""
    print(f"\n{'=' * 60}")
    print(
        f"  DEVIATION SUMMARY  ({len(results)}/{num_runs} OK, {ticker} @ {trade_date})"
    )
    print(f"  horizons: {', '.join(horizons)}")
    print(f"{'=' * 60}\n")

    # Filter metric rows by selected horizons
    horizon_key_map = {
        "long_term": [
            ("mu (long)", "mu_l"),
            ("sigma (long)", "sigma_l"),
            ("conviction (long)", "conv_l"),
        ],
        "medium_term": [
            ("mu (medium)", "mu_m"),
            ("sigma (medium)", "sigma_m"),
            ("conviction (medium)", "conv_m"),
        ],
        "short_term": [
            ("mu (short)", "mu_s"),
            ("sigma (short)", "sigma_s"),
            ("conviction (short)", "conv_s"),
        ],
    }
    metric_rows = []
    for h in horizons:
        metric_rows.extend(horizon_key_map[h])

    for label, key in metric_rows:
        vals = [r[key] for r in results]
        if len(vals) < 2:
            print(f"  {label:20s}: n={len(vals)}  (insufficient for stats)")
            continue
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals)
        spread = max(vals) - min(vals)
        print(
            f"  {label:20s}: n={len(vals):2d}  "
            f"mean={mean:+.6f}  stdev={stdev:.6f}  spread={spread:.6f}  "
            f"range=[{min(vals):+.6f}, {max(vals):+.6f}]"
        )

    # ── Pretty-print first successful report ──────────────────────────
    if not results:
        return
    report = results[0]["report"]
    print(f"\n{'=' * 60}")
    print("  SAMPLE REPORT (run 1)")
    print(f"{'=' * 60}")
    print(f"  ticker           : {report.ticker}")

    mu_parts = []
    sigma_parts = []
    conv_parts = []
    if "long_term" in horizons:
        mu_parts.append(f"L={report.mu.long_term:+.4f}")
        sigma_parts.append(f"L={report.sigma_contribution.long_term:.4f}")
        conv_parts.append(f"L={report.conviction.long_term:.2f}")
    if "medium_term" in horizons:
        mu_parts.append(f"M={report.mu.medium_term:+.4f}")
        sigma_parts.append(f"M={report.sigma_contribution.medium_term:.4f}")
        conv_parts.append(f"M={report.conviction.medium_term:.2f}")
    if "short_term" in horizons:
        mu_parts.append(f"S={report.mu.short_term:+.4f}")
        sigma_parts.append(f"S={report.sigma_contribution.short_term:.4f}")
        conv_parts.append(f"S={report.conviction.short_term:.2f}")

    print(f"  mu               : {'  '.join(mu_parts)}")
    print(f"  sigma            : {'  '.join(sigma_parts)}")
    print(f"  conviction       : {'  '.join(conv_parts)}")

    # Print any extra fields (e.g. confidence_rationale for news analyst)
    extra = results[0].get("extra", {})
    for key, val in extra.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for h, txt in val.items():
                if txt and h in horizons:
                    print(
                        f"    • [{h[:1].upper()}] {txt[:100]}…"
                        if len(txt) > 100
                        else f"    • [{h[:1].upper()}] {txt}"
                    )

    print(f"  catalysts        : {len(report.key_catalysts)}")
    for cat in report.key_catalysts:
        if cat.term in horizons:
            print(f"    • [{cat.term}] {cat.catalyst}")
    print(f"  risks            : {len(report.key_risks)}")
    for rsk in report.key_risks:
        if rsk.term in horizons:
            print(f"    • [{rsk.term}] {rsk.risk}")
    print(f"  citations        : {len(report.citation_chain)}")
    for cit in report.citation_chain[:5]:
        print(f"    • {cit.source_uuid}  {cit.source}")
    if len(report.citation_chain) > 5:
        print(f"    … and {len(report.citation_chain) - 5} more")
    print(f"{'=' * 60}")
    print("\nFull JSON of run 1:")
    print(report.model_dump_json(indent=2))


def _format_run_line(
    res: dict, run_idx: int, total: int, horizons: tuple[str, ...]
) -> str:
    """Format a single run result line, showing only selected horizons."""
    parts = [f"  Run {run_idx:02d}/{total}  OK"]
    mu_parts = []
    conv_parts = []
    if "long_term" in horizons:
        mu_parts.append(f"{res['mu_l']:+.4f}")
        conv_parts.append(f"{res['conv_l']:.2f}")
    if "medium_term" in horizons:
        mu_parts.append(f"{res['mu_m']:+.4f}")
        conv_parts.append(f"{res['conv_m']:.2f}")
    if "short_term" in horizons:
        mu_parts.append(f"{res['mu_s']:+.4f}")
        conv_parts.append(f"{res['conv_s']:.2f}")
    parts.append(f"mu=[{','.join(mu_parts)}]")
    parts.append(f"conv=[{','.join(conv_parts)}]")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Deviation test for analyst agents")
    parser.add_argument(
        "--analyst",
        required=True,
        choices=["news", "fundamental", "market"],
        help="Which analyst to test",
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    parser.add_argument("--date", default="2025-03-14", help="Trade date (yyyy-mm-dd)")
    parser.add_argument(
        "--depth", default="shallow", choices=["shallow", "medium", "deep"]
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LLM")
    parser.add_argument(
        "--horizons",
        default="long,medium,short",
        help="Comma-separated horizons to evaluate (long,medium,short)",
    )
    args = parser.parse_args()

    active_horizons = _parse_horizons(args.horizons)
    _log.info("Active horizons: %s", active_horizons)

    _log.info("Creating LLM client …")
    llm = create_llm_client(
        provider="ollama",
        model="minimax-m2.7:cloud",
        base_url="http://localhost:11434/v1",
        max_retries=10,
        reasoning_effort="high",
        temperature=0,
        seed=args.seed,
    ).get_llm()

    _log.info("Building %s analyst node (depth=%s) …", args.analyst, args.depth)
    node = _build_node(args.analyst, llm, args.depth, active_horizons)
    init_state = {"company_of_interest": args.ticker, "trade_date": args.date}

    _log.info(
        "Running %d analyses in parallel (max_workers=%d) …",
        args.runs,
        args.workers,
    )
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_once, node, init_state, i, args.analyst): i
            for i in range(1, args.runs + 1)
        }
        for fut in as_completed(futures):
            res = fut.result()
            idx = res["idx"]
            if res["ok"]:
                print(_format_run_line(res, idx, args.runs, active_horizons))
                results.append(res)
            else:
                print(f"  Run {idx:02d}/{args.runs}  FAIL  {res['error']}")

    _print_summary(results, args.ticker, args.date, args.runs, active_horizons)
    print("\nDone.")


if __name__ == "__main__":
    main()
