"""
Market (Technical) Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather OHLCV data and technical indicators
  2. LLM plans which statistical/technical metrics to compute
  3. Code agent computes them
  4. LLM interprets results and forms a thesis on price direction
"""

from src.agents.code_agent.code_agent import CodeValidationAgent
from src.agents.utils.schemas import AgentType, ResearchReport
from src.agents.analysts.base_analyst import (
    create_analyst_node,
    compute_date_range,
)
from src.agents.prompts import load_prompt
from src.llm_clients import create_llm_client

# ── Load prompts from external markdown file ────────────────────────────────

_market_prompts = load_prompt("market_analyst")
HORIZON_FOCUS: dict[str, str] = _market_prompts["HORIZON_FOCUS"]  # type: ignore[assignment]
PHASE1_PROMPT: str = _market_prompts["PHASE1_PROMPT"]  # type: ignore[assignment]
PHASE2_PROMPT: str = _market_prompts["PHASE2_PROMPT"]  # type: ignore[assignment]
PHASE3_PROMPT: str = _market_prompts["PHASE3_PROMPT"]  # type: ignore[assignment]


# ── Data gathering (deterministic, no LLM) ──────────────────────────────────


def gather_technical_data(ticker: str, trade_date: str, lookback_days: int) -> dict:
    """Fetch OHLCV data and a comprehensive set of technical indicators.

    Ensures at least 252 trading days of history are fetched so that
    long-period indicators (200-day SMA) are computable even for short-term
    horizons (which only require 30 days of lookback for the k-NN analysis).
    """
    from src.agents.utils.agent_utils import get_stock_data

    effective_lookback = max(lookback_days, 252)
    start_date, end_date = compute_date_range(trade_date, effective_lookback)

    data = {}
    data["stock_data"] = get_stock_data.invoke(
        {"symbol": ticker, "start_date": start_date, "end_date": end_date}
    )

    return data


# ── Node factory ─────────────────────────────────────────────────────────────


def create_market_analyst(reasoning_llm, code_agent, research_depth="medium"):
    """Create a market/technical analyst node for the outer AgentState graph.

    Parameters
    ----------
    reasoning_llm  : LangChain chat model for Phase 1 (plan) and Phase 3 (thesis)
    code_agent     : CodeValidationAgent instance for Phase 2 (compute)
    research_depth : "shallow" | "medium" | "deep"
    """
    analyst_config = {
        "agent_type": AgentType.TECHNICAL,
        "state_key": "market_report",
        "gather_fn": gather_technical_data,
        "phase1_system_prompt": PHASE1_PROMPT,
        "phase2_system_prompt": PHASE2_PROMPT,
        "phase3_system_prompt": PHASE3_PROMPT,
        "horizon_focus": HORIZON_FOCUS,
        "active_horizons": (
            "long_term",
        ),  # Only run short-term; LT/MT get zero placeholders
    }

    return create_analyst_node(
        reasoning_llm,
        code_agent,
        analyst_config,
        research_depth,
        verbose=True,
    )


if __name__ == "__main__":
    import bisect
    import pathlib
    import random
    import statistics
    import yfinance as yf

    # ── Regime-aware long-term accuracy test (2010-2024) ────────────────────
    _project_root = str(pathlib.Path(__file__).resolve().parents[3])

    TICKER = "SPY"
    FORWARD_DAYS = 252
    RESEARCH_DEPTH = "shallow"
    RANDOM_SEED = 42
    MAX_WORKERS = 3

    # Regime definitions: (label, start, end, samples)
    REGIMES = [
        ("Post-GFC recovery", "2010-01-01", "2013-12-31", 10),
        ("Mid bull", "2014-01-01", "2017-12-31", 12),
        ("Volatile / trade war", "2018-01-01", "2019-12-31", 10),
        ("COVID crash", "2020-02-15", "2020-03-31", 6),
        ("Recovery bull", "2020-04-01", "2021-12-31", 14),
        ("2022 bear", "2022-01-01", "2022-10-31", 12),
        ("Post-bear bull", "2022-11-01", "2024-12-01", 16),
    ]

    random.seed(RANDOM_SEED)

    # ── Fetch full price history once ────────────────────────────────────────
    print("Fetching SPY trading days ...")
    spy = yf.download(
        TICKER,
        start="2010-01-01",
        end="2024-12-31",
        progress=False,
        auto_adjust=True,
    )
    prices = spy["Close"].values.flatten()
    dates = spy.index.strftime("%Y-%m-%d").tolist()

    # ── Sample dates per regime (bisect handles non-trading-day boundaries) ──
    test_dates: list[
        tuple[str, float, float, str]
    ] = []  # (date, start_px, end_px, regime)
    for label, r_start, r_end, n_samples in REGIMES:
        r_start_idx = bisect.bisect_left(dates, r_start)
        r_end_idx = bisect.bisect_right(dates, r_end) - 1
        if r_start_idx > r_end_idx:
            continue
        valid_in_regime = [
            i
            for i in range(r_start_idx, r_end_idx + 1)
            if i + FORWARD_DAYS < len(dates)
        ]
        if not valid_in_regime:
            continue
        chosen = random.sample(valid_in_regime, min(n_samples, len(valid_in_regime)))
        for i in chosen:
            test_dates.append((dates[i], prices[i], prices[i + FORWARD_DAYS], label))

    print(f"  {len(test_dates)} dates selected across {len(REGIMES)} regimes\n")

    # ── LLM + code-agent setup ───────────────────────────────────────────────
    llm_client = create_llm_client(
        provider="ollama",
        model="minimax-m2.7:cloud",
        base_url="http://localhost:11434/v1",
        max_retries=10,
        reasoning_effort="high",
        temperature=0,
        seed=RANDOM_SEED,
    )
    reasoning_llm = llm_client.get_llm()

    code_agent = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        timeout=60,
        max_iterations=5,
        analyst_type="market",
        project_root=_project_root,
        verbose=False,
    )

    market_node = create_market_analyst(reasoning_llm, code_agent, RESEARCH_DEPTH)

    # ── Run tests (max 3 concurrent) ─────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor
    import threading

    results: list[dict] = []
    results_lock = threading.Lock()

    def _run_one(trade_date: str, start_price: float, end_price: float, regime: str):
        actual_ret = (end_price - start_price) / start_price
        print(f"  [{regime}] {trade_date}  actual_ret={actual_ret:+.6f}")

        init_state = {
            "company_of_interest": TICKER,
            "trade_date": trade_date,
        }
        try:
            result = market_node(init_state)
            report_json = result.get("market_report", "{}")
            report = ResearchReport.model_validate_json(report_json)
            pred_mu = report.mu.long_term
            pred_sigma = report.sigma_contribution.long_term
            conviction = report.conviction.long_term
        except Exception as exc:
            import traceback

            print(f"      ERROR: {exc}")
            traceback.print_exc()
            return

        direction_correct = (pred_mu > 0) == (actual_ret > 0)
        row = {
            "date": trade_date,
            "regime": regime,
            "pred_mu": pred_mu,
            "pred_sigma": pred_sigma,
            "conviction": conviction,
            "actual_ret": actual_ret,
            "direction_correct": direction_correct,
        }
        with results_lock:
            results.append(row)
        print(
            f"      pred_mu={pred_mu:+.6f}  correct_dir={direction_correct}  "
            f"conviction={conviction:.4f}  pred_sigma={pred_sigma:.6f}"
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for d, sp, ep, reg in test_dates:
            executor.submit(_run_one, d, sp, ep, reg)

    # ── Aggregate report ─────────────────────────────────────────────────────
    if not results:
        print("\nNo successful results. Exiting.")
        raise SystemExit(1)

    print(f"\n{'=' * 70}")
    print(f"  REGIME-AWARE LONG-TERM ACCURACY  ({len(results)} dates, {TICKER})")
    print(f"{'=' * 70}\n")

    def _report_batch(label: str, rows: list[dict]):
        if not rows:
            return
        dir_acc = sum(r["direction_correct"] for r in rows) / len(rows)
        mae = statistics.mean(abs(r["pred_mu"] - r["actual_ret"]) for r in rows)
        mse = statistics.mean((r["pred_mu"] - r["actual_ret"]) ** 2 for r in rows)
        avg_conv = statistics.mean(r["conviction"] for r in rows)
        avg_sigma = statistics.mean(r["pred_sigma"] for r in rows)
        print(f"  {label} (n={len(rows)}):")
        print(f"    directional accuracy = {dir_acc:.2%}")
        print(f"    MAE                  = {mae:.6f}")
        print(f"    MSE                  = {mse:.6f}")
        print(f"    RMSE                 = {mse**0.5:.6f}")
        print(f"    avg sigma            = {avg_sigma:.6f}")
        print(f"    avg conviction       = {avg_conv:.4f}")
        print()

    # Overall
    _report_batch("Overall", results)

    # Per-regime
    regimes_present = sorted({r["regime"] for r in results})
    for reg in regimes_present:
        _report_batch(reg, [r for r in results if r["regime"] == reg])

    # Conviction split (global)
    sorted_by_conv = sorted(results, key=lambda r: r["conviction"], reverse=True)
    mid = len(sorted_by_conv) // 2
    top = sorted_by_conv[:mid]
    bot = sorted_by_conv[mid:]
    if top:
        _report_batch("High conviction", top)
    if bot:
        _report_batch("Low conviction", bot)

    print(f"{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


# if __name__ == "__main__":
#     import pathlib
#     import statistics
#
#     # Simple local run settings (plain text, no CLI args).
#     today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
#     # market_analyst.py lives at src/agents/analysts/ — 3 levels up = Multi-Agent-AI-PM/
#     _project_root = str(pathlib.Path(__file__).resolve().parents[3])
#
#     ticker = "AAPL"
#     trade_date = today_str
#     research_depth = "shallow"
#     num_runs = 5  # number of repeated runs
#
#     llm_provider = "ollama"
#     llm_model = "minimax-m2.7:cloud"
#     llm_base_url = "http://localhost:11434/v1"
#     llm_max_retries = 10
#
#     code_model = "minimax-m2.7:cloud"
#     code_timeout = 60
#     code_max_iterations = 5
#
#     llm_client = create_llm_client(
#         provider=llm_provider,
#         model=llm_model,
#         base_url=llm_base_url,
#         max_retries=llm_max_retries,
#         reasoning_effort="high",
#         temperature=0,
#         seed=42,
#     )
#     reasoning_llm = llm_client.get_llm()
#
#     code_agent = CodeValidationAgent(
#         model=code_model,
#         timeout=code_timeout,
#         max_iterations=code_max_iterations,
#         analyst_type="market",
#         project_root=_project_root,
#         verbose=False,
#     )
#
#     market_node = create_market_analyst(
#         reasoning_llm,
#         code_agent,
#         research_depth,
#     )
#
#     init_state = {
#         "company_of_interest": ticker,
#         "trade_date": trade_date,
#     }
#
#     # ── Collect mu, sigma, and conviction from each run ──────────────────────
#     horizons = ["short_term"]
#     mu_runs: dict[str, list[float]] = {h: [] for h in horizons}
#     sigma_runs: dict[str, list[float]] = {h: [] for h in horizons}
#     conviction_runs: dict[str, list[float]] = {h: [] for h in horizons}
#
#     for run_idx in range(1, num_runs + 1):
#         print(
#             f"\n{'=' * 60}\n"
#             f"  Run {run_idx}/{num_runs} — {ticker} on {trade_date} "
#             f"(depth={research_depth})\n"
#             f"{'=' * 60}",
#             flush=True,
#         )
#
#         result = market_node(init_state)
#         report_json = result.get("market_report", "{}")
#
#         try:
#             report = ResearchReport.model_validate_json(report_json)
#             print(f"\n--- Run {run_idx} Report ---")
#             print(report.model_dump_json(indent=2))
#
#             for h in horizons:
#                 mu_runs[h].append(getattr(report.mu, h))
#                 sigma_runs[h].append(getattr(report.sigma_contribution, h))
#                 conviction_runs[h].append(getattr(report.conviction, h))
#
#         except Exception as exc:  # noqa: BLE001
#             print(f"\nRun {run_idx}: Failed to parse ResearchReport. Reason:", exc)
#             print(report_json)
#
#     # ── Deviation summary ───────────────────────────────────────────────────
#     print(f"\n{'=' * 60}")
#     print(f"  DEVIATION SUMMARY  ({num_runs} runs, {ticker} @ {trade_date})")
#     print(f"{'=' * 60}\n")
#
#     for label, runs in [
#         ("mu", mu_runs),
#         ("sigma_contribution", sigma_runs),
#         ("conviction", conviction_runs),
#     ]:
#         print(f"  {label}:")
#         for h in horizons:
#             values = runs[h]
#             if len(values) < 2:
#                 print(f"    {h}: only {len(values)} successful run(s) — skipping")
#                 continue
#             mean = statistics.mean(values)
#             stdev = statistics.stdev(values)
#             spread = max(values) - min(values)
#             print(
#                 f"    {h}:  values={[round(v, 6) for v in values]}  "
#                 f"mean={mean:.6f}  stdev={stdev:.6f}  spread={spread:.6f}"
#             )
#         print()
