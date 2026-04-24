"""
Fundamentals Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather financial statements & fundamentals
  2. LLM plans which valuation models and metrics to compute
  3. Code agent computes them using exact formulas
  4. LLM interprets results and forms an investment thesis
"""

import json

from src.agents.utils.schemas import AgentType
from src.agents.analysts.base_analyst import create_analyst_node
from src.agents.prompts import load_prompt

# ── Load prompts from external markdown file ────────────────────────────────

_fundamentals_prompts = load_prompt("fundamentals_analyst")
HORIZON_FOCUS: dict[str, str] = _fundamentals_prompts["HORIZON_FOCUS"]  # type: ignore[assignment]
PHASE1_PROMPT: str = _fundamentals_prompts["PHASE1_PROMPT"]  # type: ignore[assignment]
PHASE2_PROMPT: str = _fundamentals_prompts["PHASE2_PROMPT"]  # type: ignore[assignment]
PHASE3_PROMPT: str = _fundamentals_prompts["PHASE3_PROMPT"]  # type: ignore[assignment]


# ── Data gathering (deterministic, no LLM) ──────────────────────────────────


def gather_fundamental_data(ticker: str, trade_date: str, lookback_days: int) -> dict:
    """Fetch financial statements, fundamentals, and insider transactions."""
    from src.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_earnings_dates,
        get_fundamentals,
        get_income_statement,
        get_insider_transactions,
        get_quarterly_history,
    )

    data = {}
    data["fundamentals"] = get_fundamentals.invoke(
        {"ticker": ticker, "curr_date": trade_date}
    )
    data["balance_sheet_quarterly"] = get_balance_sheet.invoke(
        {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}
    )
    data["balance_sheet_annual"] = get_balance_sheet.invoke(
        {"ticker": ticker, "freq": "annual", "curr_date": trade_date}
    )
    data["cashflow"] = get_cashflow.invoke(
        {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}
    )
    data["income_statement"] = get_income_statement.invoke(
        {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}
    )
    data["insider_transactions"] = get_insider_transactions.invoke({"ticker": ticker})
    data["earnings_dates"] = get_earnings_dates.invoke({"ticker": ticker})
    data["quarterly_history"] = get_quarterly_history.invoke({"ticker": ticker})
    return data


# ── Node factory ─────────────────────────────────────────────────────────────


def create_fundamentals_analyst(reasoning_llm, code_agent, research_depth="medium"):
    """Create a fundamentals analyst node for the outer AgentState graph.

    Parameters
    ----------
    reasoning_llm  : LangChain chat model for Phase 1 (plan) and Phase 3 (thesis)
    code_agent     : CodeValidationAgent instance for Phase 2 (compute)
    research_depth : "shallow" | "medium" | "deep"
    """
    analyst_config = {
        "agent_type": AgentType.FUNDAMENTAL,
        "state_key": "fundamentals_report",
        "gather_fn": gather_fundamental_data,
        "phase1_system_prompt": PHASE1_PROMPT,
        "phase2_system_prompt": PHASE2_PROMPT,
        "phase3_system_prompt": PHASE3_PROMPT,
        "horizon_focus": HORIZON_FOCUS,
        "active_horizons": ("long_term",),
    }

    return create_analyst_node(
        reasoning_llm,
        code_agent,
        analyst_config,
        research_depth,
        verbose=True,
    )


# if __name__ == "__main__":
#     import pathlib
#     import statistics
#
#     # ── Multi-run deviation test (repeatability check) ──────────────────────
#     _project_root = str(pathlib.Path(__file__).resolve().parents[3])
#
#     TICKER = "AAPL"
#     TRADE_DATE = "2025-03-14"
#     RESEARCH_DEPTH = "shallow"
#     RANDOM_SEED = 42
#     NUM_RUNS = 5
#
#     from src.llm_clients import create_llm_client
#     from src.agents.code_agent.code_agent import CodeValidationAgent
#     from src.agents.utils.schemas import ResearchReport
#
#     llm_client = create_llm_client(
#         provider="ollama",
#         model="minimax-m2.7:cloud",
#         base_url="http://localhost:11434/v1",
#         max_retries=10,
#         reasoning_effort="high",
#         temperature=0,
#         seed=RANDOM_SEED,
#     )
#     reasoning_llm = llm_client.get_llm()
#
#     code_agent = CodeValidationAgent(
#         model="minimax-m2.7:cloud",
#         timeout=60,
#         max_iterations=5,
#         analyst_type="fundamental",
#         project_root=_project_root,
#         verbose=False,
#     )
#
#     fundamentals_node = create_fundamentals_analyst(
#         reasoning_llm, code_agent, RESEARCH_DEPTH
#     )
#
#     init_state = {
#         "company_of_interest": TICKER,
#         "trade_date": TRADE_DATE,
#     }
#
#     mu_runs: list[float] = []
#     sigma_runs: list[float] = []
#     conviction_runs: list[float] = []
#
#     for run_idx in range(1, NUM_RUNS + 1):
#         print(
#             f"\n{'=' * 60}\n"
#             f"  Run {run_idx}/{NUM_RUNS} — {TICKER} on {TRADE_DATE} "
#             f"(depth={RESEARCH_DEPTH})\n"
#             f"{'=' * 60}",
#             flush=True,
#         )
#
#         result = fundamentals_node(init_state)
#         report_json = result.get("fundamentals_report", "{}")
#
#         try:
#             report = ResearchReport.model_validate_json(report_json)
#             print(f"\n--- Run {run_idx} Report ---")
#             print(report.model_dump_json(indent=2))
#             mu_runs.append(report.mu.long_term)
#             sigma_runs.append(report.sigma_contribution.long_term)
#             conviction_runs.append(report.conviction.long_term)
#         except Exception as exc:
#             print(f"\nRun {run_idx}: Failed to parse ResearchReport. Reason:", exc)
#             print(report_json)
#
#     # ── Deviation summary ───────────────────────────────────────────────────
#     print(f"\n{'=' * 60}")
#     print(f"  DEVIATION SUMMARY  ({NUM_RUNS} runs, {TICKER} @ {TRADE_DATE})")
#     print(f"{'=' * 60}\n")
#
#     for label, values in [
#         ("mu", mu_runs),
#         ("sigma_contribution", sigma_runs),
#         ("conviction", conviction_runs),
#     ]:
#         if len(values) < 2:
#             print(f"  {label}: only {len(values)} successful run(s) — skipping")
#             continue
#         mean = statistics.mean(values)
#         stdev = statistics.stdev(values)
#         spread = max(values) - min(values)
#         print(
#             f"  {label}:  values={[round(v, 6) for v in values]}  "
#             f"mean={mean:.6f}  stdev={stdev:.6f}  spread={spread:.6f}"
#         )
#     print()


if __name__ == "__main__":
    import pathlib

    # ── Simple single-run test ────────────────────────────────────────────────
    _project_root = str(pathlib.Path(__file__).resolve().parents[3])

    TICKER = "AAPL"
    TRADE_DATE = "2026-04-24"
    RESEARCH_DEPTH = "shallow"
    RANDOM_SEED = 42

    from src.llm_clients import create_llm_client
    from src.agents.code_agent.code_agent import CodeValidationAgent
    from src.agents.utils.schemas import ResearchReport

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
        analyst_type="fundamental",
        project_root=_project_root,
        verbose=True,
    )

    fundamentals_node = create_fundamentals_analyst(
        reasoning_llm, code_agent, RESEARCH_DEPTH
    )

    init_state = {
        "company_of_interest": TICKER,
        "trade_date": TRADE_DATE,
    }

    print(f"Running fundamentals analysis for {TICKER} on {TRADE_DATE} ...")
    result = fundamentals_node(init_state)

    report_json = result.get("fundamentals_report", "{}")
    try:
        report = ResearchReport.model_validate_json(report_json)
    except Exception as exc:
        print(f"Failed to parse ResearchReport: {exc}")
        print("Raw JSON:")
        print(report_json)
        raise SystemExit(1)

    print(f"\n{'=' * 60}")
    print("Full Report.")
    print(json.dumps(report, indent=2))
    print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"  FUNDAMENTALS REPORT — {TICKER} @ {TRADE_DATE}")
    print(f"{'=' * 60}\n")

    print(f"mu:")
    print(f"  long_term   = {report.mu.long_term:+.6f}")

    print(f"\nsigma_contribution:")
    print(f"  long_term   = {report.sigma_contribution.long_term:.6f}")

    print(f"\nconviction:")
    print(f"  long_term   = {report.conviction.long_term:.4f}")

    print(f"\nthesis")
    print(report.investment_thesis)

    print(f"\ncomputed_metrics ({len(report.computed_metrics)}):")
    for m in report.computed_metrics:
        print(f"  {m.metric_name}: {m.value}")

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}")
