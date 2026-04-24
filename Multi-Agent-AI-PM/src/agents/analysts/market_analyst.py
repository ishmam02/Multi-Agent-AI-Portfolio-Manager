"""
Market (Technical) Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather OHLCV data and technical indicators
  2. LLM plans which statistical/technical metrics to compute
  3. Code agent computes them
  4. LLM interprets results and forms a thesis on price direction
"""

import json

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
        "agent_type": AgentType.MARKET,
        "log_tag": "market",
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
        analyst_type="market",
        project_root=_project_root,
        verbose=True,
    )

    market_node = create_market_analyst(reasoning_llm, code_agent, RESEARCH_DEPTH)

    init_state = {
        "company_of_interest": TICKER,
        "trade_date": TRADE_DATE,
    }

    print(f"Running market analysis for {TICKER} on {TRADE_DATE} ...")
    result = market_node(init_state)

    report_json = result.get("market_report", "{}")
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
    print(f"  MARKET REPORT — {TICKER} @ {TRADE_DATE}")
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