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


def create_market_analyst(reasoning_llm, code_agent, research_depth="medium", active_horizons=("long_term",)):
    """Create a market/technical analyst node for the outer AgentState graph.

    Parameters
    ----------
    reasoning_llm  : LangChain chat model for Phase 1 (plan) and Phase 3 (thesis)
    code_agent     : CodeValidationAgent instance for Phase 2 (compute)
    research_depth : "shallow" | "medium" | "deep"
    active_horizons : Tuple of horizons to evaluate (e.g. ("short_term",) or ("long_term", "medium_term", "short_term"))
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
        "active_horizons": active_horizons,
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

    _project_root = str(pathlib.Path(__file__).resolve().parents[3])

    from src.llm_clients import create_llm_client
    from src.agents.code_agent.code_agent import CodeValidationAgent
    from src.agents.utils.schemas import ResearchReport

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
        timeout=60,
        max_iterations=5,
        analyst_type="market",
        project_root=_project_root,
        verbose=False,
    )
    node = create_market_analyst(llm, ca, "shallow", active_horizons=("long_term",))
    result = node({"company_of_interest": "SPY", "trade_date": "2025-03-14"})
    report = ResearchReport.model_validate_json(result.get("market_report", "{}"))
    print(report.model_dump_json(indent=2))
