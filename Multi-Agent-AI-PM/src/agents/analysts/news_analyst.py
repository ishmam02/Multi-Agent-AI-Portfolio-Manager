"""
News (Macro) Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather macro news, company news, and insider data
  2. LLM plans which regime/sensitivity models to compute
  3. Code agent computes them
  4. LLM interprets macro implications and forms a thesis
"""

from src.agents.utils.schemas import AgentType
from src.agents.analysts.base_analyst import (
    create_analyst_node,
    compute_date_range,
)

# ── Horizon-specific focus instructions ─────────────────────────────────────

HORIZON_FOCUS = {
    "long_term": (
        "HORIZON: Long-term (1+ years). "
        "Focus on secular macro regime shifts, structural interest-rate "
        "and inflation trends, geopolitical risk cycles, and long-run "
        "sector sensitivity to macro factors."
    ),
    "medium_term": (
        "HORIZON: Medium-term (3-12 months). "
        "Focus on the current business cycle phase, monetary policy trajectory, "
        "earnings cycle positioning, sector rotation signals, and "
        "upcoming macro data releases."
    ),
    "short_term": (
        "HORIZON: Short-term (days to weeks). "
        "Focus on imminent policy announcements, surprise macro prints, "
        "geopolitical flash events, insider transaction clusters, and "
        "event-driven sentiment spikes in company news."
    ),
}

# ── Phase 1 system prompt (plan — decide WHAT to compute) ───────────────────

PHASE1_PROMPT = (
    "You are a senior macroeconomist. Review the GDP, employment, inflation, "
    "and central bank data. Determine which regime classification models, "
    "sensitivity analyses, and sector rotation signals should be computed. "
    "Do NOT form a thesis yet — only plan the quantitative work."
)

# ── Phase 3 system prompt (thesis — interpret results) ──────────────────────

PHASE3_PROMPT = (
    "You are a senior macroeconomist. You have received regime classification "
    "outputs, sensitivity analyses, and sector signals. Interpret the macro "
    "environment's implications for this stock. Cite specific model outputs."
)


# ── Data gathering (deterministic, no LLM) ──────────────────────────────────


def gather_macro_data(ticker: str, trade_date: str, lookback_days: int) -> dict:
    """Fetch macro news, company news, stock data, and insider transactions."""
    from src.agents.utils.agent_utils import (
        get_global_news,
        get_insider_transactions,
        get_news,
        get_stock_data,
    )

    start_date, end_date = compute_date_range(trade_date, lookback_days)

    data = {}
    data["stock_data"] = get_stock_data.invoke(
        {"symbol": ticker, "start_date": start_date, "end_date": end_date}
    )
    data["company_news"] = get_news.invoke(
        {"ticker": ticker, "start_date": start_date, "end_date": end_date}
    )
    data["global_news"] = get_global_news.invoke(
        {"curr_date": trade_date, "look_back_days": min(lookback_days, 30), "limit": 10}
    )
    data["insider_transactions"] = get_insider_transactions.invoke(
        {"ticker": ticker}
    )
    return data


# ── Node factory ─────────────────────────────────────────────────────────────


def create_news_analyst(reasoning_llm, code_agent, research_depth="medium"):
    """Create a news/macro analyst node for the outer AgentState graph.

    Parameters
    ----------
    reasoning_llm  : LangChain chat model for Phase 1 (plan) and Phase 3 (thesis)
    code_agent     : CodeValidationAgent instance for Phase 2 (compute)
    research_depth : "shallow" | "medium" | "deep"
    """
    analyst_config = {
        "agent_type": AgentType.MACRO,
        "state_key": "news_report",
        "gather_fn": gather_macro_data,
        "phase1_system_prompt": PHASE1_PROMPT,
        "phase3_system_prompt": PHASE3_PROMPT,
        "horizon_focus": HORIZON_FOCUS,
    }

    return create_analyst_node(
        reasoning_llm, code_agent, analyst_config, research_depth,
    )
