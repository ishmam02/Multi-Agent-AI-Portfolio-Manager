"""
Fundamentals Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather financial statements & fundamentals
  2. LLM plans which valuation metrics to compute
  3. Code agent computes them
  4. LLM interprets results and forms an investment thesis
"""

from src.agents.utils.schemas import AgentType
from src.agents.analysts.base_analyst import (
    create_analyst_node,
)

# ── Horizon-specific focus instructions ─────────────────────────────────────

HORIZON_FOCUS = {
    "long_term": (
        "HORIZON: Long-term (1+ years). "
        "Focus on intrinsic valuation (DCF, residual income), durable competitive "
        "advantages, secular revenue growth, capital allocation track record, "
        "and balance-sheet resilience over multiple business cycles."
    ),
    "medium_term": (
        "HORIZON: Medium-term (3-12 months). "
        "Focus on upcoming earnings catalysts, margin trajectory, guidance "
        "revisions, peer-relative valuation multiples (P/E, EV/EBITDA), and "
        "near-term capital structure changes."
    ),
    "short_term": (
        "HORIZON: Short-term (days to weeks). "
        "Focus on earnings-surprise risk, short-term free-cash-flow dynamics, "
        "imminent catalyst events (ex-dividend, lock-up expiry, index rebalance), "
        "and any anomalous insider activity."
    ),
}

# ── Phase 1 system prompt (plan — decide WHAT to compute) ───────────────────

PHASE1_PROMPT = (
    "You are a senior equity research analyst. Review the financial data, "
    "SEC filings, and earnings transcripts. Determine which valuation models, "
    "ratio analyses, and scoring frameworks should be computed to properly "
    "evaluate this company. Do NOT form a thesis yet — only plan the "
    "quantitative work."
)

# ── Phase 3 system prompt (thesis — interpret results) ──────────────────────

PHASE3_PROMPT = (
    "You are a senior equity research analyst. You have received computed "
    "valuation metrics, financial ratios, and scoring results. Interpret "
    "these numbers in context. Form an investment thesis on intrinsic value "
    "vs. market price. Cite specific computed metrics to support each claim."
)


# ── Data gathering (deterministic, no LLM) ──────────────────────────────────


def gather_fundamental_data(ticker: str, trade_date: str, lookback_days: int) -> dict:
    """Fetch financial statements and fundamentals for the given ticker."""
    from src.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_fundamentals,
        get_income_statement,
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
        "phase3_system_prompt": PHASE3_PROMPT,
        "horizon_focus": HORIZON_FOCUS,
    }

    return create_analyst_node(
        reasoning_llm, code_agent, analyst_config, research_depth,
    )
