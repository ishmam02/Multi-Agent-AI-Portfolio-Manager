"""
Social Media (Sentiment) Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather company news and insider transaction data
  2. LLM plans which sentiment scoring/cross-validation metrics to compute
  3. Code agent computes them
  4. LLM interprets behavioral signals and forms a sentiment thesis
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
        "Focus on structural narrative shifts, long-run brand sentiment "
        "trajectories, persistent institutional positioning changes, and "
        "multi-year insider accumulation or distribution patterns."
    ),
    "medium_term": (
        "HORIZON: Medium-term (3-12 months). "
        "Focus on sentiment trend vs. price divergence, earnings-driven "
        "sentiment cycles, analyst opinion momentum, and medium-term "
        "insider transaction clusters around catalysts."
    ),
    "short_term": (
        "HORIZON: Short-term (days to weeks). "
        "Focus on near-term sentiment extremes, viral news impact, "
        "abnormal mention-volume spikes, short-squeeze risk signals, "
        "and imminent insider filing deadlines."
    ),
}

# ── Phase 1 system prompt (plan — decide WHAT to compute) ───────────────────

PHASE1_PROMPT = (
    "You are a senior sentiment analyst. Review the news, social media, and "
    "insider transaction data. Determine which sentiment scoring methods and "
    "cross-validation metrics should be computed against quantitative proxies. "
    "Do NOT form a thesis yet — only plan the quantitative work."
)

# ── Phase 3 system prompt (thesis — interpret results) ──────────────────────

PHASE3_PROMPT = (
    "You are a senior sentiment analyst. You have received sentiment scores "
    "cross-validated against volume, IV, and short interest. Interpret "
    "behavioral signals. Form a thesis on market sentiment positioning. "
    "Cite specific cross-validation results."
)


# ── Data gathering (deterministic, no LLM) ──────────────────────────────────


def gather_sentiment_data(ticker: str, trade_date: str, lookback_days: int) -> dict:
    """Fetch company news and insider transactions for sentiment analysis."""
    from src.agents.utils.agent_utils import (
        get_insider_transactions,
        get_news,
    )

    start_date, end_date = compute_date_range(trade_date, lookback_days)

    data = {}
    data["company_news"] = get_news.invoke(
        {"ticker": ticker, "start_date": start_date, "end_date": end_date}
    )
    data["insider_transactions"] = get_insider_transactions.invoke(
        {"ticker": ticker}
    )
    return data


# ── Node factory ─────────────────────────────────────────────────────────────


def create_social_media_analyst(reasoning_llm, code_agent, research_depth="medium"):
    """Create a sentiment analyst node for the outer AgentState graph.

    Parameters
    ----------
    reasoning_llm  : LangChain chat model for Phase 1 (plan) and Phase 3 (thesis)
    code_agent     : CodeValidationAgent instance for Phase 2 (compute)
    research_depth : "shallow" | "medium" | "deep"
    """
    analyst_config = {
        "agent_type": AgentType.SENTIMENT,
        "state_key": "sentiment_report",
        "gather_fn": gather_sentiment_data,
        "phase1_system_prompt": PHASE1_PROMPT,
        "phase3_system_prompt": PHASE3_PROMPT,
        "horizon_focus": HORIZON_FOCUS,
    }

    return create_analyst_node(
        reasoning_llm, code_agent, analyst_config, research_depth,
    )
