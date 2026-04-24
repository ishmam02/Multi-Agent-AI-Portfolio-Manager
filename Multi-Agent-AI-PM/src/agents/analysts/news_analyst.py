"""
News (Macro) Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather macro news, company news, insider transactions, and price data
  2. LLM plans which empirical news/sentiment signals to compute
  3. Code agent executes event-study and keyword-conditioned return analysis
  4. LLM interprets macro/news implications and forms a thesis

Empirical core:
  • Event-study: abnormal returns and abnormal volume after news articles
  • Keyword-conditioned returns: forward returns conditioned on article keyword content
  • Insider transaction signal: net buy ratio as a directional signal
  • Macro regime signal: bullish/bearish keyword ratio from global news
"""

import json

from src.agents.utils.schemas import AgentType
from src.agents.analysts.base_analyst import (
    create_analyst_node,
    compute_date_range,
)
from src.agents.prompts import load_prompt

# ── Load prompts from external markdown file ────────────────────────────────

_news_prompts = load_prompt("news_analyst")
HORIZON_FOCUS: dict[str, str] = _news_prompts["HORIZON_FOCUS"]  # type: ignore[assignment]
PHASE1_PROMPT: str = _news_prompts["PHASE1_PROMPT"]  # type: ignore[assignment]
PHASE2_PROMPT: str = _news_prompts["PHASE2_PROMPT"]  # type: ignore[assignment]
PHASE3_PROMPT: str = _news_prompts["PHASE3_PROMPT"]  # type: ignore[assignment]


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
        {"curr_date": trade_date, "look_back_days": min(lookback_days, 30)}
    )
    data["insider_transactions"] = get_insider_transactions.invoke({"ticker": ticker})
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
        "agent_type": AgentType.NEWS,
        "state_key": "news_report",
        "gather_fn": gather_macro_data,
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
        analyst_type="news",
        project_root=_project_root,
        verbose=True,
    )

    news_node = create_news_analyst(reasoning_llm, code_agent, RESEARCH_DEPTH)

    init_state = {
        "company_of_interest": TICKER,
        "trade_date": TRADE_DATE,
    }

    print(f"Running news analysis for {TICKER} on {TRADE_DATE} ...")
    result = news_node(init_state)

    report_json = result.get("news_report", "{}")
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
    print(f"  NEWS REPORT — {TICKER} @ {TRADE_DATE}")
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
