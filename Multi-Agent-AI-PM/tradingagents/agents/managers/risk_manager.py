import time
import json
import os
import logging


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]
        risk_profile = state.get("risk_profile")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        risk_profile_str = ""
        if risk_profile:
            experience_desc = {
                "No experience": "no prior investment experience — unfamiliar with market mechanics and risk",
                "Beginner": "some basic investment experience — aware of fundamentals but limited exposure",
                "Intermediate": "moderate investment experience — comfortable with market cycles and portfolio concepts",
                "Expert": "extensive investment experience — sophisticated investor with deep market knowledge",
            }.get(risk_profile["experience"], risk_profile["experience"])

            goal_desc = {
                "Capital Preservation": "capital preservation — minimizing loss is the top priority over returns",
                "Balanced Growth": "balanced growth — seeks moderate returns while managing downside risk",
                "Aggressive Growth": "aggressive growth — maximizes long-term returns and accepts higher volatility",
            }.get(risk_profile["goal"], risk_profile["goal"])

            risk_desc = {
                "Very Conservative": "extremely risk-averse — only suitable for near risk-free positions",
                "Conservative": "low risk tolerance — prefers stable, defensive positions",
                "Moderate": "moderate risk tolerance — accepts some volatility for reasonable returns",
                "Aggressive": "high risk tolerance — comfortable with significant drawdowns for growth",
                "Very Aggressive": "maximum risk tolerance — willing to accept extreme volatility for maximum upside",
            }.get(risk_profile["risk"], risk_profile["risk"])

            income_contrib = {
                "Under $25k": "severely limits position sizing and eliminates leverage; losses cannot be replenished from income",
                "$25k-$49k": "constrains position sizing; recovery from significant losses would take years of savings",
                "$50k-$99k": "supports moderate position sizes; can absorb small losses but large drawdowns are career-threatening",
                "$100k-$199k": "allows meaningful position sizes; can tolerate moderate drawdowns and rebuild over 1-2 years",
                "$200k-$499k": "enables larger allocations; high income provides a safety net for aggressive strategies",
                "$500k+": "removes income as a constraint; investor can absorb significant losses without lifestyle impact",
            }.get(risk_profile["income"], "contributes to overall risk capacity")

            net_worth_contrib = {
                "Under $5k": "portfolio is the investor's entire financial cushion — any loss is existential; only near-zero-risk positions are appropriate",
                "$5k-$24k": "very limited capital base — losses directly threaten financial stability; preservation is paramount",
                "$25k-$49k": "small capital base — moderate losses would be painful but survivable; size positions conservatively",
                "$50k-$99k": "growing capital base — can absorb some volatility but a 30%+ drawdown materially impacts financial goals",
                "$100k-$249k": "solid capital base — provides buffer for moderate risk-taking; diversification becomes meaningful",
                "$250k-$499k": "substantial capital — can weather significant market downturns; supports a wider range of strategies",
                "$500k-$999k": "large capital base — losses are percentage-based concerns not survival concerns; enables sophisticated strategies",
                "$1M+": "wealth provides significant cushion — risk tolerance is driven by goals and psychology, not financial necessity",
            }.get(risk_profile["net_worth"], "contributes to overall risk capacity")

            period_contrib = {
                "1-5 years": "short horizon — no time to recover from drawdowns; favors capital preservation and high-conviction-only positions",
                "5-10 years": "medium horizon — can ride out one market cycle but not two; balance growth with downside protection",
                "10-20 years": "long horizon — multiple cycles to recover; can tolerate higher volatility for compounding gains",
                "20-40 years": "very long horizon — short-term drawdowns are noise; growth-oriented strategies are strongly favored",
                "40-60 years": "generational horizon — maximum compounding runway; aggressive growth is mathematically optimal if psychologically tolerable",
                "60+ years": "ultra-long horizon — virtually any drawdown is recoverable; risk tolerance should be driven by psychology not math",
            }.get(risk_profile["period"], "shapes recovery window and strategy horizon")

            risk_profile_str = f"""
**Investor Risk Profile — You MUST align your decision to this profile.**
Each dimension below was derived from psychometric questionnaires, behavioral data, and macroeconomic sentiment. Understand what each contributes to the risk assessment:

- **Experience: {risk_profile["experience"]}** ({experience_desc})
  → Shapes how the investor *interprets* volatility. {risk_profile["experience"]} experience means {"market swings feel unfamiliar and threatening — avoid complex or volatile positions" if risk_profile["experience"] in ("No experience", "Beginner") else "the investor can contextualize drawdowns within historical patterns — more nuanced strategies are viable"}.

- **Annual Income: {risk_profile["income"]}**
  → Determines *replenishment capacity* — how quickly losses can be offset by new capital. {income_contrib}.

- **Net Worth: {risk_profile["net_worth"]}**
  → Determines *loss absorption capacity* — the financial buffer available before losses become destructive. {net_worth_contrib}.

- **Investment Goal: {risk_profile["goal"]}** ({goal_desc})
  → Sets the *optimization target*. {"The goal is to NOT lose money — downside protection overrides return maximization" if risk_profile["goal"] == "Capital Preservation" else "The goal balances returns against risk — neither pure growth nor pure preservation" if risk_profile["goal"] == "Balanced Growth" else "The goal is to maximize returns — accept higher volatility as the cost of compounding"}.

- **Risk Tolerance: {risk_profile["risk"]}** ({risk_desc})
  → Captures *psychological capacity* for loss — derived from behavioral responses, not just stated preference. This is the investor's emotional floor. {"Even if the math supports a trade, this investor will panic-sell at the first drawdown — position accordingly" if risk_profile["risk"] in ("Very Conservative", "Conservative") else "This investor can hold through moderate turbulence but has limits — size positions to avoid testing those limits" if risk_profile["risk"] == "Moderate" else "This investor has demonstrated comfort with significant drawdowns — psychology is not the binding constraint"}.

- **Investment Period: {risk_profile["period"]}**
  → Defines the *recovery window* — how many market cycles the investor can wait through. {period_contrib}.

Your recommendation must reflect the **combined effect** of all six dimensions. A technically valid trade that contradicts even one binding constraint (e.g., conservative risk tolerance + short horizon = no speculative positions, regardless of upside) is the wrong trade for this investor.
"""

        past_mistakes_guideline = f"4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money." if past_memory_str.strip() else ""

        prompt = f"""You are the Chief Risk Officer making the final trade decision. The trader has submitted a structured trade proposal and three risk analysts — Aggressive, Neutral, and Conservative — have debated it. Your job is to render a final verdict: approve the trade as-is, modify its parameters (sizing, entry, stops, targets), or reject it entirely.

Your decision must be tailored to this specific investor. A technically valid trade that contradicts the investor's profile is the wrong trade. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.
{risk_profile_str}
TRADER'S ORIGINAL PROPOSAL:
{trader_plan}

---

ANALYSTS DEBATE HISTORY:
{history}

---

Guidelines for Decision-Making:
1. **Evaluate the Debate**: Extract the strongest points from each analyst. Which arguments are most relevant to this investor's profile?
2. **Stress-Test the Trader's Parameters**: The trader proposed specific entry levels, stop-losses, targets, position sizing, and conviction. For each parameter, determine whether it is appropriate for this investor or needs adjustment. For example:
   - Is the proposed allocation too large for this investor's risk tolerance or net worth?
   - Is the stop-loss wide enough to avoid noise but tight enough to protect against the investor's maximum tolerable loss?
   - Are the price targets realistic given the investor's time horizon?
   - Does the order strategy (market vs. limit vs. bracket) match the investor's experience level?
3. **Render Your Verdict**: Either approve the trader's proposal, modify specific parameters, or reject the trade entirely.
{past_mistakes_guideline}

Deliverables — present these as a flowing narrative, not a form:

1. **Debate Summary** — The key arguments from each analyst and which you found most compelling for this investor.

2. **Investor Profile Alignment** — Walk through each of the six profile dimensions and explain how it individually contributed to your final decision. For each dimension, state: (1) the investor's value, (2) the constraint or opportunity it creates, and (3) how it influenced which analyst's arguments were weighted more heavily. Example:
   - "**Experience (Beginner):** Limited experience means this investor would likely panic-sell during the volatility the Aggressive analyst accepts as normal. This deprioritized the Aggressive case."
   - "**Risk Tolerance (Conservative):** The investor's psychological floor rules out positions with >10% downside. This made the Conservative analyst's hedging argument the binding constraint."
   - "**Investment Period (1-5 years):** Short horizon leaves no room to recover from a drawdown, reinforcing the Conservative position over the Aggressive analyst's long-term thesis."
   Do this for ALL six dimensions (Experience, Income, Net Worth, Goal, Risk Tolerance, Period), not just the ones that dominate.

3. **Final Trade Decision** — Your approved, modified, or rejected trade. If approving or modifying, conclude with an APPROVED TRANSACTION SUMMARY in this format:

APPROVED TRANSACTION SUMMARY:
- Direction: BUY / SELL / SHORT SELL / HOLD
- Order Strategy: (e.g., Market, Limit at $X, Bracket with TP $X / SL $X, Scaled Entry, etc.)
- Entry: (price or "at market")
- Stop-Loss: (price and type, or "None" for HOLD)
- Target(s): (price level(s), or "N/A" for HOLD)
- Risk/Reward: (ratio)
- Conviction: High / Moderate / Low
- Approved Allocation: (percentage of portfolio — may differ from trader's suggestion)
- Timeframe: (expected holding period)
- Modifications from Trader's Proposal: (list any parameter changes and why, or "None — approved as submitted")

If rejecting the trade entirely, state APPROVED TRANSACTION SUMMARY with Direction: HOLD and explain in Modifications why the trade was rejected."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state[
                "current_aggressive_response"
            ],
            "current_conservative_response": risk_debate_state[
                "current_conservative_response"
            ],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
