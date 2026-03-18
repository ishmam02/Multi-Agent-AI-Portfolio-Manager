import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"

        past_reflections_section = f"""

PAST TRADE REFLECTIONS:
The following are lessons from similar situations you have traded in before. Use these to avoid repeating mistakes and to refine your current proposal:
{past_memory_str}""" if past_memory_str.strip() else ""

        context = {
            "role": "user",
            "content": f"""The research team has completed their analysis of {company_name}. Below are the consolidated reports and the investment plan derived from the bull/bear debate. Use these as the foundation for your trade proposal.

PROPOSED INVESTMENT PLAN:
{investment_plan}

MARKET RESEARCH REPORT:
{market_research_report}

SOCIAL MEDIA SENTIMENT REPORT:
{sentiment_report}

LATEST NEWS REPORT:
{news_report}

COMPANY FUNDAMENTALS REPORT:
{fundamentals_report}

Based on the above, draft your full trade proposal for {company_name}.""",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a senior institutional trader drafting a formal trade proposal for submission to the risk management desk. Your proposal must be structured, decisive, and detailed enough for the risk analysts to evaluate, challenge, and ultimately refine into an executable order.

Write your proposal as a cohesive, flowing document — not a form or checklist. Present your reasoning naturally, as you would in a verbal pitch to a portfolio manager, but ensure you cover all of the following elements within your narrative:

1. TRADE THESIS — State your directional conviction and the core reasoning behind it. What is the setup? Why now? What edge do you see that the market may be mispricing?

2. CONVICTION & SIZING — How confident are you? (High / Moderate / Low conviction.) What portfolio allocation do you recommend as a percentage? Scale your position size to your conviction — high conviction justifies larger positions, low conviction demands smaller or no exposure.

3. ENTRY STRATEGY — How should the position be initiated? A single market order for immediacy, or a limit order at a specific level? Should the entry be scaled in (e.g., 50% now, 50% on a pullback to a support level)? If scaling, specify the price levels and allocation splits.

4. PRICE TARGETS & EXIT PLAN — Where do you take profit? Specify at least one price target with reasoning (technical level, valuation multiple, catalyst timeline). If appropriate, propose a tiered exit (e.g., sell 1/3 at target 1, 1/3 at target 2, trail the rest).

5. STOP-LOSS & DOWNSIDE PROTECTION — Where is the trade invalidated? Specify a stop-loss level and whether it should be a hard stop (stop-market), a stop-limit, or a trailing stop. Explain why this level was chosen (technical support, percentage-based max loss, volatility band, etc.). If proposing a bracket order with both take-profit and stop-loss, say so explicitly.

6. RISK/REWARD ASSESSMENT — Quantify the expected risk/reward ratio (e.g., risking 5% to gain 15% = 3:1 R/R). Is this ratio attractive enough to justify the trade given current volatility and your conviction level?

7. KEY CATALYSTS & TIMELINE — What upcoming events or data releases could validate or invalidate this thesis? (Earnings, Fed decisions, product launches, regulatory rulings, etc.) When do you expect the thesis to play out?

8. RISKS & WHAT COULD GO WRONG — Proactively identify the top 2-3 risks to this trade. What would make you wrong? This gives the risk analysts specific points to challenge.

IMPORTANT GUIDELINES:
- Be specific with numbers. Don't say "set a stop-loss below support" — say "stop-loss at $142.50, just below the 200-day MA at $143.10."
- If the analysis supports holding and there is no compelling setup, explicitly state that no trade is warranted and explain why the risk/reward is insufficient to initiate.
- If the analysis is bearish, you may propose a short sell. State this clearly.
- Present your proposal conversationally and naturally. Avoid bullet-point formatting — write as if you are presenting to the desk.

Always conclude your proposal with a TRANSACTION SUMMARY block in this exact format:

TRANSACTION SUMMARY:
- Direction: BUY / SELL / SHORT SELL / HOLD
- Order Strategy: (e.g., Market, Limit at $X, Bracket with TP $X / SL $X, Scaled Entry, etc.)
- Entry: (price or "at market")
- Stop-Loss: (price and type, or "None" for HOLD)
- Target(s): (price level(s), or "N/A" for HOLD)
- Risk/Reward: (ratio, e.g., "1:3.2")
- Conviction: High / Moderate / Low
- Suggested Allocation: (percentage of portfolio)
- Timeframe: (expected holding period){past_reflections_section}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
