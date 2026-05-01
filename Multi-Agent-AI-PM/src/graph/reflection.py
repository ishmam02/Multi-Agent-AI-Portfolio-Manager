# src/graph/reflection.py

import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
You are an expert financial analyst tasked with reviewing trading decisions/analysis and providing a comprehensive, step-by-step analysis.
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement, adhering strictly to the following guidelines:

1. Reasoning:
   - For each trading decision, determine whether it was correct or incorrect. A correct decision results in an increase in returns, while an incorrect decision does the opposite.
   - Analyze the contributing factors to each success or mistake. Consider:
     - Market intelligence.
     - Technical indicators.
     - Technical signals.
     - Price movement analysis.
     - Overall market data analysis
     - News analysis.
     - Social media and sentiment analysis.
     - Fundamental data analysis.
     - Weight the importance of each factor in the decision-making process.

2. Improvement:
   - For any incorrect decisions, propose revisions to maximize returns.
   - Provide a detailed list of corrective actions or improvements, including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date).

3. Summary:
   - Summarize the lessons learned from the successes and mistakes.
   - Highlight how these lessons can be adapted for future trading scenarios and draw connections between similar situations to apply the knowledge gained.

4. Query:
   - Extract key insights from the summary into a concise sentence of no more than 1000 tokens.
   - Ensure the condensed sentence captures the essence of the lessons and reasoning for easy reference.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable. You will also be given objective descriptions of the market from a price movements, technical indicator, news, and sentiment perspective to provide more context for your analysis.
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state.

        Parses ResearchReport JSON strings to build a structured summary.
        """
        from src.agents.utils.schemas import ResearchReport

        parts = []
        for key in ["market_report", "fundamentals_report"]:
            raw = current_state.get(key, "")
            if not raw:
                continue
            try:
                report = ResearchReport.model_validate_json(raw)
                parts.append(
                    f"{report.agent_type.value.upper()} REPORT:\n"
                    f"  mu: LT={report.mu.long_term:.4f} MT={report.mu.medium_term:.4f} ST={report.mu.short_term:.4f}\n"
                    f"  conviction: LT={report.conviction.long_term:.4f} MT={report.conviction.medium_term:.4f} ST={report.conviction.short_term:.4f}\n"
                    f"  thesis (short): {report.investment_thesis.short_term[:200]}"
                )
            except Exception:
                parts.append(f"{key}: {raw[:500]}")

        return "\n\n".join(parts) if parts else "No reports available."

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on synthesis agent's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        composite_raw = current_state.get("composite_signal", "")

        composite_summary = "No composite signal available."
        if composite_raw:
            try:
                from src.agents.utils.schemas import CompositeSignal
                cs = CompositeSignal.model_validate_json(composite_raw)
                composite_summary = (
                    f"Synthesis Agent Decision:\n"
                    f"  mu_final={cs.mu_final:.4f}\n"
                    f"  sigma_final={cs.sigma_final:.4f}\n"
                    f"  conviction={cs.conviction_final:.4f}\n"
                    f"  conflicts={len(cs.cross_signal_conflicts)}\n"
                    f"  rationale: {cs.weighting_rationale[:300]}"
                )
            except Exception as exc:
                composite_summary = f"Failed to parse composite signal: {exc}"

        result = self._reflect_on_component(
            "SYNTHESIS", composite_summary, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])
