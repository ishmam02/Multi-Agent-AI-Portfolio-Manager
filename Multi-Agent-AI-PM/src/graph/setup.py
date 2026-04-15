# TradingAgents/graph/setup.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START

from src.agents import *
from src.agents.utils.agent_states import AgentState


class GraphSetup:
    """Handles the setup and configuration of the agent graph.

    With the 3-phase analyst architecture, each analyst node is fully
    self-contained (data gathering + LLM planning + code execution +
    thesis formation).  No tool nodes or conditional tool-call routing
    are needed — the subgraph handles all internal looping.
    """

    def __init__(
        self,
        reasoning_llm: ChatOpenAI,
        code_agents: Dict[str, Any],
        trader_memory,
        quick_thinking_llm: ChatOpenAI = None,
        research_depth: str = "medium",
    ):
        """Initialize with required components.

        Parameters
        ----------
        reasoning_llm      : LLM for Phase 1 (plan) and Phase 3 (thesis)
        code_agents        : Dict mapping analyst key -> CodeValidationAgent
                             e.g. {"market": CodeValidationAgent(..., analyst_type="technical")}
        trader_memory      : FinancialSituationMemory for the trader
        quick_thinking_llm : LLM for the trader node (falls back to reasoning_llm)
        research_depth     : "shallow" | "medium" | "deep"
        """
        self.reasoning_llm = reasoning_llm
        self.code_agents = code_agents
        self.trader_memory = trader_memory
        self.quick_thinking_llm = quick_thinking_llm or reasoning_llm
        self.research_depth = research_depth

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market/technical analyst
                - "social": Social media/sentiment analyst
                - "news": News/macro analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes — each is fully self-contained
        analyst_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.reasoning_llm, self.code_agents["market"], self.research_depth,
            )

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.reasoning_llm, self.code_agents["social"], self.research_depth,
            )

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.reasoning_llm, self.code_agents["news"], self.research_depth,
            )

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.reasoning_llm, self.code_agents["fundamentals"], self.research_depth,
            )

        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Map each analyst type to its report field in AgentState
        report_keys_map = {
            "market": "market_report",
            "social": "sentiment_report",
            "news": "news_report",
            "fundamentals": "fundamentals_report",
        }
        required_reports = [
            report_keys_map[a] for a in selected_analysts if a in report_keys_map
        ]

        # Barrier node: convergence point for all analyst branches
        def analyst_sync(state):
            return {}

        # Only proceed to Trader once every selected analyst has written its report
        def should_proceed_to_trader(state):
            if all(state.get(k) for k in required_reports):
                return "Trader"
            return END

        # Build the workflow graph
        workflow = StateGraph(AgentState)

        # Add analyst nodes
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)

        # Barrier + trader
        workflow.add_node("Analyst Sync", analyst_sync)
        workflow.add_node("Trader", trader_node)

        # Fan-out: all analysts start in parallel from START
        for analyst_type in analyst_nodes:
            workflow.add_edge(START, f"{analyst_type.capitalize()} Analyst")

        # Each analyst goes directly to the barrier (no tool loops)
        for analyst_type in analyst_nodes:
            workflow.add_edge(
                f"{analyst_type.capitalize()} Analyst", "Analyst Sync"
            )

        # Barrier routes to Trader when all reports are present
        workflow.add_conditional_edges(
            "Analyst Sync", should_proceed_to_trader, ["Trader", END]
        )

        workflow.add_edge("Trader", END)

        return workflow.compile()
