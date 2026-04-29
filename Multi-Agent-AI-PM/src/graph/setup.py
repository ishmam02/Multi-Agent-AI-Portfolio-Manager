# TradingAgents/graph/setup.py

import time
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
        quick_thinking_llm: ChatOpenAI = None,
        research_depth: str = "medium",
        horizons=("long_term", "medium_term", "short_term"),
    ):
        """Initialize with required components.

        Parameters
        ----------
        reasoning_llm      : LLM for Phase 1 (plan) and Phase 3 (thesis)
        code_agents        : Dict mapping analyst key -> CodeValidationAgent
                             e.g. {"market": CodeValidationAgent(..., analyst_type="market")}
        quick_thinking_llm : LLM for the synthesis node (falls back to reasoning_llm)
        research_depth     : "shallow" | "medium" | "deep"
        horizons           : Tuple of horizon strings passed to the synthesis agent.
        """
        self.reasoning_llm = reasoning_llm
        self.code_agents = code_agents
        self.quick_thinking_llm = quick_thinking_llm or reasoning_llm
        self.research_depth = research_depth
        self.horizons = horizons

    def setup_graph(
        self, selected_analysts=["market", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market/technical analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes — each is fully self-contained
        analyst_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.reasoning_llm, self.code_agents["market"], self.research_depth,
                active_horizons=self.horizons,
            )

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.reasoning_llm, self.code_agents["fundamentals"], self.research_depth,
                active_horizons=self.horizons,
            )

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.reasoning_llm, self.research_depth, self.horizons
            )

        synthesis_node = create_synthesis_agent(
            self.reasoning_llm,
            horizons=self.horizons,
            research_depth=self.research_depth,
        )

        # Map each analyst type to its report field in AgentState
        report_keys_map = {
            "market": "market_report",
            "fundamentals": "fundamentals_report",
            "news": "news_report",
        }
        required_reports = [
            report_keys_map[a] for a in selected_analysts if a in report_keys_map
        ]

        # Barrier node: convergence point for all analyst branches
        def analyst_sync(state):
            return {}

        # Only proceed to Synthesis once every selected analyst has written its report
        def should_proceed_to_synthesis(state):
            if all(state.get(k) for k in required_reports):
                return "Synthesis Agent"
            return END

        # Build the workflow graph
        workflow = StateGraph(AgentState)

        # Add analyst nodes with exception handling and retry logic
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)

        # Barrier + synthesis
        workflow.add_node("Analyst Sync", analyst_sync)
        workflow.add_node("Synthesis Agent", synthesis_node)

        # Fan-out: all analysts start in parallel from START
        for analyst_type in analyst_nodes:
            workflow.add_edge(START, f"{analyst_type.capitalize()} Analyst")

        # Each analyst goes directly to the barrier (no tool loops)
        for analyst_type in analyst_nodes:
            workflow.add_edge(
                f"{analyst_type.capitalize()} Analyst", "Analyst Sync"
            )

        # Barrier routes to Synthesis when all reports are present
        workflow.add_conditional_edges(
            "Analyst Sync", should_proceed_to_synthesis, ["Synthesis Agent", END]
        )

        workflow.add_edge("Synthesis Agent", END)

        return workflow.compile()
