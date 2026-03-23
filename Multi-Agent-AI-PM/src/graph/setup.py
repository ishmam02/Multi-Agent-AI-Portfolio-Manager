# TradingAgents/graph/setup.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from src.agents import *
from src.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        trader_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.trader_memory = trader_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(self.quick_thinking_llm)
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm
            )
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(self.quick_thinking_llm)
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

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

        # Barrier node: no-op, just a convergence point for all analyst branches
        def analyst_sync(state):
            return {}

        # Only proceed to Trader once every selected analyst has written its report
        def should_proceed_to_trader(state):
            if all(state.get(k) for k in required_reports):
                return "Trader"
            return END

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Barrier + trader nodes
        workflow.add_node("Analyst Sync", analyst_sync)
        workflow.add_node("Trader", trader_node)

        # Define edges
        # Fan-out: all analysts start in parallel from START
        for analyst_type in analyst_nodes:
            workflow.add_edge(START, f"{analyst_type.capitalize()} Analyst")

        # Each analyst loops with its tools, then routes to the barrier when done
        for analyst_type in analyst_nodes:
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"

            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                {current_tools: current_tools, "Trader": "Analyst Sync"},
            )
            workflow.add_edge(current_tools, current_analyst)

        # Barrier routes to Trader only when all reports are present, else terminates branch
        workflow.add_conditional_edges(
            "Analyst Sync", should_proceed_to_trader, ["Trader", END]
        )

        workflow.add_edge("Trader", END)

        # Compile and return
        return workflow.compile()
