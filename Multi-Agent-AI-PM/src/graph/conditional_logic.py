# TradingAgents/graph/conditional_logic.py

from src.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow.

    With the 3-phase analyst subgraph architecture, all tool looping is
    internal to each analyst node.  The parent graph no longer needs
    per-analyst conditional routing.
    """

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds


def get_next_node(state: AgentState) -> str:
    """Simple router that always proceeds to synthesis.

    Kept for backward compatibility; the actual routing is now handled
    by ``should_proceed_to_synthesis`` inside ``setup.py``.
    """
    return "Synthesis Agent"
