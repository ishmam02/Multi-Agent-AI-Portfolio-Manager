from typing import Annotated, Sequence
from datetime import date, timedelta, datetime
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from src.agents import *
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START, MessagesState


class RiskProfile(TypedDict):
    experience: Annotated[str, "Investment experience level"]
    income: Annotated[str, "Annual income bracket"]
    net_worth: Annotated[str, "Net worth bracket"]
    goal: Annotated[str, "Investment goal"]
    risk: Annotated[str, "Risk tolerance level"]
    period: Annotated[str, "Investment time horizon"]


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]

    # user risk profile
    risk_profile: Annotated[RiskProfile, "User's investment risk profile"]

    sender: Annotated[str, "Agent that sent this message"]

    # research step
    market_report: Annotated[str, "Report from the Market Analyst"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Researcher"]

    composite_signal: Annotated[str, "JSON-serialized CompositeSignal from Synthesis Agent"]
