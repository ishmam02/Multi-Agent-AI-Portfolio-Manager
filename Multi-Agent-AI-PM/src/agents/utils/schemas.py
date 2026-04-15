from datetime import datetime
from enum import Enum
from typing import Any, List

from pydantic import BaseModel


class AgentType(str, Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACRO = "macro"
    SENTIMENT = "sentiment"


class Metrics(BaseModel):
    metric_name: str
    term: str  # "long_term" | "medium_term" | "short_term"
    metric_interpretation: str  # What this metric IS and what it measures
    metric_rationale: str  # Reason for choosing this metric
    computation_instruction: str


class ComputedMetric(BaseModel):
    metric_name: str  # e.g. "dcf_fair_value", "rsi", "hy_spread"
    term: str  # "long_term" | "medium_term" | "short_term"
    value: Any  # float, dict, or nested structure (from Phase 2)
    metric_interpretation: (
        str  # What this metric IS and what it measures (filled in Phase 1)
    )
    value_interpretation: str  # What this specific VALUE means for this stock right now (filled in Phase 3)
    computation_trace_id: str  # links to the code that produced it


class ComputationTrace(BaseModel):
    trace_id: str
    code: str  # The Python code that was executed
    inputs: dict  # Data fed into the code
    output: Any  # Raw code output


class Catalyst(BaseModel):
    catalyst: str
    term: str  # "long_term" | "medium_term" | "short_term"
    metric_name: str
    computation_trace_id: str


class Risk(BaseModel):
    risk: str
    term: str  # "long_term" | "medium_term" | "short_term"
    metric_name: str
    computation_trace_id: str


class Citation(BaseModel):
    claim: str
    source: str
    source_uuid: str

class ContributingFactor(BaseModel):
    factor_uuid: str
    factor_name: str
    weight_in_analysis: float
    current_ic: float  # Information coefficient
    decay_status: str


class HorizonValues(BaseModel):
    long_term: float
    medium_term: float
    short_term: float


class HorizonThesis(BaseModel):
    long_term: str
    medium_term: str
    short_term: str


class HorizonTraceIds(BaseModel):
    long_term: str  # UUID of the ComputationTrace that produced this horizon's value
    medium_term: str
    short_term: str



class ResearchReport(BaseModel):
    """Common output schema for all analyst agents."""

    # ── Identity ──
    ticker: str
    agent_type: AgentType
    timestamp: datetime

    # ── Phase 1: Computation Plan (LLM decides WHAT to compute) ──
    metrics_selected: List[Metrics]  # full metric plans selected by the LLM

    # ── Phase 2: Symbolic (Code-Computed Numerics) ──
    mu: HorizonValues  # Expected return per horizon
    mu_trace_id: HorizonTraceIds  # Trace UUIDs for mu per horizon
    sigma_contribution: HorizonValues  # Volatility contribution per horizon
    sigma_trace_id: HorizonTraceIds  # Trace UUIDs for sigma_contribution per horizon
    computed_metrics: List[ComputedMetric]  # Flexible list, not fixed dict

    # ── Phase 3: Investment Thesis (LLM interprets Phase 2 results) ──
    investment_thesis: HorizonThesis  # Formed AFTER seeing the numbers, per horizon
    conviction: HorizonValues  # [-1.0, +1.0] per horizon
    key_catalysts: List[Catalyst]
    key_risks: List[Risk]

    # ── Provenance ──
    source_uuids: List[str]
    computation_traces: List[ComputationTrace]
    citation_chain: List[
        Citation
    ]  # Each claim linked to a computed metric or data source
    contributing_factors: List[ContributingFactor]
