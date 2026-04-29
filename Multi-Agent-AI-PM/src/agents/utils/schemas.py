from datetime import datetime
from enum import Enum
from typing import Any, List

from pydantic import BaseModel


class AgentType(str, Enum):
    FUNDAMENTAL = "fundamental"
    MARKET = "market"
    NEWS = "news"


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
    metric_name: str = ""  # code-computed analysts: the supporting metric
    computation_trace_id: str = ""  # code-computed analysts: trace UUID
    claim: str = ""  # news analyst: the supporting claim text
    source_uuid: str = ""  # news analyst: links to citation_chain source_uuid


class Risk(BaseModel):
    risk: str
    term: str  # "long_term" | "medium_term" | "short_term"
    metric_name: str = ""  # code-computed analysts: the supporting metric
    computation_trace_id: str = ""  # code-computed analysts: trace UUID
    claim: str = ""  # news analyst: the supporting claim text
    source_uuid: str = ""  # news analyst: links to citation_chain source_uuid


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
    mu_trace_id: HorizonTraceIds | None = None  # Trace UUIDs for mu per horizon
    sigma_contribution: HorizonValues  # Volatility contribution per horizon
    sigma_trace_id: HorizonTraceIds | None = None  # Trace UUIDs for sigma_contribution per horizon
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


class AnalystWeights(BaseModel):
    """Per-analyst weights for a single horizon. Must sum to 1.0 (enforced in code)."""

    fundamental: float = 0.333
    market: float = 0.333
    news: float = 0.334


class HorizonWeights(BaseModel):
    """Per-horizon analyst weight allocations."""

    long_term: AnalystWeights = AnalystWeights()
    medium_term: AnalystWeights = AnalystWeights()
    short_term: AnalystWeights = AnalystWeights()


class CrossSignalConflict(BaseModel):
    """Documented contradiction between two analyst signals."""

    analyst_a: AgentType
    analyst_b: AgentType
    horizon: str  # "long_term" | "medium_term" | "short_term"
    conflict_description: str
    resolution_status: str  # "resolved" | "unresolved"
    conviction_penalty: float  # -1.0 to 0.0


class CompositeSignal(BaseModel):
    """Structured output of the Synthesis Agent — canonical Layer 2 signal."""

    # ── Identity ──
    ticker: str
    timestamp: datetime

    # ── Per-horizon composite metrics ──
    mu_composite: HorizonValues  # blended per-horizon expected return
    sigma_composite: HorizonValues  # blended per-horizon volatility

    # ── Unified final signal ──
    mu_final: float  # unified expected return after horizon blending
    sigma_final: float  # unified volatility
    conviction_final: float  # [0, +1.0]

    # ── Weighting provenance ──
    analyst_weights: HorizonWeights  # per-analyst weights per horizon
    horizon_blend_weights: (
        HorizonValues  # weights used to blend long/medium/short into final
    )
    weighting_rationale: str

    # ── Composite investment thesis (blended from all analysts) ──
    composite_thesis: HorizonThesis  # unified narrative per horizon with inline trace citations

    # ── Conflict documentation ──
    cross_signal_conflicts: List[CrossSignalConflict]
    unresolved_penalty: float  # total conviction reduction from unresolved conflicts

    # ── Source tracking ──
    source_reports: List[str]  # list of analyst AgentType values that contributed