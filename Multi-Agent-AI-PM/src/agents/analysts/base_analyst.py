"""
Base analyst: 3-phase subgraph factory and horizon-merge utilities.

Provides:
  build_3phase_subgraph(reasoning_llm, code_agent, analyst_config)
      -> compiled LangGraph StateGraph

  create_analyst_node(reasoning_llm, code_agent, analyst_config, research_depth)
      -> callable analyst node for the outer AgentState graph

The subgraph runs once per horizon (long / medium / short).  The outer
analyst node invokes it three times and merges the results into a single
ResearchReport.

Each concrete analyst file (fundamentals_analyst.py, market_analyst.py, etc.)
provides its own:
  - Phase 1 / Phase 3 system prompts
  - Data gathering function
  - analyst_config dict
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import uuid

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.dataflows.y_finance import get_fundamentals
from src.agents.utils.schemas import (
    AgentType,
    Catalyst,
    Risk,
    ComputationTrace,
    ComputedMetric,
    ContributingFactor,
    HorizonThesis,
    HorizonTraceIds,
    HorizonValues,
    Metrics,
    ResearchReport,
)


# ── Project root (for log directory) ─────────────────────────────────────────
# base_analyst.py → analysts/ → agents/ → src/ → Multi-Agent-AI-PM/
def _find_project_root() -> str:
    path = os.path.abspath(__file__)
    for _ in range(10):
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
        if path == os.path.dirname(path):
            break
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


_PROJECT_ROOT = _find_project_root()

# ── File-based logging per analyst / horizon ─────────────────────────────────
# Log files: results/{ticker}/{date}/{analyst_type}/{horizon}/analyst.log
# Cache of open loggers keyed by (analyst_type, ticker, date, horizon).
_loggers: Dict[str, logging.Logger] = {}


def _get_file_logger(
    analyst_type: str, ticker: str, trade_date: str, horizon: str
) -> logging.Logger:
    """Return (or create) a per-process file logger for the given analyst/horizon combo."""
    pid = os.getpid()
    key = f"{analyst_type}|{ticker}|{trade_date}|{horizon}|{pid}"
    if key in _loggers:
        return _loggers[key]

    log_dir = os.path.join(
        _PROJECT_ROOT, "results", ticker, trade_date, analyst_type, horizon
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"analyst_{pid}.log")

    logger = logging.getLogger(f"analyst.{key}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(fh)

    _loggers[key] = logger
    return logger


def _log(
    tag: str,
    text: str,
    *,
    verbose: bool = True,
    truncate: int = 0,
    trade_date: str = "",
) -> None:
    """Log to a per-analyst/horizon file and optionally print to stdout.

    The tag format is ``analyst_type|ticker|horizon`` (or shorter).
    The logger is resolved from the tag parts; ``trade_date`` sets the
    date subdirectory (falls back to today if omitted).
    """
    ts = datetime.now().strftime("%H:%M:%S")
    body = (
        text
        if truncate <= 0
        else text[:truncate] + ("…" if len(text) > truncate else "")
    )

    # Parse tag → route to the right file logger
    parts = tag.split("|")
    if len(parts) >= 3:
        analyst_type, ticker, horizon = parts[0], parts[1], parts[2]
        date_key = trade_date or datetime.now().strftime("%Y-%m-%d")
        logger = _get_file_logger(analyst_type, ticker, date_key, horizon)
        logger.info("[%s] %s", tag, body)

    # Also print to stdout if verbose
    # if not verbose:
    #     return
    # first, *rest = body.splitlines()
    # if rest:
    #     pad = " " * (len(ts) + len(tag) + 5)
    #     body = first + "\n" + textwrap.indent("\n".join(rest), pad)
    # print(f"[{ts}] [{tag}] {body}", flush=True)


# ── Constants ────────────────────────────────────────────────────────────────

HORIZON_LOOKBACK = {
    "long_term": 365 * 4,
    "medium_term": 90 * 4,
    "short_term": 30 * 4,
}

MAX_PASSES = {
    "shallow": 1,
    "medium": 2,
    "deep": 3,
}

# Horizon-specific focus instructions appended to Phase 1 / Phase 3 prompts
HORIZON_FOCUS = {
    "long_term": (
        "HORIZON: Long-term (1+ years). "
        "Focus on structural valuation, secular trends, durable competitive "
        "advantages, long-range growth models, and multi-year DCF projections. "
        "Weight fundamental / macro factors most heavily."
    ),
    "medium_term": (
        "HORIZON: Medium-term (3-12 months). "
        "Focus on earnings catalysts, sector rotation, cyclical indicators, "
        "mean-reversion setups, and intermediate trend signals. "
        "Balance fundamental and technical factors."
    ),
    "short_term": (
        "HORIZON: Short-term (days to weeks). "
        "Focus on momentum, order-flow signals, volatility regimes, event-driven "
        "catalysts, and sentiment extremes. "
        "Weight technical and sentiment factors most heavily."
    ),
}

# ── State sub-types ──────────────────────────────────────────────────────────

# Scalar value produced by code: float is the common case; nested structures
# (e.g. a dict of per-period returns) are represented as str (JSON-encoded) so
# the state stays fully serializable without Any.
MetricValue = Union[float, int, str, None]
TraceInput = Dict[str, Union[float, int, str, bool, None]]


class GatheredEntry(TypedDict):
    uuid: str
    data: str  # raw CSV string returned by the gather function
    headers: str  # first (header) row of the CSV


class MetricPlanEntry(TypedDict):
    metric_name: str
    term: str  # "long_term" | "medium_term" | "short_term"
    metric_interpretation: str
    metric_rationale: str
    computation_instruction: str


class RawComputedMetricEntry(TypedDict):
    metric_name: str
    value: MetricValue
    metric_interpretation: str
    computation_trace_id: str


class RawTraceEntry(TypedDict):
    trace_id: str
    code: str
    inputs: TraceInput
    output: MetricValue


class CodeAgentResult(TypedDict):
    code_succeeded: bool
    mu: float
    sigma: float
    mu_trace_id: str
    sigma_trace_id: str
    computed_metrics: List[RawComputedMetricEntry]
    computation_traces: List[RawTraceEntry]


class ValueInterpretationEntry(TypedDict):
    metric_name: str
    computation_trace_id: str
    interpretation: str


class CatalystEntry(TypedDict):
    catalyst: str
    metric_name: str
    computation_trace_id: str


class RiskEntry(TypedDict):
    risk: str
    metric_name: str
    computation_trace_id: str


class CitationEntry(TypedDict):  # for external source / news citations
    claim: str
    source: str
    source_uuid: str
    function_name: str
    function_parameters: TraceInput


class ThesisResult(TypedDict):
    investment_thesis: str
    conviction: float
    key_catalysts: List[CatalystEntry]
    key_risks: List[RiskEntry]
    value_interpretations: List[ValueInterpretationEntry]
    needs_more_research: bool
    additional_metrics_needed: str


class HorizonOutput(TypedDict):
    mu: float
    mu_trace_id: str
    sigma_contribution: float
    sigma_trace_id: str
    computed_metrics: List[ComputedMetric]
    investment_thesis: str
    conviction: float
    key_catalysts: List[CatalystEntry]
    key_risks: List[RiskEntry]
    source_uuids: List[str]
    computation_traces: List[ComputationTrace]
    contributing_factors: List[ContributingFactor]


# ── AnalystSubState ──────────────────────────────────────────────────────────


class AnalystSubState(TypedDict):
    ticker: str
    trade_date: str
    active_horizons: list[str]  # e.g. ["long_term", "medium_term", "short_term"]
    research_depth: str  # "shallow" | "medium" | "deep"
    gathered_data: Dict[str, GatheredEntry]
    all_computed_metrics: List[ComputedMetric]  # accumulated across all passes
    all_computation_traces: List[ComputationTrace]  # accumulated across all passes
    all_metrics_selected: List[MetricPlanEntry]  # accumulated across all passes
    compute_result: CodeAgentResult  # this pass's code agent output
    code_succeeded: bool
    plan_revision_count: int  # max 2 plan revisions per pass
    research_pass_count: int  # how many compute→thesis passes done
    thesis: ThesisResult
    thesis_by_horizon: Dict[str, ThesisResult]  # per-horizon thesis results
    needs_more_research: bool  # Phase 3 decision: loop back?
    additional_instructions: str  # passed from Phase 3 back into Phase 2
    final_outputs: Dict[str, HorizonOutput]


# ── Helpers ──────────────────────────────────────────────────────────────────


# ── Thesis JSON validation ──────────────────────────────────────────────────
# Top-level keys and the required sub-keys for their list entries.
_THESIS_REQUIRED_KEYS = frozenset(
    {
        "value_interpretations",
        "investment_thesis",
        "conviction",
        "key_catalysts",
        "key_risks",
        "needs_more_research",
        "additional_metrics_needed",
    }
)

_VALUE_INTERP_KEYS = frozenset(
    {"metric_name", "computation_trace_id", "interpretation"}
)
_CATALYST_KEYS = frozenset({"catalyst", "metric_name", "computation_trace_id"})
_RISK_KEYS = frozenset({"risk", "metric_name", "computation_trace_id"})


def _is_thesis_json(d: dict) -> bool:
    """True if *d* is a fully-formed thesis object with valid nested structure."""
    if not isinstance(d, dict) or not _THESIS_REQUIRED_KEYS.issubset(d):
        return False

    # value_interpretations: list of dicts with required sub-keys
    vi = d.get("value_interpretations")
    if not isinstance(vi, list):
        return False
    if vi and not all(
        isinstance(e, dict) and _VALUE_INTERP_KEYS.issubset(e) for e in vi
    ):
        return False

    # key_catalysts: list of dicts with required sub-keys (may be empty)
    cats = d.get("key_catalysts")
    if not isinstance(cats, list):
        return False
    if cats and not all(
        isinstance(e, dict) and _CATALYST_KEYS.issubset(e) for e in cats
    ):
        return False

    # key_risks: list of dicts with required sub-keys (may be empty)
    risks = d.get("key_risks")
    if not isinstance(risks, list):
        return False
    if risks and not all(isinstance(e, dict) and _RISK_KEYS.issubset(e) for e in risks):
        return False

    return True


def _parse_json_from_text(text: str) -> dict:
    """Extract the thesis JSON object from LLM output text.

    Strategies (in order):
      1. Direct ``json.loads`` on stripped text (clean output).
      2. ``raw_decode`` at position 0 — handles trailing text after JSON.
      3. Fenced ``json`` / bare ``` code blocks.
      4. Forward scan for every ``{`` — pick the candidate that passes
         ``_is_thesis_json``; break ties by key count (largest wins).
         If no candidate passes validation, fall back to largest dict.

    Returns an empty dict on total failure.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()

    # 1. Direct parse — works when the entire output is one JSON object.
    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
            if isinstance(d, dict) and _is_thesis_json(d):
                return d
        except json.JSONDecodeError:
            pass

        # 2. raw_decode at position 0 — parses the first complete JSON object
        #    and ignores any trailing text the LLM appended.
        try:
            d, _ = decoder.raw_decode(stripped, 0)
            if isinstance(d, dict) and _is_thesis_json(d):
                return d
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Fenced code blocks — ```json ... ``` or bare ``` ... ```
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            d = json.loads(m.group(1).strip())
            if isinstance(d, dict) and _is_thesis_json(d):
                return d
        except json.JSONDecodeError:
            continue

    # 4. Forward scan — prefer thesis-shaped objects; among those pick the
    #    one with the most keys (the full thesis, not a fragment).
    best_thesis: dict | None = None
    best_thesis_size = -1
    best_any: dict | None = None
    best_any_size = -1
    for m in re.finditer(r"\{", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        n = len(d)
        if _is_thesis_json(d) and n > best_thesis_size:
            best_thesis = d
            best_thesis_size = n
        if n > best_any_size:
            best_any = d
            best_any_size = n

    return best_thesis or best_any or {}


def _parse_plan_json(text: str) -> dict:
    """Extract the computation-plan JSON from Phase 1 LLM output.

    Unlike ``_parse_json_from_text`` (which is tuned for thesis objects and
    picks the dict with the *most keys*), this parser specifically looks for
    a dict containing a ``"computation_plan"`` key.  This avoids the bug where
    inner metric objects (4 keys each) are preferred over the outer plan
    dict (1 key: "computation_plan").

    Strategies (in order):
      1. Direct ``json.loads`` on stripped text.
      2. ``raw_decode`` at position 0 (handles trailing text).
      3. Fenced ``json`` / bare ``` code blocks.
      4. Forward scan for every ``{``, keeping the dict that contains
         ``"computation_plan"`` (or the largest dict if none has it).

    Returns an empty dict on total failure.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()

    # 1. Direct parse
    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
            if isinstance(d, dict) and "computation_plan" in d:
                return d
        except json.JSONDecodeError:
            pass

        # 2. raw_decode at position 0
        try:
            d, _ = decoder.raw_decode(stripped, 0)
            if isinstance(d, dict) and "computation_plan" in d:
                return d
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Fenced code blocks
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            d = json.loads(m.group(1).strip())
            if isinstance(d, dict) and "computation_plan" in d:
                return d
        except json.JSONDecodeError:
            continue

    # 4. Forward scan — prefer dicts with "computation_plan", else largest
    best_plan: dict | None = None
    best_plan_size = -1
    best_any: dict | None = None
    best_any_size = -1
    for m in re.finditer(r"\{", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        n = len(d)
        if "computation_plan" in d and n > best_plan_size:
            best_plan = d
            best_plan_size = n
        if n > best_any_size:
            best_any = d
            best_any_size = n

    return best_plan or best_any or {}


def _parse_thesis_json(text: str) -> dict:
    """Extract the thesis JSON from Phase 3 LLM output.

    Unlike ``_parse_json_from_text`` (which picks the dict with the most
    keys, causing it to select inner ``value_interpretations`` items with
    3 keys over an incomplete outer thesis with 2 keys), this parser
    specifically looks for a dict containing ``"value_interpretations"``.

    Strategies (in order):
      1. Direct ``json.loads`` on stripped text.
      2. ``raw_decode`` at position 0 (handles trailing text).
      3. Fenced ``json`` / bare ``` code blocks.
      4. Forward scan for every ``{``, keeping the dict that contains
         ``"value_interpretations"`` (or the largest dict if none has it).

    Returns an empty dict on total failure.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()

    # 1. Direct parse
    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
            if isinstance(d, dict) and "value_interpretations" in d:
                return d
        except json.JSONDecodeError:
            pass

        # 2. raw_decode at position 0
        try:
            d, _ = decoder.raw_decode(stripped, 0)
            if isinstance(d, dict) and "value_interpretations" in d:
                return d
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Fenced code blocks
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            d = json.loads(m.group(1).strip())
            if isinstance(d, dict) and "value_interpretations" in d:
                return d
        except json.JSONDecodeError:
            continue

    # 4. Forward scan — prefer dicts with "value_interpretations", else largest
    best_thesis: dict | None = None
    best_thesis_size = -1
    best_any: dict | None = None
    best_any_size = -1
    for m in re.finditer(r"\{", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        n = len(d)
        if "value_interpretations" in d and n > best_thesis_size:
            best_thesis = d
            best_thesis_size = n
        if n > best_any_size:
            best_any = d
            best_any_size = n

    return best_thesis or best_any or {}


def _is_multi_horizon_thesis_json(d: dict) -> bool:
    """True if *d* is a multi-horizon thesis object with per-horizon entries.

    Validation is lenient: each horizon only needs an "investment_thesis" key.
    Missing keys (conviction, catalysts, risks, etc.) are filled with defaults
    downstream. This avoids rejecting valid multi-horizon output because one
    horizon omitted a field.
    """
    if not isinstance(d, dict) or "horizons" not in d:
        return False
    horizons = d.get("horizons")
    if not isinstance(horizons, dict):
        return False
    for h_data in horizons.values():
        if not isinstance(h_data, dict):
            return False
        # Minimum requirement: must have at least an investment_thesis
        if "investment_thesis" not in h_data:
            return False
        # Validate nested list types if present
        for list_key in ("value_interpretations", "key_catalysts", "key_risks"):
            val = h_data.get(list_key)
            if val is not None and not isinstance(val, list):
                return False
    return True


def _parse_multi_horizon_thesis_json(text: str) -> dict:
    """Extract a multi-horizon thesis JSON from LLM output.

    Returns a dict with keys:
      horizons: dict[str, thesis_dict]
      needs_more_research: bool
      additional_metrics_needed: str
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()

    # 1. Direct parse
    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
            if _is_multi_horizon_thesis_json(d):
                return d
        except json.JSONDecodeError:
            pass
        try:
            d, _ = decoder.raw_decode(stripped, 0)
            if _is_multi_horizon_thesis_json(d):
                return d
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Fenced code blocks
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            d = json.loads(m.group(1).strip())
            if _is_multi_horizon_thesis_json(d):
                return d
        except json.JSONDecodeError:
            continue

    # 3. Forward scan — prefer multi-horizon thesis
    best_mh: dict | None = None
    best_mh_size = -1
    best_any: dict | None = None
    best_any_size = -1
    for m in re.finditer(r"\{", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        n = len(d)
        if _is_multi_horizon_thesis_json(d) and n > best_mh_size:
            best_mh = d
            best_mh_size = n
        if n > best_any_size:
            best_any = d
            best_any_size = n

    return best_mh or best_any or {}


def _coerce_value_interpretations(raw: object) -> List[ValueInterpretationEntry]:
    if not isinstance(raw, list):
        return []
    result: List[ValueInterpretationEntry] = []
    for item in raw:
        if isinstance(item, dict) and all(
            k in item for k in ("metric_name", "computation_trace_id", "interpretation")
        ):
            result.append(
                ValueInterpretationEntry(
                    metric_name=str(item["metric_name"]),
                    computation_trace_id=str(item["computation_trace_id"]),
                    interpretation=str(item["interpretation"]),
                )
            )
    return result


def _coerce_catalysts(raw: object) -> List[CatalystEntry]:
    if not isinstance(raw, list):
        return []
    result: List[CatalystEntry] = []
    for item in raw:
        if isinstance(item, dict) and all(
            k in item for k in ("catalyst", "metric_name", "computation_trace_id")
        ):
            result.append(
                CatalystEntry(
                    catalyst=str(item["catalyst"]),
                    metric_name=str(item["metric_name"]),
                    computation_trace_id=str(item["computation_trace_id"]),
                )
            )
    return result


def _coerce_risks(raw: object) -> List[RiskEntry]:
    if not isinstance(raw, list):
        return []
    result: List[RiskEntry] = []
    for item in raw:
        if isinstance(item, dict) and all(
            k in item for k in ("risk", "metric_name", "computation_trace_id")
        ):
            result.append(
                RiskEntry(
                    risk=str(item["risk"]),
                    metric_name=str(item["metric_name"]),
                    computation_trace_id=str(item["computation_trace_id"]),
                )
            )
    return result


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_date_range(trade_date: str, lookback_days: int) -> tuple[str, str]:
    """Return (start_date, end_date) strings given a trade date and lookback.

    ``lookback_days`` represents the desired number of **trading days** of data.
    To ensure we get at least that many data points, we extend the calendar-day
    range to account for weekends (≈5 trading days per 7 calendar days) and a
    small buffer for market holidays.
    """
    end = datetime.strptime(trade_date, "%Y-%m-%d")
    # 1 trading day ≈ 1.5 calendar days (7/5 ratio) + 5-day holiday buffer
    calendar_days = int(lookback_days * 1.5) + 5
    start = end - timedelta(days=calendar_days)
    return start.strftime("%Y-%m-%d"), trade_date


# ── Subgraph factory ────────────────────────────────────────────────────────


def build_3phase_subgraph(
    reasoning_llm, code_agent, analyst_config: dict, verbose: bool = True
):
    """Build and compile a LangGraph StateGraph for one analyst-horizon run.

    Parameters
    ----------
    reasoning_llm : LangChain chat model (e.g. ChatOpenAI)
    code_agent    : CodeValidationAgent instance
    analyst_config : dict with keys:
        agent_type           : AgentType enum value
        phase2_system_prompt : str (domain prompt for the code agent)
        phase3_system_prompt : str
        gather_fn            : callable(ticker, trade_date, lookback) -> dict

    Returns
    -------
    Compiled LangGraph StateGraph
    """
    agent_type: AgentType = analyst_config["agent_type"]
    phase2_prompt: str | None = analyst_config.get("phase2_system_prompt")
    phase3_prompt: str = analyst_config["phase3_system_prompt"]
    gather_fn = analyst_config["gather_fn"]
    horizon_focus: dict = analyst_config.get("horizon_focus", HORIZON_FOCUS)

    # ── Node: data_gathering ─────────────────────────────────────────────

    def data_gathering(sub_state: AnalystSubState) -> dict:
        ticker = sub_state["ticker"]
        trade_date = sub_state["trade_date"]
        active_horizons = sub_state["active_horizons"]
        # Use the maximum lookback across all active horizons so the code agent
        # has enough data to compute every horizon in a single pass.
        max_lookback = max(HORIZON_LOOKBACK[h] for h in active_horizons)
        # Extend lookback by 300 trading days so the k-NN search space has
        # enough historical points with valid forward returns (t+h exists).
        extended_lookback = max_lookback + 300

        _log(
            f"{agent_type.value}|{ticker}|multi",
            f"data_gathering: horizons={active_horizons} max_lookback={max_lookback}d extended={extended_lookback}d",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        raw_data = gather_fn(ticker, trade_date, extended_lookback)

        # Assign source UUIDs for provenance tracking
        sourced_data: Dict[str, GatheredEntry] = {}
        for key, val in raw_data.items():
            uid = str(uuid.uuid4())
            headers = (
                val.splitlines()[0] if isinstance(val, str) and val.strip() else ""
            )
            sourced_data[key] = GatheredEntry(uuid=uid, data=val, headers=headers)

        _log(
            f"{agent_type.value}|{ticker}|multi",
            f"data_gathering: collected {len(sourced_data)} data keys: {list(sourced_data.keys())}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{ticker}|multi",
            f"data_gathering: full gathered data:\n{json.dumps(sourced_data, indent=2)}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        return {"gathered_data": sourced_data}

    # ── Node: phase2_compute ─────────────────────────────────────────────

    def phase2_compute(sub_state: AnalystSubState) -> dict:
        sourced_data = sub_state["gathered_data"]
        revision_count = sub_state["plan_revision_count"]
        pass_count = sub_state["research_pass_count"] + 1
        additional_instructions = sub_state.get("additional_instructions", "")
        active_horizons = sub_state["active_horizons"]

        # Flatten sourced_data for code agent: strip UUIDs/headers, keep raw CSV
        flat_data: Dict[str, str] = {
            key: entry["data"] for key, entry in sourced_data.items()
        }

        # Build the planning payload for the code agent.
        # The code agent now analyses data, plans metrics, and writes code.
        plan_payload = {
            "ticker": sub_state["ticker"],
            "analysis_date": sub_state["trade_date"],
            "active_horizons": active_horizons,
            "analyst_type": agent_type.value,
            "research_pass": pass_count,
            "revision_count": revision_count,
            "additional_instructions": additional_instructions,
        }

        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase2_compute: executing code agent  pass={pass_count} revision={revision_count}  data_keys={list(flat_data.keys())}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase2_compute INPUT — plan payload:\n{json.dumps(plan_payload, indent=2)}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase2_compute INPUT — domain prompt:\n{phase2_prompt or '(default)'}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        try:
            result = code_agent.execute_plan(plan_payload, flat_data, phase2_prompt)
        except Exception as e:
            _log(
                f"{agent_type.value}|{sub_state['ticker']}|multi",
                f"phase2_compute: code agent exception: {e}",
                verbose=verbose,
            )
            result = {
                "code_succeeded": False,
                "horizons": {},
                "computed_metrics": [],
                "computation_traces": [],
                "metrics_selected": [],
            }

        succeeded = result.get("code_succeeded", False)
        horizon_summary = ""
        if "horizons" in result and result["horizons"]:
            parts = []
            for h, hv in result["horizons"].items():
                parts.append(f"{h}: mu={hv.get('mu', 0.0):.4f} sigma={hv.get('sigma', 0.0):.4f}")
            horizon_summary = "  " + "; ".join(parts)
        else:
            horizon_summary = f"mu={result.get('mu', 0.0):.4f} sigma={result.get('sigma', 0.0):.4f}"

        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase2_compute: code agent finished  succeeded={succeeded}  {horizon_summary}  metrics_computed={len(result.get('computed_metrics', []))}  traces={len(result.get('computation_traces', []))}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase2_compute OUTPUT:\n{json.dumps(result, indent=2, default=str)}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        all_metrics: List[ComputedMetric] = list(sub_state["all_computed_metrics"])
        all_traces: List[ComputationTrace] = list(sub_state["all_computation_traces"])
        all_selected: List[MetricPlanEntry] = list(sub_state["all_metrics_selected"])

        # Support both multi-horizon output (new) and single-horizon fallback
        raw_horizons = result.get("horizons", {})
        if raw_horizons and isinstance(raw_horizons, dict):
            typed_result = CodeAgentResult(
                code_succeeded=bool(result.get("code_succeeded", False)),
                mu=0.0,
                sigma=0.01,
                mu_trace_id="",
                sigma_trace_id="",
                computed_metrics=result.get("computed_metrics", []),
                computation_traces=result.get("computation_traces", []),
            )
            # Attach horizons dict for downstream nodes
            typed_result["horizons"] = raw_horizons
        else:
            typed_result = CodeAgentResult(
                code_succeeded=bool(result.get("code_succeeded", False)),
                mu=float(result.get("mu", 0.0)),
                sigma=float(result.get("sigma", 0.01)),
                mu_trace_id=str(result.get("mu_trace_id", "")),
                sigma_trace_id=str(result.get("sigma_trace_id", "")),
                computed_metrics=result.get("computed_metrics", []),
                computation_traces=result.get("computation_traces", []),
            )

        # ── Safety net: ensure mu and sigma are present in computed_metrics ──
        # The code agent sometimes omits them; we backfill from horizons/top-level.
        _existing_metric_names = {
            m.get("metric_name", "") for m in typed_result["computed_metrics"] if isinstance(m, dict)
        }
        if raw_horizons and isinstance(raw_horizons, dict):
            for h, hv in raw_horizons.items():
                if not isinstance(hv, dict):
                    continue
                for metric_name, trace_key in [("mu", "mu_trace_id"), ("sigma", "sigma_trace_id")]:
                    if metric_name not in _existing_metric_names:
                        typed_result["computed_metrics"].append({
                            "metric_name": metric_name,
                            "value": hv.get(metric_name),
                            "computation_trace_id": hv.get(trace_key, ""),
                            "term": h,
                        })
        else:
            for metric_name, trace_key in [("mu", "mu_trace_id"), ("sigma", "sigma_trace_id")]:
                if metric_name not in _existing_metric_names:
                    typed_result["computed_metrics"].append({
                        "metric_name": metric_name,
                        "value": typed_result.get(metric_name),
                        "computation_trace_id": typed_result.get(trace_key, ""),
                    })

        if succeeded:
            # Post-computation sanity check: sigma at floor (0.01) for any
            # horizon almost certainly means the code agent failed.
            sigmas_to_check: list[float] = []
            if raw_horizons and isinstance(raw_horizons, dict):
                for hv in raw_horizons.values():
                    if isinstance(hv, dict):
                        sigmas_to_check.append(float(hv.get("sigma", 0.01)))
            else:
                sigmas_to_check.append(float(result.get("sigma", 0.01)))

            any_at_floor = any(s <= 0.0101 for s in sigmas_to_check)
            if any_at_floor:
                _log(
                    f"{agent_type.value}|{sub_state['ticker']}|multi",
                    f"phase2_compute: SIGMA AT FLOOR in at least one horizon — "
                    f"sigmas={sigmas_to_check}.  Real stocks have sigma >> 0.01; "
                    f"treating as code failure and triggering revision.",
                    verbose=verbose,
                    trade_date=sub_state.get("trade_date", "")
                    if isinstance(sub_state, dict)
                    else "",
                )
                return {
                    "compute_result": typed_result,
                    "code_succeeded": False,
                    "plan_revision_count": sub_state["plan_revision_count"] + 1,
                }

            # Build metric_interpretation lookup from the code agent's plan output
            selected_metrics = result.get("metrics_selected", [])
            interpretation_lookup: Dict[str, str] = {
                m.get("metric_name", ""): m.get("metric_interpretation", "")
                for m in selected_metrics
                if isinstance(m, dict)
            }
            # Accumulate metrics_selected from code agent output
            for m in selected_metrics:
                if isinstance(m, dict):
                    try:
                        all_selected.append(
                            MetricPlanEntry(
                                metric_name=m.get("metric_name", ""),
                                term=m.get("term", "long_term"),
                                metric_interpretation=m.get("metric_interpretation", ""),
                                metric_rationale=m.get("metric_rationale", ""),
                                computation_instruction=m.get("computation_instruction", ""),
                            )
                        )
                    except Exception:
                        pass

            # Build typed ComputedMetric instances.
            # If the code agent output includes per-horizon computed_metrics,
            # we use those; otherwise we tag everything with the (legacy) horizon.
            per_horizon_metrics: Dict[str, list[dict]] = {}
            if raw_horizons and isinstance(raw_horizons, dict):
                for h, hv in raw_horizons.items():
                    if isinstance(hv, dict):
                        per_horizon_metrics[h] = hv.get("computed_metrics", [])

            has_per_horizon_metrics = any(metrics for metrics in per_horizon_metrics.values())
            if has_per_horizon_metrics:
                for h, metrics in per_horizon_metrics.items():
                    for m in metrics:
                        trace_id = m.get("computation_trace_id") or str(uuid.uuid4())
                        all_metrics.append(
                            ComputedMetric(
                                metric_name=m["metric_name"],
                                term=h,
                                value=m["value"],
                                metric_interpretation=interpretation_lookup.get(
                                    m["metric_name"], m.get("metric_interpretation", "")
                                ),
                                value_interpretation="",
                                computation_trace_id=trace_id,
                            )
                        )
            else:
                for m in typed_result["computed_metrics"]:
                    trace_id = m.get("computation_trace_id") or str(uuid.uuid4())
                    all_metrics.append(
                        ComputedMetric(
                            metric_name=m["metric_name"],
                            term=m.get("term", "long_term"),
                            value=m["value"],
                            metric_interpretation=interpretation_lookup.get(
                                m["metric_name"], m.get("metric_interpretation", "")
                            ),
                            value_interpretation="",
                            computation_trace_id=trace_id,
                        )
                    )

            for t in typed_result["computation_traces"]:
                all_traces.append(
                    ComputationTrace(
                        trace_id=t.get("trace_id") or str(uuid.uuid4()),
                        code=t.get("code", ""),
                        inputs=t.get("inputs", {}),
                        output=t.get("output"),
                    )
                )
            return {
                "compute_result": typed_result,
                "code_succeeded": True,
                "all_computed_metrics": all_metrics,
                "all_computation_traces": all_traces,
                "all_metrics_selected": all_selected,
                "research_pass_count": pass_count,
            }
        else:
            return {
                "compute_result": typed_result,
                "code_succeeded": False,
                "plan_revision_count": sub_state["plan_revision_count"] + 1,
            }

    # ── Node: phase3_thesis ──────────────────────────────────────────────

    def phase3_thesis(sub_state: AnalystSubState) -> dict:
        active_horizons = sub_state["active_horizons"]
        gathered = sub_state["gathered_data"]
        all_metrics = sub_state["all_computed_metrics"]
        pass_count = sub_state["research_pass_count"]
        depth = sub_state["research_depth"]
        max_passes = MAX_PASSES.get(depth, 2)
        code_succeeded = sub_state["code_succeeded"]

        horizon_focus_text = "\n\n".join(
            f"### {h}\n{horizon_focus.get(h, '')}" for h in active_horizons
        )

        system_content = (
            f"{phase3_prompt}\n\n{horizon_focus_text}\n\n"
            "You are given a list of computed metrics, each with a computation_trace_id that links "
            "to the exact code that produced it. Your job is to interpret the numbers and form a "
            "fully grounded view — no claims without a metric.\n\n"
            "You MUST return a single JSON object with EXACTLY the structure below.\n"
            "Do NOT omit any key. Do NOT add keys. Do NOT wrap in markdown fences.\n"
            "Do NOT output any text before or after the JSON.\n\n"
            "OUTPUT THE SCALAR AND LIST FIELDS FIRST, then the long-text fields.\n"
            "This ensures critical fields are never truncated.\n\n"
            "Required output structure (follow this key order):\n"
            "{\n"
            '  "horizons": {\n'
            '    "long_term": {\n'
            '      "conviction": 0.0,  // Placeholder — computed deterministically.\n'
            '      "key_catalysts": [\n'
            '        { "catalyst": "<description>", "metric_name": "<name>", "computation_trace_id": "<id>" }\n'
            "      ],\n"
            '      "key_risks": [\n'
            '        { "risk": "<description>", "metric_name": "<name>", "computation_trace_id": "<id>" }\n'
            "      ],\n"
            '      "investment_thesis": "<string built ONLY from the computed metrics. Cite each supporting metric inline as [metric_name | trace:<computation_trace_id>]. Keep CONCISE — under 300 words.>",\n'
            '      "value_interpretations": [\n'
            "        // One entry per computed metric for THIS horizon\n"
            '        { "metric_name": "<name>", "computation_trace_id": "<id>", "interpretation": "<1-2 sentences>" }\n'
            "      ]\n"
            "    },\n"
            '    "medium_term": { ... },\n'
            '    "short_term": { ... }\n'
            "  },\n"
            '  "needs_more_research": <bool — true ONLY if critical metrics are missing or code failed>,\n'
            '  "additional_metrics_needed": "<string — what to compute next (only relevant if looping)>"\n'
            "}\n\n"
            "CRITICAL: Your entire response must be the JSON object above and nothing else.\n"
            "CRITICAL: Output horizons (with conviction, key_catalysts, key_risks) and\n"
            "needs_more_research BEFORE investment_thesis and value_interpretations.\n"
            "These fields are required and must not be truncated.\n\n"
            "NOTE: Only include the horizons that are present in the input. If a horizon "
            "has no computed metrics, you may omit it or provide empty arrays."
        )

        user_parts = [
            f"Ticker: {sub_state['ticker']}",
            f"Trade Date: {sub_state['trade_date']}",
            f"Active Horizons: {', '.join(active_horizons)}",
            f"Research Pass: {pass_count} / Max {max_passes}",
            f"Code Succeeded: {code_succeeded}",
        ]

        # Show headers + first 3 data rows per key for context window management
        preview_parts = []
        for key, entry in gathered.items():
            raw = entry["data"]
            if isinstance(raw, str):
                h = entry["headers"]
                lines = raw.splitlines()
                data_rows = lines[1:2] if len(lines) > 1 else []
                snippet = "\n".join([h] + data_rows) if h else "\n".join(data_rows)
            else:
                # Dict / list / other non-string payloads — pretty-print JSON
                text = (
                    json.dumps(raw, indent=2, default=str)
                    if isinstance(raw, (dict, list))
                    else str(raw)
                )
                lines = text.splitlines()
                snippet = "\n".join(lines[:2])
            preview_parts.append(f"[{key}]\n{snippet}")
        user_parts.append("\n=== Gathered Data ===\n" + "\n\n".join(preview_parts))

        # Group computed metrics by horizon for the prompt
        metrics_by_horizon = {h: [] for h in active_horizons}
        for m in all_metrics:
            h = m.term
            if h in metrics_by_horizon:
                metrics_by_horizon[h].append(m.model_dump())
        for h in active_horizons:
            user_parts.append(
                f"\n=== Computed Metrics for {h} ===\n{json.dumps(metrics_by_horizon[h], indent=2)}"
            )

        # Provide fundamentals context to Phase 3 so the thesis can reference
        # the fundamental profile and calibration rules that shaped the metrics.
        fundamentals_csv_p3 = get_fundamentals(sub_state["ticker"])
        if isinstance(fundamentals_csv_p3, str) and not fundamentals_csv_p3.startswith(
            "Error"
        ):
            user_parts.append(
                f"\n=== Company Fundamentals (context for interpretation) ===\n{fundamentals_csv_p3}"
            )

        sys_msg = SystemMessage(content=system_content)
        user_msg = HumanMessage(content="\n".join(user_parts))

        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase3_thesis: invoking LLM  pass={pass_count}/{max_passes}  system_chars={len(system_content)}  user_chars={len(user_msg.content)}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase3_thesis LLM INPUT — system:\n{system_content}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase3_thesis LLM INPUT — user:\n{user_msg.content}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        result = reasoning_llm.invoke([sys_msg, user_msg])

        # Future Improvements. Deterministically:
        # 1. Should check if all the value interpretation is there and the trace matches.
        # 2. Should check if all the citation match the metric name and trace.
        # 3. Should check if all key_catalysts key_risk match the given metric and trace.

        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase3_thesis: LLM responded  output_chars={len(result.content)}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )
        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase3_thesis LLM OUTPUT (raw):\n{result.content}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        # Try multi-horizon parse first, fall back to single-horizon
        raw_thesis = _parse_multi_horizon_thesis_json(result.content)
        if raw_thesis and "horizons" in raw_thesis:
            _log(
                f"{agent_type.value}|{sub_state['ticker']}|multi",
                f"phase3_thesis: parsed MULTI-HORIZON thesis JSON:\n{json.dumps(raw_thesis, indent=2)}",
                verbose=verbose,
                trade_date=sub_state.get("trade_date", "")
                if isinstance(sub_state, dict)
                else "",
            )
        else:
            # Fallback: parse as single-horizon thesis and replicate across all active horizons
            single_thesis = _parse_thesis_json(result.content)
            raw_thesis = {
                "horizons": {},
                "needs_more_research": single_thesis.get("needs_more_research", False),
                "additional_metrics_needed": single_thesis.get("additional_metrics_needed", ""),
            }
            for h in active_horizons:
                # Make a shallow copy and prepend horizon context so the thesis
                # is not identically replicated across all horizons.
                h_copy = dict(single_thesis)
                thesis_text = h_copy.get("investment_thesis", "")
                if thesis_text:
                    h_copy["investment_thesis"] = (
                        f"[{h.upper().replace('_', ' ')}] {thesis_text}"
                    )
                raw_thesis["horizons"][h] = h_copy
            _log(
                f"{agent_type.value}|{sub_state['ticker']}|multi",
                f"phase3_thesis: parsed SINGLE-HORIZON thesis JSON (replicated):\n{json.dumps(raw_thesis, indent=2)}",
                verbose=verbose,
                trade_date=sub_state.get("trade_date", "")
                if isinstance(sub_state, dict)
                else "",
            )

        # Determine whether to loop back for more research
        wants_more = raw_thesis.get("needs_more_research", False)
        can_loop = pass_count < max_passes
        will_loop = wants_more and can_loop

        # Build per-horizon ThesisResult objects
        thesis_by_horizon = {}
        for h in active_horizons:
            h_data = raw_thesis.get("horizons", {}).get(h, {})
            typed = ThesisResult(
                investment_thesis=str(h_data.get("investment_thesis", "")),
                conviction=float(h_data.get("conviction", 0.0)),
                key_catalysts=_coerce_catalysts(h_data.get("key_catalysts")),
                key_risks=_coerce_risks(h_data.get("key_risks")),
                value_interpretations=_coerce_value_interpretations(
                    h_data.get("value_interpretations")
                ),
                needs_more_research=bool(h_data.get("needs_more_research", False)),
                additional_metrics_needed=str(h_data.get("additional_metrics_needed", "")),
            )
            thesis_by_horizon[h] = typed

        # Use the first active horizon's thesis as the legacy "thesis" field
        first_horizon = active_horizons[0] if active_horizons else "long_term"
        first_thesis = thesis_by_horizon.get(first_horizon, ThesisResult(
            investment_thesis="",
            conviction=0.0,
            key_catalysts=[],
            key_risks=[],
            value_interpretations=[],
            needs_more_research=False,
            additional_metrics_needed="",
        ))

        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase3_thesis: pass={pass_count} wants_more={wants_more} can_loop={can_loop} will_loop={will_loop}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        return {
            "thesis": first_thesis,
            "thesis_by_horizon": thesis_by_horizon,
            "needs_more_research": will_loop,
            "plan_revision_count": 0,
            "code_succeeded": True,
            "additional_instructions": str(raw_thesis.get("additional_metrics_needed", "")),
        }

    # ── Node: phase4_output ──────────────────────────────────────────────

    def phase4_output(sub_state: AnalystSubState) -> dict:
        thesis_by_horizon = sub_state.get("thesis_by_horizon", {})
        compute_result = sub_state["compute_result"]
        sourced_data = sub_state["gathered_data"]
        active_horizons = sub_state["active_horizons"]

        source_uuids: List[str] = [entry["uuid"] for entry in sourced_data.values()]

        # Support both multi-horizon and single-horizon compute_result formats
        raw_horizons = compute_result.get("horizons", {})

        final_outputs: Dict[str, HorizonOutput] = {}

        for horizon in active_horizons:
            h_thesis = thesis_by_horizon.get(horizon, ThesisResult(
                investment_thesis="",
                conviction=0.0,
                key_catalysts=[],
                key_risks=[],
                value_interpretations=[],
                needs_more_research=False,
                additional_metrics_needed="",
            ))

            if raw_horizons and isinstance(raw_horizons, dict) and horizon in raw_horizons:
                h_result = raw_horizons[horizon]
                mu = _safe_float(h_result.get("mu", 0.0))
                sigma = max(_safe_float(h_result.get("sigma", 0.01), 0.01), 1e-6)
                mu_trace_id = str(h_result.get("mu_trace_id", ""))
                sigma_trace_id = str(h_result.get("sigma_trace_id", ""))
            else:
                mu = _safe_float(compute_result.get("mu", 0.0))
                sigma = max(_safe_float(compute_result.get("sigma", 0.01), 0.01), 1e-6)
                mu_trace_id = str(compute_result.get("mu_trace_id", ""))
                sigma_trace_id = str(compute_result.get("sigma_trace_id", ""))

            if sigma <= 0.0101:
                _log(
                    f"{agent_type.value}|{sub_state['ticker']}|multi",
                    f"phase4_output: WARNING sigma={sigma:.4f} at floor for {horizon} — "
                    f"code agent likely failed to compute volatility",
                    verbose=verbose,
                    trade_date=sub_state.get("trade_date", "")
                    if isinstance(sub_state, dict)
                    else "",
                )

            # Patch value_interpretation from Phase 3 onto each ComputedMetric for this horizon
            value_interpretation_by_trace: Dict[str, str] = {
                v["computation_trace_id"]: v["interpretation"]
                for v in h_thesis["value_interpretations"]
            }
            horizon_metrics: List[ComputedMetric] = [
                ComputedMetric(
                    metric_name=m.metric_name,
                    term=horizon,
                    value=m.value,
                    metric_interpretation=m.metric_interpretation,
                    value_interpretation=value_interpretation_by_trace.get(
                        m.computation_trace_id, ""
                    ),
                    computation_trace_id=m.computation_trace_id,
                )
                for m in sub_state["all_computed_metrics"]
                if m.term == horizon
            ]

            # ── Deterministic conviction computation (per-horizon) ───────────────────────
            import math

            n_computed = len(horizon_metrics)
            data_completeness = min(n_computed / max(len(horizon_metrics) if horizon_metrics else 1, 1), 1.0)

            signal_concordance: Optional[float] = None
            for m in horizon_metrics:
                if m.metric_name == "signal_concordance" and m.value is not None:
                    try:
                        signal_concordance = float(m.value)
                        signal_concordance = max(0.0, min(1.0, signal_concordance))
                    except (TypeError, ValueError):
                        signal_concordance = None
                    break

            signal_dispersion: Optional[float] = None
            for m in horizon_metrics:
                if m.metric_name == "signal_dispersion" and m.value is not None:
                    try:
                        signal_dispersion = float(m.value)
                        signal_dispersion = max(0.0, signal_dispersion)
                    except (TypeError, ValueError):
                        signal_dispersion = None
                    break

            if sigma <= 0.0101 or n_computed < 2:
                deterministic_conviction = 0.0
            elif signal_concordance is not None:
                if signal_dispersion is not None:
                    dispersion_score = max(0.0, 1.0 - signal_dispersion / 0.05)
                else:
                    dispersion_score = 0.5

                regime_clarity = 0.5
                for m in horizon_metrics:
                    if m.metric_name == "kaufman_efficiency_ratio" and m.value is not None:
                        try:
                            er = abs(float(m.value))
                            if er < 0.20 or er > 0.40:
                                regime_clarity = 1.0
                            else:
                                regime_clarity = 0.5
                        except (TypeError, ValueError):
                            pass
                        break

                signal_significance = min(abs(mu) / 0.02, 1.0)
                logit = (
                    3.0 * (signal_concordance - 0.5)
                    + 1.5 * (dispersion_score - 0.5)
                    + 0.8 * (regime_clarity - 0.5)
                    + 0.8 * (signal_significance - 0.5)
                    + 0.4 * (data_completeness - 0.5)
                )
                raw_conviction = 1.0 / (1.0 + math.exp(-2.0 * logit))
                deterministic_conviction = round(max(0.0, min(raw_conviction, 1.0)), 2)
            else:
                snr_fallback = min(abs(mu) / max(sigma, 0.01), 1.0)
                raw_conviction = data_completeness * snr_fallback
                deterministic_conviction = round(min(raw_conviction, 1.0), 2)

            output = HorizonOutput(
                mu=mu,
                mu_trace_id=mu_trace_id,
                sigma_contribution=sigma,
                sigma_trace_id=sigma_trace_id,
                computed_metrics=horizon_metrics,
                investment_thesis=h_thesis["investment_thesis"],
                conviction=deterministic_conviction,
                key_catalysts=h_thesis["key_catalysts"],
                key_risks=h_thesis["key_risks"],
                source_uuids=source_uuids,
                computation_traces=list(sub_state["all_computation_traces"]),
                contributing_factors=[],
            )

            final_outputs[horizon] = output

            _log(
                f"{agent_type.value}|{sub_state['ticker']}|multi",
                f"phase4_output: {horizon} mu={output['mu']:.4f} sigma={output['sigma_contribution']:.4f} conviction={output['conviction']:.2f}  metrics={len(output['computed_metrics'])}  catalysts={len(output['key_catalysts'])}  risks={len(output['key_risks'])}",
                verbose=verbose,
                trade_date=sub_state.get("trade_date", "")
                if isinstance(sub_state, dict)
                else "",
            )

        _log(
            f"{agent_type.value}|{sub_state['ticker']}|multi",
            f"phase4_output: full outputs for {list(final_outputs.keys())}",
            verbose=verbose,
            trade_date=sub_state.get("trade_date", "")
            if isinstance(sub_state, dict)
            else "",
        )

        return {"final_outputs": final_outputs}
    # ── Routing functions ────────────────────────────────────────────────

    def route_after_compute(sub_state: AnalystSubState) -> str:
        if sub_state["code_succeeded"]:
            return "phase3_thesis"
        # Allow up to 2 code revisions: count goes 1 -> 2 -> (give up at 3)
        if sub_state["plan_revision_count"] <= 2:
            return "phase2_compute"
        return "phase3_thesis"

    def route_after_thesis(sub_state: AnalystSubState) -> str:
        if sub_state["needs_more_research"]:
            return "phase2_compute"
        return "phase4_output"

    # ── Build graph ──────────────────────────────────────────────────────

    graph = StateGraph(AnalystSubState)

    graph.add_node("data_gathering", data_gathering)
    graph.add_node("phase2_compute", phase2_compute)
    graph.add_node("phase3_thesis", phase3_thesis)
    graph.add_node("phase4_output", phase4_output)

    graph.add_edge(START, "data_gathering")
    graph.add_edge("data_gathering", "phase2_compute")

    graph.add_conditional_edges(
        "phase2_compute",
        route_after_compute,
        ["phase2_compute", "phase3_thesis"],
    )
    graph.add_conditional_edges(
        "phase3_thesis",
        route_after_thesis,
        ["phase2_compute", "phase4_output"],
    )

    graph.add_edge("phase4_output", END)

    return graph.compile()


# ── Horizon merge ────────────────────────────────────────────────────────────


def merge_horizon_results(
    horizon_results: Dict[str, HorizonOutput],
    all_metrics_selected: List[MetricPlanEntry],
    agent_type: AgentType,
    ticker: str,
    trade_date: str,
) -> ResearchReport:
    """Deterministically merge three horizon outputs into one ResearchReport."""
    parsed: Dict[str, HorizonOutput] = {
        h: horizon_results.get(
            h,
            HorizonOutput(  # type: ignore[call-arg]
                mu=0.0,
                mu_trace_id="",
                sigma_contribution=0.01,
                sigma_trace_id="",
                computed_metrics=[],
                investment_thesis="",
                conviction=0.0,
                key_catalysts=[],
                key_risks=[],
                source_uuids=[],
                computation_traces=[],
                contributing_factors=[],
            ),
        )
        for h in ("long_term", "medium_term", "short_term")
    }

    def _horizon_value(key: str, default: float = 0.0) -> HorizonValues:
        return HorizonValues(
            long_term=_safe_float(parsed["long_term"][key], default),  # type: ignore[literal-required]
            medium_term=_safe_float(parsed["medium_term"][key], default),  # type: ignore[literal-required]
            short_term=_safe_float(parsed["short_term"][key], default),  # type: ignore[literal-required]
        )

    def _horizon_trace_ids(key: str) -> HorizonTraceIds:
        return HorizonTraceIds(
            long_term=str(parsed["long_term"][key]),  # type: ignore[literal-required]
            medium_term=str(parsed["medium_term"][key]),  # type: ignore[literal-required]
            short_term=str(parsed["short_term"][key]),  # type: ignore[literal-required]
        )

    all_computed_metrics: List[ComputedMetric] = []
    all_traces: List[ComputationTrace] = []
    all_factors: List[ContributingFactor] = []
    all_source_uuids: List[str] = []
    all_catalysts: List[Catalyst] = []
    all_risks: List[Risk] = []

    for h in ("long_term", "medium_term", "short_term"):
        p = parsed[h]

        all_source_uuids.extend(p["source_uuids"])

        for m in p["computed_metrics"]:
            all_computed_metrics.append(
                ComputedMetric(
                    metric_name=f"{h}:{m.metric_name}",
                    value=m.value,
                    term=h,
                    metric_interpretation=m.metric_interpretation,
                    value_interpretation=m.value_interpretation,
                    computation_trace_id=m.computation_trace_id,
                )
            )

        all_traces.extend(p["computation_traces"])

        for c in p["key_catalysts"]:
            all_catalysts.append(
                Catalyst(
                    catalyst=c["catalyst"],
                    term=h,
                    metric_name=c["metric_name"],
                    computation_trace_id=c["computation_trace_id"],
                )
            )
        for r in p["key_risks"]:
            all_risks.append(
                Risk(
                    risk=r["risk"],
                    term=h,
                    metric_name=r["metric_name"],
                    computation_trace_id=r["computation_trace_id"],
                )
            )

    metrics_selected = [
        Metrics(
            metric_name=m["metric_name"],
            term=m["term"],
            metric_interpretation=m["metric_interpretation"],
            metric_rationale=m["metric_rationale"],
            computation_instruction=m["computation_instruction"],
        )
        for m in all_metrics_selected
    ]

    return ResearchReport(
        ticker=ticker,
        agent_type=agent_type,
        timestamp=datetime.strptime(trade_date, "%Y-%m-%d"),
        metrics_selected=metrics_selected,
        mu=_horizon_value("mu"),
        mu_trace_id=_horizon_trace_ids("mu_trace_id"),
        sigma_contribution=_horizon_value("sigma_contribution", 0.01),
        sigma_trace_id=_horizon_trace_ids("sigma_trace_id"),
        computed_metrics=all_computed_metrics,
        investment_thesis=HorizonThesis(
            long_term=parsed["long_term"]["investment_thesis"],
            medium_term=parsed["medium_term"]["investment_thesis"],
            short_term=parsed["short_term"]["investment_thesis"],
        ),
        conviction=_horizon_value("conviction"),
        key_catalysts=all_catalysts,
        key_risks=all_risks,
        source_uuids=all_source_uuids,
        computation_traces=all_traces,
        citation_chain=[],
        contributing_factors=all_factors,
    )


# ── Analyst node factory ────────────────────────────────────────────────────


def create_analyst_node(
    reasoning_llm,
    code_agent,
    analyst_config: dict,
    research_depth: str = "medium",
    verbose: bool = True,
):
    """Create an analyst node function for the outer AgentState graph.

    The returned function:
      1. Runs the 3-phase subgraph ONCE for all active horizons together
      2. Merges the per-horizon outputs into a single ResearchReport
      3. Returns {state_key: report_json} for the outer graph state

    Parameters
    ----------
    reasoning_llm  : LangChain chat model
    code_agent     : CodeValidationAgent instance
    analyst_config : dict with keys:
        agent_type           : AgentType
        state_key            : str — key in AgentState to write the report to
        gather_fn            : callable(ticker, trade_date, lookback_days) -> dict
        phase2_system_prompt : str
        phase3_system_prompt : str
    research_depth : "shallow" | "medium" | "deep"
    """
    agent_type: AgentType = analyst_config["agent_type"]
    state_key: str = analyst_config["state_key"]
    log_tag: str = analyst_config.get("log_tag", agent_type.value)

    sub_graph = build_3phase_subgraph(
        reasoning_llm, code_agent, analyst_config, verbose=verbose
    )

    def analyst_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]

        active_horizons = list(
            analyst_config.get(
                "active_horizons", ("long_term", "medium_term", "short_term")
            )
        )

        _log(
            log_tag,
            f"Starting analysis for {ticker} on {trade_date} (depth={research_depth}, horizons={active_horizons})",
            verbose=verbose,
            trade_date=trade_date,
        )

        sub_state: AnalystSubState = {
            "ticker": ticker,
            "trade_date": trade_date,
            "active_horizons": active_horizons,
            "research_depth": research_depth,
            "gathered_data": {},
            "all_computed_metrics": [],
            "all_computation_traces": [],
            "all_metrics_selected": [],
            "compute_result": CodeAgentResult(
                code_succeeded=False,
                mu=0.0,
                sigma=0.01,
                mu_trace_id="",
                sigma_trace_id="",
                computed_metrics=[],
                computation_traces=[],
            ),
            "code_succeeded": False,
            "plan_revision_count": 0,
            "research_pass_count": 0,
            "thesis": ThesisResult(
                investment_thesis="",
                conviction=0.0,
                key_catalysts=[],
                key_risks=[],
                value_interpretations=[],
                needs_more_research=False,
                additional_metrics_needed="",
            ),
            "thesis_by_horizon": {},
            "needs_more_research": False,
            "additional_instructions": "",
            "final_outputs": {},
        }

        result = sub_graph.invoke(sub_state)

        _log(
            f"{log_tag}|{ticker}|_",
            f"Completed subgraph for horizons: {active_horizons}",
            verbose=verbose,
            trade_date=trade_date,
        )

        horizon_results: Dict[str, HorizonOutput] = result.get("final_outputs", {})
        all_metrics_selected: List[MetricPlanEntry] = result.get("all_metrics_selected", [])

        report = merge_horizon_results(
            horizon_results,
            all_metrics_selected,
            agent_type,
            ticker,
            trade_date,
        )

        # ── Log the final report to results/{ticker}/{date}/report_{uuid}.log ───
        report_log_dir = os.path.join(_PROJECT_ROOT, "results", ticker, trade_date)
        os.makedirs(report_log_dir, exist_ok=True)
        _run_uuid = uuid.uuid4().hex[:8]
        report_log_path = os.path.join(report_log_dir, f"report_{_run_uuid}.log")
        with open(report_log_path, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{ts}] {agent_type.value} analyst report for {ticker}\n")
            f.write(f"{'=' * 60}\n")
            f.write(report.model_dump_json(indent=2))
            f.write("\n")

        _log(
            f"{log_tag}|{ticker}|_",
            f"Final report logged to {report_log_path}",
            verbose=verbose,
            trade_date=trade_date,
        )

        return {state_key: report.model_dump_json()}

    return analyst_node
