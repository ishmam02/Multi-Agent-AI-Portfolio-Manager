# Layer 2: Deep Research Analyst Agents — Implementation Plan

## Context

Layer 1 (StockScreener) produces an "Analyzed Universe" of tickers. The current research layer has 4 simple analyst agents (Market, Social Media, News, Fundamentals) that call tools and produce free-form markdown reports. This plan redesigns that layer into a sophisticated, institutional-grade research pipeline where each analyst follows a 3-phase workflow (LLM Reasoning → Code Validation → Structured Output) with multi-horizon analysis.

---

## Architecture

Each of the 4 new analysts becomes a **LangGraph sub-graph** that runs the 3-phase workflow for each of 3 time horizons, then composites results. The parent graph preserves the existing fan-out → barrier → trader pattern.

```
Parent Graph:
  START ──┬── Fundamental Analyst (sub-graph) ──┐
          ├── Technical Analyst (sub-graph)   ──┤
          ├── Macro Analyst (sub-graph)       ──┼── Analyst Sync ── Synthesis Agent ── END
          └── Sentiment Analyst (sub-graph)   ──┘

Each Analyst Sub-Graph (runs 3x, once per horizon):

  data_gathering → phase1_plan ──→ phase2_compute ──→ phase3_thesis ──→ phase4_output
                       ↑                ↑       |            |
                       |                |       │            │  Phase 3 checks research depth:
                       |                |       │            │  - shallow: proceed to phase4
                       |                |       │            │  - medium/deep: may loop back
                       |                |       │            │    to phase1 for MORE metrics
                       |                |       │            │
                       |                |       │            └──→ needs_more_research? ──→ phase1
                       |                └───────┘                 (accumulated results preserved)
                       |                  code retry loop
                       └──────────────────────────────────────────┘
                         research deepening loop

  Phase 1 (Plan):     LLM reads data + any prior results, decides WHAT to compute next
  Phase 2 (Compute):  Code agent executes computations, results ACCUMULATE (never discarded)
  Phase 3 (Thesis):   LLM interprets ALL results so far. Based on research_depth config:
                        - "shallow": 1 pass only, proceed to phase4
                        - "medium":  up to 2 passes (1 deepening loop)
                        - "deep":    up to 3 passes (2 deepening loops)
                       LLM decides if more research is needed based on gaps in evidence.
                       If looping back: specifies ADDITIONAL metrics/models to compute.
                       All prior Phase 2 code + computations are preserved and accumulated.
  Phase 4 (Output):   Assemble final ResearchReport with full provenance from ALL passes

  GUARANTEED OUTPUT: If all retries and revisions exhausted, produce a degraded
  ResearchReport with plan-only qualitative output (mu/sigma set to neutral
  defaults, conviction lowered, computed_metrics empty, provenance notes the failure).
  The system NEVER blocks or requires human intervention.
```

---

## Step 1: Pydantic Schemas

**Create** `src/agents/analysts/v2/schemas.py`

Common output schema used by **all 4 analyst agents**:

```python
class AgentType(str, Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACRO = "macro"
    SENTIMENT = "sentiment"

class ComputedMetric(BaseModel):
    metric_name: str              # e.g. "dcf_fair_value", "rsi", "hy_spread"
    value: Any                    # float, dict, or nested structure (from Phase 2)
    metric_interpretation: str    # What this metric IS and what it measures (filled in Phase 3)
    value_interpretation: str     # What this specific VALUE means for this stock right now (filled in Phase 3)
    computation_trace_id: str     # links to the code that produced it

class ComputationTrace(BaseModel):
    trace_id: str
    code: str                     # The Python code that was executed
    inputs: dict                  # Data fed into the code
    output: Any                   # Raw code output

class Citation(BaseModel):
    claim: str
    source: str
    source_uuid: str
    function_name: str            # Tool function that produced the source data (e.g., "get_balance_sheet")
    function_parameters: dict     # Parameters it was called with (e.g., {"ticker": "AAPL", "freq": "quarterly"})

class ContributingFactor(BaseModel):
    factor_uuid: str
    factor_name: str
    weight_in_analysis: float
    current_ic: float             # Information coefficient
    decay_status: str

class HorizonValues(BaseModel):
    long_term: float
    medium_term: float
    short_term: float

class HorizonInterval(BaseModel):
    long_term: Tuple[float, float]
    medium_term: Tuple[float, float]
    short_term: Tuple[float, float]

class ResearchReport(BaseModel):
    """Common output schema for all analyst agents."""
    # ── Identity ──
    ticker: str
    agent_type: AgentType
    timestamp: datetime

    # ── Phase 1: Computation Plan (LLM decides WHAT to compute) ──
    metrics_selected: List[str]   # LLM declares what it will compute
    metrics_rationale: str        # Why these metrics for this stock right now
    computation_instructions: str # Detailed instructions for each metric

    # ── Phase 2: Symbolic (Code-Computed Numerics) ──
    mu: HorizonValues             # Expected return per horizon
    sigma_contribution: HorizonValues  # Volatility contribution per horizon
    confidence_interval: HorizonInterval
    computed_metrics: List[ComputedMetric]  # Flexible list, not fixed dict

    # ── Phase 3: Investment Thesis (LLM interprets Phase 2 results) ──
    investment_thesis: str        # Formed AFTER seeing the numbers
    conviction: float             # [-1.0, +1.0] based on quantitative evidence
    key_catalysts: List[str]
    key_risks: List[str]

    # ── Provenance ──
    source_uuids: List[str]
    computation_traces: List[ComputationTrace]
    citation_chain: List[Citation]  # Each claim linked to a computed metric or data source
    contributing_factors: List[ContributingFactor]
```

**Key design**: `computed_metrics` is a flexible list, not a fixed dictionary. The LLM declares in `metrics_selected` what it intends to compute and why, then each metric lands in `computed_metrics` with value, interpretation, and trace ID linking to exact code. A Fundamental report for AAPL might have 8 metrics while the same agent's report for a pre-revenue biotech might have 3. The thesis drives metric selection, not a template.

**Provenance fields**: `source_uuids` are generated by data tools (each tool call produces a UUID for its data source). `contributing_factors.current_ic` and `decay_status` are **placeholders** for now (ic=0.0, decay="unknown") — these fields exist for a future backtesting layer to populate with actual information coefficients.

---

## Step 2: Code Validation Agent (Claude Agent SDK)

**Create** `src/agents/analysts/v2/code_agent.py`

Uses the **Claude Agent SDK** to create an autonomous coding sub-agent configured with an Ollama model. This agent handles the full Phase 2 loop: generate code → execute → check errors → fix → iterate.

```python
class CodeValidationAgent:
    """Phase 2 sub-agent using Claude Agent SDK with Ollama model."""

    def __init__(self, model: str, base_url: str, timeout: int, max_iterations: int):
        # Configure Claude Agent SDK client with Ollama endpoint
        # Agent has: code execution tool (sandboxed), data access tool

    def execute_plan(self, computation_plan: dict, data: dict) -> dict:
        # 1. Agent receives computation plan JSON (metrics_selected, computation_instructions)
        #    + raw data JSON
        # 2. Agent generates Python code for each metric/model using analyst-specific libraries
        # 3. Agent executes code via SDK's built-in execution
        # 4. If error: agent autonomously debugs and retries (up to max_iterations)
        # 5. Returns dict with: mu (float), sigma (float), computed_metrics (list of
        #    {metric_name, value}), computation_traces (list of {trace_id, code, inputs,
        #    output}), code_succeeded (bool)
```

Key design:
- The SDK handles sandboxed code execution internally
- Agent prompt specifies: allowed libraries (analyst-specific), expected JSON output format, data schema
- **Analyst-specific libraries**: each analyst type gets a tailored set of allowed packages:
  - Fundamental: numpy, pandas, scipy, statistics (DCF, ratios, scoring models)
  - Technical: numpy, pandas, **TA-Lib** (talib), scipy, statistics (indicators, vol surface)
  - Macro: numpy, pandas, scipy, statsmodels (regime models, rate sensitivity)
  - Sentiment: numpy, pandas, scipy, statistics (correlation, statistical tests)
  - All analysts also get: math, json, datetime
- Data is injected as a context variable (JSON-serialized financials/prices/indicators)
- Agent returns dict with mu, sigma, computed_metrics list, computation_traces list, code_succeeded bool
- **The metrics are not hardcoded** — the Phase 1 plan LLM decides which metrics to compute based on its reasoning. The listed metrics (DCF, Piotroski, etc.) are examples; the plan LLM can propose additional industry-verified metrics as appropriate.
- **Agent-specific system prompts**: Each of the 4 CodeValidationAgent instances gets a tailored system prompt defining its domain expertise, expected metric types, and validation standards. For example, the Fundamental code agent's prompt emphasizes DCF methodology, accounting standards, and valuation rigor, while the Technical code agent's prompt emphasizes indicator computation correctness and statistical signal processing.
- Config: `code_agent_model` (Ollama model name), `code_agent_base_url` (Ollama endpoint), `code_agent_max_iterations`

---

## Step 3: New Data Tools

**Create** `src/agents/utils/macro_data_tools.py`:
- `get_fred_data(series_id, start_date, end_date)` — GDP, CPI, employment, rates via fredapi
- `get_treasury_yields(curr_date, look_back_days)` — yield curve via yfinance Treasury ETFs
- `get_sector_performance(curr_date, look_back_days)` — sector ETF returns for rotation analysis

**Create** `src/agents/utils/options_data_tools.py`:
- `get_options_chain(ticker, curr_date)` — options data + IV via yfinance
- `get_short_interest(ticker, curr_date)` — short ratio/interest from ticker.info

**Create** `src/agents/utils/sec_filing_tools.py`:
- `get_sec_filings(ticker, filing_type, limit)` — SEC EDGAR full-text search API (free, User-Agent header only)

**Modify** `src/dataflows/interface.py` — register new tool categories: `macro_data`, `options_data`, `sec_filings`
**Modify** `src/agents/utils/agent_utils.py` — export new tools

---

## Step 4: Base Analyst Sub-Graph

**Create** `src/agents/analysts/base_analyst.py`

Factory function `build_3phase_subgraph(reasoning_llm, code_agent, analyst_config)` that builds a LangGraph `StateGraph` with internal state:

```python
class AnalystSubState(TypedDict):
    ticker: str
    trade_date: str
    horizon: str                    # "long_term" | "medium_term" | "short_term"
    research_depth: str             # "shallow" | "medium" | "deep" (from user config)
    gathered_data: str              # Raw data from tools (JSON)
    computation_plan: str           # Phase 1: JSON — current pass's metrics to compute
    all_computed_metrics: str       # ACCUMULATED across all passes (JSON list)
    all_computation_traces: str     # ACCUMULATED across all passes (JSON list)
    all_metrics_selected: str       # ACCUMULATED list of all metrics planned across passes
    compute_result: str             # Phase 2: JSON — this pass's results
    code_succeeded: bool
    plan_revision_count: int        # Max 2 plan revisions per pass (for code failures)
    research_pass_count: int        # How many plan→compute→thesis passes done so far
    thesis: str                     # Phase 3: JSON — investment_thesis, conviction, catalysts, risks, citations
    needs_more_research: bool       # Phase 3 decision: loop back for more metrics?
    final_output: str               # Phase 4: JSON subset of ResearchReport for this horizon
```

**Phase 1 — Plan (LLM decides WHAT to compute):**
- `data_gathering`: calls analyst-specific tools deterministically with horizon-appropriate lookback windows (long: 365d, medium: 90d, short: 10d). Runs once (not repeated on deepening loops).
- `phase1_plan`: reasoning LLM reads gathered data **+ any accumulated results from prior passes** and produces a **computation plan**. On the first pass, this is the initial analysis plan. On subsequent passes (deepening loops), the LLM sees what was already computed and plans ADDITIONAL metrics to fill gaps. Output includes:
  - `metrics_selected`: list of NEW metrics/models to compute this pass
  - `metrics_rationale`: why these specific metrics are needed (on deepening: "Phase 2 showed X, need to investigate Y further")
  - `computation_instructions`: detailed instructions for each metric
  - The LLM does NOT form a thesis yet — it only plans the quantitative work.

**Phase 2 — Compute (Code agent executes the plan):**
- `phase2_compute`: invokes `CodeValidationAgent.execute_plan()` — the Claude Agent SDK sub-agent receives this pass's computation plan + raw data + any prior results for context. Generates code, executes, debugs, iterates.
  - **Code succeeds** → new `computed_metrics` and `computation_traces` are **appended** to the accumulated lists (never replacing prior results). Proceed to phase3.
  - **Code fails after max iterations** → reasoning LLM revises this pass's plan to simplify, re-run phase2 (up to 2 plan revisions per pass)
  - **All revisions exhausted** → skip this pass's metrics, proceed to phase3 with what we have. **System never blocks.**

**Phase 3 — Thesis (LLM interprets ALL accumulated results):**
- `phase3_thesis`: reasoning LLM receives the **original data + ALL accumulated computed metrics from ALL passes** and forms/updates the investment thesis. The LLM:
  - Interprets each computed metric (what does a Piotroski F-Score of 8 mean for this company?)
  - Forms a conviction-based investment thesis grounded in the numbers
  - Identifies key catalysts and risks, citing specific computed metrics and data sources
  - Assigns conviction score [-1.0, +1.0] based on the quantitative evidence
  - Produces `citation_chain` linking each claim to its source data and computation trace
  - **Research depth decision**: Based on `research_depth` config:
    - `"shallow"`: always proceeds to phase4 (1 pass only)
    - `"medium"`: LLM may request 1 additional pass if it identifies significant evidence gaps (max 2 passes total)
    - `"deep"`: LLM may request up to 2 additional passes for thorough multi-model analysis (max 3 passes total)
  - If looping back: sets `needs_more_research=True` and specifies what additional metrics/models are needed and why. All prior Phase 2 results are preserved.

**Phase 4 — Output (Assemble ResearchReport):**
- `phase4_output`: deterministic assembly of the full horizon output — combines ALL computation plans from all passes, ALL accumulated computed metrics with LLM interpretations, the final thesis, citations, and provenance into the horizon's portion of the `ResearchReport`

### 4b: Horizon Merge (Outer Analyst Node)

Each analyst node is a Python function that runs the sub-graph 3 times (once per horizon), then **merges** the results into a single `ResearchReport`:

```python
def analyst_node(state: AgentState):
    horizon_results = {}
    for horizon in ["long_term", "medium_term", "short_term"]:
        sub_state = {"ticker": ..., "trade_date": ..., "horizon": horizon, ...}
        result = sub_graph.invoke(sub_state)
        horizon_results[horizon] = result["final_output"]  # JSON per horizon

    # Merge: mu.long_term from long_term run, mu.medium_term from medium_term run, etc.
    report = merge_horizon_results(horizon_results, analyst_type, ticker, trade_date)
    return {"fundamental_report": report.model_dump_json(), ...}
```

The merge is deterministic (no LLM call) — it assembles `HorizonValues` from the 3 runs and concatenates `computed_metrics`, `computation_traces`, and `citation_chain` from all horizons.

### 4c: Phase 1 (Plan) and Phase 3 (Thesis) System Prompts

Each analyst has **two distinct system prompts** for the reasoning LLM:

**Phase 1 — Computation Plan prompts** (decide WHAT to compute):
- **Fundamental**: "You are a senior equity research analyst. Review the financial data, SEC filings, and earnings transcripts. Determine which valuation models, ratio analyses, and scoring frameworks should be computed to properly evaluate this company. Do NOT form a thesis yet — only plan the quantitative work."
- **Technical**: "You are a senior technical analyst. Review the price/volume data and market microstructure. Determine which indicators, statistical measures, and regime models should be computed. Do NOT form a thesis yet — only plan the quantitative work."
- **Macro**: "You are a senior macroeconomist. Review the GDP, employment, inflation, and central bank data. Determine which regime classification models, sensitivity analyses, and sector rotation signals should be computed. Do NOT form a thesis yet — only plan the quantitative work."
- **Sentiment**: "You are a senior sentiment analyst. Review the news, social media, and insider transaction data. Determine which sentiment scoring methods and cross-validation metrics should be computed against quantitative proxies. Do NOT form a thesis yet — only plan the quantitative work."

**Phase 3 — Thesis Formation prompts** (interpret results, form thesis):
- **Fundamental**: "You are a senior equity research analyst. You have received computed valuation metrics, financial ratios, and scoring results. Interpret these numbers in context. Form an investment thesis on intrinsic value vs. market price. Cite specific computed metrics to support each claim."
- **Technical**: "You are a senior technical analyst. You have received computed indicators, volatility measures, and regime classifications. Interpret these signals. Form a thesis on price direction and momentum. Cite specific metrics and their computation traces."
- **Macro**: "You are a senior macroeconomist. You have received regime classification outputs, sensitivity analyses, and sector signals. Interpret the macro environment's implications for this stock. Cite specific model outputs."
- **Sentiment**: "You are a senior sentiment analyst. You have received sentiment scores cross-validated against volume, IV, and short interest. Interpret behavioral signals. Form a thesis on market sentiment positioning. Cite specific cross-validation results."

Each prompt also includes: horizon-specific instructions (what to focus on for long/medium/short term), the output schema, and data availability.

### 4d: Data Gathering Implementation

The `data_gathering` node calls the existing `@tool`-decorated functions **directly as Python functions** (bypassing LLM tool-calling). This is deterministic — no LLM needed to decide what data to fetch:

```python
def data_gathering(sub_state: AnalystSubState) -> AnalystSubState:
    ticker = sub_state["ticker"]
    lookback = HORIZON_LOOKBACK[sub_state["horizon"]]
    # Call tools directly (they return strings)
    data = {
        "fundamentals": get_fundamentals.invoke({"ticker": ticker}),
        "balance_sheet": get_balance_sheet.invoke({"ticker": ticker, "freq": "quarterly"}),
        ...
    }
    return {"gathered_data": json.dumps(data)}
```

Each tool call generates a UUID stored in the data dict for `source_uuids` provenance tracking.

---

## Step 5: Four Analyst Implementations

Each calls `build_3phase_subgraph` with analyst-specific config (prompts, tools, expected metrics).

Each analyst outputs a full **`ResearchReport`** (the common schema from Step 1).

### Multi-Horizon Caching

Each horizon has different freshness requirements following institutional practice:
- **Short-Term (Daily)**: Computed **fresh** on every run.

The analyst node checks the cache before running each horizon's sub-graph. If a valid cached result exists (within its freshness window), the sub-graph is skipped and the cached `HorizonAnalysis` is used directly. This reduces LLM calls significantly for repeated analysis of the same ticker.

**Create** `src/agents/analysts/fundamental_analyst.py`:
- Tools: get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_sec_filings
- Example metrics (not limited to): multi-stage DCF, residual income, EV/EBITDA, P/E, PEG, DuPont, Altman Z, Piotroski F — thesis LLM decides which to compute
- Analyst-specific libs: numpy, pandas, scipy, statistics
- Code agent system prompt: domain expertise in valuation methodology, accounting standards, financial modeling rigor
- Output: full `ResearchReport` with `agent_type=FUNDAMENTAL`

**Create** `src/agents/analysts/technical_analyst.py`:
- Tools: get_stock_data, get_indicators, get_options_chain, get_short_interest
- Example metrics (not limited to): RSI, MACD, Bollinger, ADX, Ichimoku, VWMA, realized/implied vol surface, options skew
- Analyst-specific libs: numpy, pandas, **TA-Lib (talib)**, scipy, statistics
- Code agent system prompt: domain expertise in technical indicator computation, statistical signal processing, regime detection
- Output: full `ResearchReport` with `agent_type=TECHNICAL`

**Create** `src/agents/analysts/macro_analyst.py`:
- Tools: get_news, get_global_news, get_fred_data, get_treasury_yields, get_sector_performance
- Example metrics (not limited to): regime classification, sector rotation signals, rate sensitivity, credit spread decomposition
- Analyst-specific libs: numpy, pandas, scipy, statsmodels
- Code agent system prompt: domain expertise in macroeconomic modeling, regime classification, rate sensitivity analysis
- Output: full `ResearchReport` with `agent_type=MACRO`

**Create** `src/agents/analysts/sentiment_analyst.py`:
- Tools: get_news, get_insider_transactions
- Example metrics (not limited to): cross-reference sentiment scores against abnormal volume, IV changes, short interest shifts
- Analyst-specific libs: numpy, pandas, scipy, statistics
- Code agent system prompt: domain expertise in sentiment quantification, behavioral signals, statistical validation of qualitative assessments
- Output: full `ResearchReport` with `agent_type=SENTIMENT`

---

## Step 6: Horizon Cache Layer

**Create** `src/agents/analysts/v2/horizon_cache.py`

Persistent cache for horizon analysis results:

```python
class HorizonCache:
    """Caches horizon analysis results with freshness rules."""

    def __init__(self, cache_dir: str):
        # File-based cache: {cache_dir}/{ticker}/{analyst_type}/{horizon}.json

    def get(self, ticker: str, analyst_type: str, horizon: str,
            trade_date: str) -> Optional[dict]:
        # Returns cached HorizonAnalysis JSON if still fresh:
        # - long_term: valid until next quarter or material event trigger
        # - medium_term: valid within same calendar week
        # - short_term: never cached (always returns None)

    def put(self, ticker: str, analyst_type: str, horizon: str,
            trade_date: str, result: dict):
        # Stores result with timestamp metadata

    def invalidate(self, ticker: str, analyst_type: str = None,
                   horizon: str = None, reason: str = "manual"):
        # Invalidates cache entries (e.g., on material event trigger)
```

Freshness rules:
- **Long-term**: cached result valid if no new 10-K/10-Q has been filed since cache timestamp (checked via SEC EDGAR filing date API) AND no material corporate event detected in news
- **Medium-term**: cached result valid if `trade_date` falls in the same ISO calendar week as cache timestamp
- **Short-term**: always recomputed, never served from cache

The analyst node's outer loop becomes:
```python
for horizon in ["long_term", "medium_term", "short_term"]:
    cached = cache.get(ticker, analyst_type, horizon, trade_date)
    if cached:
        horizon_results[horizon] = cached
    else:
        result = sub_graph.invoke(sub_state)
        cache.put(ticker, analyst_type, horizon, trade_date, result)
        horizon_results[horizon] = result
```

## Step 8: LLM Synthesis Agent (Replaces Trader)

**Replace** `src/agents/trader/trader.py` with a new Synthesis Agent that performs dynamic signal aggregation instead of simple report consumption.

### 8a: New Schemas (add to `schemas.py`)

```python
class AnalystWeights(BaseModel):
    """Per-horizon weights for each analyst signal."""
    fundamental: float    # w_f
    market: float      # w_m
    news: float          # w_n
    justification: str    # Why these weights for this stock/horizon

class CrossSignalConflict(BaseModel):
    """Flagged contradiction between analyst reports."""
    analyst_a: str
    analyst_b: str
    description: str
    conviction_adjustment: float  # How much to reduce conviction

class CompositeSignal(BaseModel):
    """Final synthesized signal for one stock."""
    ticker: str
    per_horizon_mu: Dict[str, float]          # {"short_term": X, "medium_term": Y, "long_term": Z}
    per_horizon_weights: Dict[str, AnalystWeights]  # Weights per horizon
    horizon_blend_weights: Dict[str, float]    # How horizons are blended into final
    mu_final: float                            # Unified expected return
    sigma_final: float                         # From covariance computation
    conviction_final: float                    # 0.0-1.0 after conflict adjustments
    conflicts: List[CrossSignalConflict]
    weighting_rationale: str                   # Full reasoning chain (provenance)
```

### 8b: Synthesis Agent Implementation

**Rewrite** `src/agents/trader/trader.py` → `create_synthesis_agent(llm, code_agent, memory)`:

**Phase 1 — Dynamic Weight Assignment (LLM):**
- Deserialize all 3 `ResearchReport` objects from state
- LLM reads all reports and determines per-analyst, per-horizon weights (w_f, w_t, w_m, w_s) for this specific stock in the current market context based on the investment thesis and catalyst and risks
- LLM provides written justification for weighting rationale
- Weights must sum to 1.0 per horizon

**Phase 2 — Cross-Signal Consistency Check (LLM):**
- LLM identifies contradictions across analyst reports (e.g., fundamental bullish but technical breakdown)
- Each conflict is documented with a `CrossSignalConflict` object
- Conviction is adjusted downward for unresolved contradictions

**Phase 3 — Composite Signal Generation:**
- Fixed formulae computes the composite signal mathematically:
  ```
  μ_composite(h) = w_f(h)·μ_fund(h) + w_t(h)·μ_tech(h) + w_m(h)·μ_macro(h) + w_s(h)·μ_sent(h)
  ```
  for each horizon h
- Fixed formulae blends per-horizon composites into unified μ_final using LLM-determined horizon weights
- Fixed formulae computes covariance matrix Σ across the Analyzed Universe from individual analyst sigma outputs
- Code validates mathematical consistency (weights sum to 1, μ within bounds, Σ positive semi-definite)

**Output**: The Synthesis Agent outputs the `CompositeSignal` (mu_final, sigma_final, conviction_final, per-horizon weights, conflicts, weighting rationale). **Trade proposal generation (direction, entry, targets, stop-loss, etc.) is NOT done here** — it belongs to Layer 3 (Portfolio Management). The Synthesis Agent's job ends at producing the composite conviction signal with full provenance.


### 8d: State Changes

Add to `AgentState`:
- `composite_signal: str` — JSON-serialized `CompositeSignal` (Layer 2 final output)
- Remove `trader_investment_plan` (trade proposals are Layer 3's responsibility)

### 8e: Memory Integration

The Synthesis Agent uses `FinancialSituationMemory` (existing BM25-based memory at `src/agents/utils/memory.py`). The current situation is built from the structured reports (concatenating investment thesis and catalyst and risks from all 3 reports). Past similar situations are retrieved and included in the Synthesis Agent's context for Phases 1, helping avoid past mistakes.

### 8g: Multi-Ticker Covariance

When `propagate_multi()` runs multiple tickers, after all individual syntheses complete, a final deterministic step computes the **full cross-asset covariance matrix** from the individual analyst sigma outputs. This feeds into portfolio-level position sizing (Layer 3 concern, but the data is prepared here).

---

## Step 9: Integration — State and Graph Wiring

*(Horizon compositing is handled by the Synthesis Agent in Step 8, not a separate component.)*

**Modify** `src/agents/utils/agent_states.py`:
 fields (`market_report`, `news_report`, `fundamentals_report`)

**Modify** `src/graph/setup.py`:
- New analyst types: `["fundamental", "market", "news"]`
- New report_keys_map mapping to new state fields
- Simplified conditional edges (sub-graphs are self-contained, no parent-level tool loops)

**Modify** `src/graph/trading_graph.py`:
- Create **3 separate `CodeValidationAgent` instances**, each with an analyst-specific system prompt and allowed library set:
  - `fundamental_code_agent` (valuation, accounting, financial modeling)
  - `market_code_agent` (indicators, signal processing, regime detection)
  - `news_code_agent` (sentiment quantification, behavioral signals)
- All 3 use the same Ollama model but differ in system prompt and allowed libs
- Update `_create_tool_nodes()` with new tool sets for each analyst
- Pass reasoning LLM + analyst-specific code agent to each analyst factory
- Pass synthesis_code_agent to the Synthesis Agent factory
- Remove old analyst references

**Modify** `src/graph/conditional_logic.py`:
- Simplify: no more tool-loop routing, just check if report field is populated

**Modify** `src/graph/propagation.py`:
- Initialize new state fields to empty strings

**Modify** `src/graph/setup.py` and `src/graph/trading_graph.py`:

Each analyst node wrapper catches exceptions and retries with exponential backoff (max 2 retries). If still failing after retries, the analyst is **skipped** — its report field stays empty. The Synthesis Agent detects missing reports and:
- Adjusts weights to compensate (redistribute among available analysts)
- Notes the missing analyst in the provenance chain (`weighting_rationale`)
- Lowers overall conviction proportionally
- **System never blocks or requires human intervention**

---

## Step 10: Reflection and Signal Processing Updates

**Modify** `src/graph/reflection.py`:
- `_extract_current_situation()` now reads structured `ResearchReport` JSON fields instead of old free-form report fields
- Reflection prompt updated to reference composite signals, computed metrics, and thesis narratives
- `reflect_trader()` uses `composite_signal` + `trader_investment_plan` as the decision to reflect on

**Modify** `src/graph/signal_processing.py`:
- `process_signal()` now reads from `CompositeSignal.trade_proposal` (structured) instead of parsing free-text for BUY/SELL/HOLD
- The `CompositeSignal` already contains the direction, so extraction is simpler
- `propose_trade()` can use `mu_final`, `conviction_final`, and the trade proposal to generate more precise Alpaca order parameters

---

## Step 11: CLI Updates

**Modify** `cli/main.py`:
- Analyst selection updated: offer `["fundamental", "market", "news"]` instead of old types
- `MessageBuffer` progress tracking updated for new agent names and 3-phase workflow stages
- Streaming output shows per-horizon progress (e.g., "Fundamental Analyst: long_term thesis → code validation → ✓")
- Results display shows structured output: composite mu, conviction, key metrics, conflicts
- Config prompts include `code_agent_model` and `code_agent_base_url` options

---

## Step 12: Config and Dependencies

**Modify** `src/default_config.py`:
```python
"code_agent_model": "qwen2.5-coder:32b",       # Ollama model for code generation
"code_agent_base_url": "http://localhost:11434",  # Ollama endpoint
"code_agent_max_iterations": 5,                   # Max generate-execute-fix cycles
"research_depth": "medium",                        # "shallow" (1 pass), "medium" (up to 2), "deep" (up to 3)
"horizons_enabled": ["long_term", "medium_term", "short_term"],
"horizon_lookback": {"long_term": 365, "medium_term": 90, "short_term": 10},
```

**Modify** `pyproject.toml`: add `pydantic>=2.0`, `fredapi>=0.5.0`, `claude-agent-sdk`, `TA-Lib`, `statsmodels`

---

## Step 13: Cleanup

- Update `src/agents/__init__.py` to export v2 factories, remove old imports
- Update `_log_state` in trading_graph.py for new report fields
- Update CLI (`cli/main.py`) analyst selection to use new analyst types (`fundamental`, `technical`, `macro`, `sentiment`)
- Delete old analyst files: `market_analyst.py`, `social_media_analyst.py`, `news_analyst.py`, `fundamentals_analyst.py`

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sub-graphs vs flat nodes | Sub-graphs | Encapsulates 3-phase complexity; parent graph stays simple |
| State fields as JSON strings vs nested dicts | JSON strings | LangGraph state works best with primitives; Pydantic handles ser/deser |
| Phase 2 code engine | Claude Agent SDK + Ollama | Autonomous generate-execute-debug loop; SDK handles sandboxed execution |
| Horizon parallelism | Sequential (3 passes) | Simpler debugging, predictable rate limits; can upgrade to `Send()` later |
| Migration strategy | Full replacement | Clean codebase; old analysts removed entirely |

---

## Files to Create (12)
1. `src/agents/analysts/v2/__init__.py`
2. `src/agents/analysts/v2/schemas.py`
3. `src/agents/analysts/v2/base_analyst.py`
4. `src/agents/analysts/v2/code_agent.py` — Claude Agent SDK coding sub-agent
5. `src/agents/analysts/v2/horizon_cache.py` — persistent horizon caching with freshness rules
6. `src/agents/analysts/v2/fundamental_analyst.py`
7. `src/agents/analysts/v2/technical_analyst.py`
8. `src/agents/analysts/v2/macro_analyst.py`
9. `src/agents/analysts/v2/sentiment_analyst.py`
10. `src/agents/utils/macro_data_tools.py`
11. `src/agents/utils/options_data_tools.py`
12. `src/agents/utils/sec_filing_tools.py`

## Files to Delete (4)
1. `src/agents/analysts/market_analyst.py`
2. `src/agents/analysts/social_media_analyst.py`
3. `src/agents/analysts/news_analyst.py`
4. `src/agents/analysts/fundamentals_analyst.py`

## Files to Rewrite (1)
1. `src/agents/trader/trader.py` — full rewrite as LLM Synthesis Agent

## Files to Modify (13)
1. `src/agents/utils/agent_states.py` — new report fields, remove legacy
2. `src/agents/__init__.py` — export v2 factories, remove old
3. `src/graph/setup.py` — new analyst types, simplified edges, parent retry logic
4. `src/graph/trading_graph.py` — 5 code agents, new tool nodes, analyst factories
5. `src/graph/conditional_logic.py` — simplified routing
6. `src/graph/propagation.py` — initialize new state fields
7. `src/agents/utils/agent_utils.py` — export new data tools
8. `src/dataflows/interface.py` — register new tool categories
9. `src/default_config.py` — code agent config, horizon config
10. `src/graph/reflection.py` — structured report consumption
11. `src/graph/signal_processing.py` — read from CompositeSignal
12. `cli/main.py` — new analyst names, progress tracking, config prompts
13. `pyproject.toml` — new dependencies

---

## Verification

1. **Unit test schemas**: verify Pydantic models serialize/deserialize correctly
2. **Unit test CodeValidationAgent**: verify it can generate, execute, and return results for a simple validation task (requires Ollama running)
3. **Unit test each new data tool**: verify data retrieval for a known ticker (AAPL)
4. **Integration test single analyst**: run Fundamental Analyst for AAPL with one horizon, verify `ResearchReport` output
5. **Integration test full graph**: run all 4 analysts for one ticker, verify Trader receives structured reports and produces trade proposal
6. **End-to-end test**: `discover_and_analyze()` with new Layer 2, verify complete pipeline output
