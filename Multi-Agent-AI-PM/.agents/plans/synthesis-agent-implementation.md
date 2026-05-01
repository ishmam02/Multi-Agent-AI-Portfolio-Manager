# Feature: Synthesis Agent Implementation (8b, 8d, 8e, 8g)

Read these files before implementing. Pay attention to naming of existing utils, types, and models.

## Feature Description

Replace the current free-text `Trader` node with a structured `Synthesis Agent` that consumes all four analyst `ResearchReport` outputs (JSON-serialized in `AgentState`), performs a 3-phase deterministic + LLM-assisted synthesis, and emits a `CompositeSignal` — the canonical Layer 2 output.

The Synthesis Agent does **not** generate trade proposals (direction, entry, targets, stop-loss). Those remain Layer 3 (Portfolio Management) responsibilities. Its job ends at producing the composite conviction signal with full provenance.

Additionally, update `AgentState` to store the `CompositeSignal`, integrate `FinancialSituationMemory` for contextual weighting, and add a post-processing step to `propagate_multi()` that computes a cross-asset covariance matrix from individual analyst sigma outputs.

## User Story

As a portfolio manager reviewing multi-analyst research, I want a structured composite signal that blends per-horizon analyst convictions with documented weights, identified conflicts, and mathematical validation, so that Layer 3 position-sizing decisions are grounded in reproducible, auditable math rather than free-text narratives.

## Problem Statement

The current `Trader` node (`src/agents/trader/trader.py`) produces an unstructured prose trade proposal. This is impossible to validate, backtest deterministically, or feed into portfolio optimizers. There is no explicit weighting of analyst inputs, no systematic contradiction detection, and no mathematical guarantee that the final signal is internally consistent.

## Solution Statement

Introduce a `create_synthesis_agent(llm, code_agent, memory)` factory that:
1. Parses all four `ResearchReport` JSON blobs from `AgentState`.
2. Uses the LLM for Phase 1 (dynamic weight assignment + rationale) and Phase 2 (cross-signal contradiction detection).
3. Uses the `code_agent` for Phase 3 (deterministic composite mu/sigma math + PSD validation).
4. Returns a JSON-serialized `CompositeSignal` written to `AgentState.composite_signal`.
5. Adds multi-ticker covariance post-processing in `TradingGraph.propagate_multi()`.

## Feature Metadata
- **Type**: Enhancement / Refactor
- **Complexity**: High
- **Systems Affected**:
  - `src/agents/utils/schemas.py`
  - `src/agents/utils/agent_states.py`
  - `src/agents/trader/trader.py`
  - `src/agents/__init__.py`
  - `src/graph/setup.py`
  - `src/graph/trading_graph.py`
  - `src/graph/conditional_logic.py`
  - `src/graph/propagation.py`
  - `src/graph/signal_processing.py`
  - `src/graph/reflection.py`
  - `cli/main.py`
  - `src/default_config.py`
- **Dependencies**: `pydantic` (already used), `numpy` (transitive via pandas; add explicit if used directly)

---

## CONTEXT REFERENCES

### Files to Read Before Implementing
- `src/agents/utils/schemas.py` (lines 1–118) — Why: Existing `ResearchReport`, `HorizonValues`, `HorizonThesis`, `Catalyst`, `Risk`, `ComputationTrace`, `ContributingFactor`, `Citation` schemas. The new `CompositeSignal` must reference these.
- `src/agents/utils/agent_states.py` (lines 1–36) — Why: Current `AgentState` definition with `trader_investment_plan`. Shows how reports are stored as strings and how `MessagesState` is extended.
- `src/agents/utils/memory.py` (lines 1–107) — Why: `FinancialSituationMemory` API (`add_situations`, `get_memories`). The synthesis agent must build a situation string from reports and call `get_memories`.
- `src/agents/trader/trader.py` (lines 1–100) — Why: Current `create_trader` to be replaced. Shows how reports are concatenated and how memory is used.
- `src/agents/analysts/base_analyst.py` (lines 1653–1806) — Why: `create_analyst_node` returns `{state_key: report.model_dump_json()}`. The synthesis agent must deserialize these JSON strings back into `ResearchReport`.
- `src/agents/code_agent/code_agent.py` (lines 1–150) — Why: `CodeValidationAgent.execute_plan()` signature and return format. The synthesis agent invokes the code_agent for Phase 3 math validation.
- `src/graph/setup.py` (lines 1–134) — Why: Graph wiring. `create_trader` is called here; must swap in `create_synthesis_agent` and pass a `code_agent`.
- `src/graph/trading_graph.py` (lines 1–459) — Why: `TradingGraph.__init__` creates code agents, `_log_state` logs `trader_investment_plan`, `propagate` returns signal, `propagate_multi` runs parallel tickers.
- `src/graph/conditional_logic.py` (lines 1–60) — Why: Conditional edge routing logic. Will be simplified since sub-graphs are self-contained.
- `src/graph/propagation.py` (lines 1–49) — Why: `create_initial_state` initializes `trader_investment_plan`.
- `src/graph/signal_processing.py` (lines 1–442) — Why: `SignalProcessor` and `propose_portfolio_trades` reference `trader_investment_plan`. These must read `composite_signal` instead.
- `src/graph/reflection.py` (lines 60–90) — Why: `reflect_trader` accesses `trader_investment_plan`. Update to `composite_signal`.
- `cli/main.py` (lines 1–1600+) — Why: Many references to `trader_investment_plan` for report rendering, streaming, and saving.
- `src/default_config.py` (lines 1–60) — Why: Default configuration values. New fields for code agent and research depth.

### New Files to Create
- None. All changes fit in existing files.

### Documentation to Read
- [Pydantic v2 BaseModel docs](https://docs.pydantic.dev/latest/concepts/models/) — Why: `model_dump_json()` vs `json()`.

### Patterns to Follow
**Analyst node return pattern (from `base_analyst.py:1803`):**
```python
return {state_key: report.model_dump_json()}
```

**Graph node factory pattern (from `trader/trader.py`):**
```python
def create_trader(llm, memory):
    def trader_node(state, name):
        ...
        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }
    return functools.partial(trader_node, name="Trader")
```

**Memory retrieval pattern (from `trader/trader.py:14–16`):**
```python
curr_situation = f"{market_research_report}\n\..."
past_memories = memory.get_memories(curr_situation, n_matches=2)
```

**StateGraph node registration (from `setup.py:110–115`):**
```python
workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
workflow.add_node("Analyst Sync", analyst_sync)
workflow.add_node("Trader", trader_node)
```

---

## STEP-BY-STEP TASKS

Execute every task in order. Each task is atomic and independently testable.

### Task 1: ADD CompositeSignal schemas to `src/agents/utils/schemas.py`
- **IMPLEMENT**: Add the following Pydantic models:
  - `AnalystWeights` — fields: `fundamental: float`, `technical: float`, `macro: float`, `sentiment: float` (sums to 1.0 per horizon)
  - `HorizonWeights` — fields: `long_term: AnalystWeights`, `medium_term: AnalystWeights`, `short_term: AnalystWeights`
  - `CrossSignalConflict` — fields: `analyst_a: AgentType`, `analyst_b: AgentType`, `horizon: str`, `conflict_description: str`, `resolution_status: str` ("resolved" | "unresolved"), `conviction_penalty: float` (–1.0 - 0.0)
  - `CompositeSignal` — fields:
    - `ticker: str`
    - `timestamp: datetime`
    - `mu_composite: HorizonValues` — blended per-horizon expected return
    - `sigma_composite: HorizonValues` — blended per-horizon volatility
    - `mu_final: float` — unified expected return after horizon blending
    - `sigma_final: float` — unified volatility
    - `conviction_final: float` — [0, +1.0]
    - `analyst_weights: HorizonWeights`
    - `horizon_blend_weights: HorizonValues` — weights used to blend long/medium/short into final
    - `weighting_rationale: str`
    - `cross_signal_conflicts: List[CrossSignalConflict]`
    - `unresolved_penalty: float` — total conviction reduction from unresolved conflicts
    - `source_reports: List[str]` — list of analyst `AgentType` values that contributed
    - `computation_provenance: str` — brief note on validation (e.g., "weights_sum_ok, psd_ok")
- **PATTERN**: Follow existing schema style: `BaseModel` subclasses, type annotations, docstrings.
- **VALIDATE**: `python -c "from src.agents.utils.schemas import CompositeSignal; print(CompositeSignal.__fields__.keys())"`

### Task 2: UPDATE AgentState in `src/agents/utils/agent_states.py`
- **IMPLEMENT**:
  - Add `composite_signal: Annotated[str, "JSON-serialized CompositeSignal from Synthesis Agent"]`
  - Keep `trader_investment_plan` for now but mark it deprecated in comment (removing it immediately breaks CLI and reflection which are updated in later tasks). Actually, **remove it** per spec. We will fix all references in subsequent tasks.
  - **Remove** `trader_investment_plan: Annotated[str, "Plan generated by the Trader"]`
  - Update report field names to match new analyst types: `market_report`, `news_report`, `fundamentals_report` (verify these already exist or map correctly)
- **PATTERN**: TypedDict field style with `Annotated[...]`.
- **VALIDATE**: `python -c "from src.agents.utils.agent_states import AgentState; print(list(AgentState.__annotations__.keys()))"`

### Task 3: REWRITE `src/agents/trader/trader.py` → `create_synthesis_agent`
- **IMPLEMENT**:
  - Rename factory to `create_synthesis_agent(llm, code_agent, memory)`.
  - The returned `synthesis_node(state, name)` performs:

    **Step A — Deserialize reports:**
    ```python
    reports = {}
    for key, agent_type in [
        ("market_report", AgentType.TECHNICAL),
        ("sentiment_report", AgentType.SENTIMENT),
        ("news_report", AgentType.NEWS),
        ("fundamentals_report", AgentType.FUNDAMENTAL),
    ]:
        raw = state.get(key, "")
        if raw:
            reports[agent_type] = ResearchReport.model_validate_json(raw)
    ```
    Handle missing reports gracefully (e.g., if only 2 analysts selected).

    **Step B — Build situation string for memory:**
    Concatenate `investment_thesis` and `key_catalysts` + `key_risks` from all deserialized reports. Call `memory.get_memories(curr_situation, n_matches=2)`.

    **Step C — Phase 1: Dynamic Weight Assignment (LLM call 1):**
    Prompt the LLM with:
    - All four reports (serialized as readable JSON)
    - Past memories (if any)
    - Instruction: "Assign per-analyst, per-horizon weights that sum to 1.0 per horizon. Provide a written rationale."
    
    LLM must return a JSON object matching:
    ```json
    {
      "rationale": "...",
      "weights": {
        "long_term": {"fundamental": 0.5, "technical": 0.15, "macro": 0.25, "sentiment": 0.1},
        "medium_term": {...},
        "short_term": {...}
      }
    }
    ```
    Parse with `_parse_json_from_text` (reuse logic from `base_analyst.py` or inline a lightweight parser). Normalize weights so each horizon sums to exactly 1.0.

    **Step D — Phase 2: Cross-Signal Consistency Check (LLM call 2 or same call):**
    Prompt the LLM to identify contradictions across analyst reports. Each contradiction yields a `CrossSignalConflict`. If conflicts are unresolved, apply a conviction penalty.
    
    LLM returns JSON:
    ```json
    {
      "conflicts": [
        {"analyst_a": "fundamental", "analyst_b": "technical", "horizon": "short_term", "description": "...", "resolution_status": "unresolved", "conviction_penalty": 0.15}
      ]
    }
    ```
    Parse and instantiate `CrossSignalConflict` objects.

    **Step E — Phase 3: Composite Signal Generation (deterministic code via code_agent):**
    Build a Python script that:
    1. Reads the parsed weights and per-horizon `mu` / `sigma_contribution` from each report.
    2. Computes `mu_composite(h) = Σ w_i(h) · μ_i(h)` for each horizon.
    3. Computes `sigma_composite(h) = sqrt(Σ w_i(h)^2 · σ_i(h)^2)` (assuming analyst independence within a ticker; this is a simplifying assumption documented in comments).
    4. Blends horizons into `mu_final` and `sigma_final` using LLM-determined horizon weights (or default equal weights if LLM did not provide them).
    5. Validates:
       - All per-horizon analyst weights sum to 1.0 (±1e-6).
       - All `mu_composite` values are within [-1.0, +1.0] (annualized return bounds).
       - `sigma_final > 0`.
    
    Pass this script to `code_agent.execute_plan()` or write it to a temp file and run it via the code agent's Bash tool. The code agent returns a dict with `mu_composite`, `sigma_composite`, `mu_final`, `sigma_final`, and validation flags.
    
    Alternatively, if using `CodeValidationAgent` directly is too heavy, construct a prompt that tells the code agent to execute the math and return JSON. The code agent already has JSON-parsing fallbacks.

    **Step F — Build CompositeSignal:**
    Instantiate `CompositeSignal` with all fields. JSON-serialize it.

    **Step G — Return dict:**
    ```python
    return {
        "messages": [result_msg],
        "composite_signal": composite_signal_json,
        "sender": name,
    }
    ```
  - Add file-based logging for the synthesis agent (follow pattern in `base_analyst.py` or keep simple stdout logging).
- **PATTERN**: Node factory returns `functools.partial(synthesis_node, name="Synthesis Agent")`.
- **VALIDATE**: `python -c "from src.agents.trader.trader import create_synthesis_agent; print('import ok')"`

### Task 4: UPDATE `src/agents/__init__.py`
- **IMPLEMENT**:
  - Replace `from .trader.trader import create_trader` with `from .trader.trader import create_synthesis_agent`
  - Replace `"create_trader"` in `__all__` with `"create_synthesis_agent"`
  - Remove old analyst imports (e.g., `create_social_media_analyst`) if they are deleted / no longer used
- **VALIDATE**: `python -c "from src.agents import create_synthesis_agent; print('ok')"`

### Task 5: UPDATE graph wiring in `src/graph/setup.py`
- **IMPLEMENT**:
  - In `GraphSetup.__init__`, add `code_agents: Dict[str, Any]` (already present) — no signature change needed except rename `trader_memory` to `memory` for clarity (optional).
  - In `setup_graph()`, replace:
    ```python
    trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)
    ```
    with:
    ```python
    # Use the fundamentals code_agent as the generic math executor for synthesis
    synthesis_code_agent = self.code_agents.get("fundamentals", list(self.code_agents.values())[0])
    synthesis_node = create_synthesis_agent(
        self.reasoning_llm,
        synthesis_code_agent,
        self.trader_memory,
    )
    ```
  - Rename graph node from `"Trader"` to `"Synthesis Agent"`.
  - Update conditional edge target from `"Trader"` to `"Synthesis Agent"`.
  - Update `required_reports` logic (already correct — just make sure it still maps to the 4 report keys).
  - New analyst types: `["fundamental", "market", "news"]`
  - New report_keys_map mapping to new state fields
  - Simplified conditional edges (sub-graphs are self-contained, no parent-level tool loops)
  - Each analyst node wrapper catches exceptions and retries with exponential backoff (max 2 retries). If still failing after retries, the analyst is **skipped** — its report field stays empty. The Synthesis Agent detects missing reports and adjusts weights accordingly.
- **VALIDATE**: `python -c "from src.graph.setup import GraphSetup; print('import ok')"`

### Task 6: UPDATE `src/graph/conditional_logic.py`
- **IMPLEMENT**:
  - Simplify conditional edge routing: no more tool-loop routing, just check if the report field is populated
  - Remove complex routing logic that directed analysts back to tool loops
- **VALIDATE**: `python -c "from src.graph.conditional_logic import get_next_node; print('ok')"`

### Task 7: UPDATE `src/graph/trading_graph.py`
- **IMPLEMENT**:
  - Create **3 separate `CodeValidationAgent` instances**, each with an analyst-specific system prompt and allowed library set:
    - `fundamental_code_agent` (valuation, accounting, financial modeling)
    - `market_code_agent` (indicators, signal processing, regime detection)
    - `news_code_agent` (sentiment quantification, behavioral signals)
  - All 3 use the same Ollama model but differ in system prompt and allowed libs
  - Update `_create_tool_nodes()` with new tool sets for each analyst
  - Pass reasoning LLM + analyst-specific code agent to each analyst factory
  - Pass synthesis_code_agent to the Synthesis Agent factory
  - Remove old analyst references (e.g., social_media)
  - In `propagate()` (line ~190), change return from:
    ```python
    return final_state, self.process_signal(final_state["trader_investment_plan"])
    ```
    to:
    ```python
    composite = final_state.get("composite_signal", "")
    decision = "HOLD"
    if composite:
        try:
            cs = CompositeSignal.model_validate_json(composite)
            decision = "BUY" if cs.mu_final > 0.05 else "SELL" if cs.mu_final < -0.05 else "HOLD"
        except Exception:
            pass
    return final_state, decision
    ```
    (This keeps the return signature `(final_state, decision)` compatible with existing callers.)
  - In `_log_state()`, replace `"trader_investment_plan"` key with `"composite_signal"`.
  - In `discover_and_analyze()` (line ~393), replace `state.get("trader_investment_plan", "")` with `state.get("composite_signal", "")`.
  - Add multi-ticker covariance method `compute_multi_ticker_covariance(analysis_results: Dict[str, Dict]) -> Dict[str, Any]`:
    ```python
    def compute_multi_ticker_covariance(self, analysis_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute cross-asset covariance matrix from composite signals.
        
        Returns dict with:
          - tickers: List[str]
          - covariance_matrix: List[List[float]]  # n x n
          - annualized_volatilities: Dict[str, float]
        """
    ```
    Implementation steps:
    1. For each ticker, parse `composite_signal` JSON into `CompositeSignal`.
    2. Extract `sigma_final` (unified volatility) for each ticker.
    3. Extract per-analyst sigma contributions from the underlying reports (available in `market_report`, `sentiment_report`, etc., inside `analysis_results[ticker]`). These are still JSON strings.
    4. Build an `n x n` covariance matrix. Since cross-ticker correlations are not directly observed, use a single-factor model:
       - Let `σ_j` = `sigma_final` for ticker `j`.
       - Assume a market-factor correlation of `ρ = 0.3` (configurable default).
       - `Cov(j, k) = ρ · σ_j · σ_k` for `j != k`.
       - `Var(j) = σ_j^2`.
       - Ensure the matrix is symmetric and positive semi-definite.
    5. Return the dict.
    
    Call this method at the end of `propagate_multi()` after all results are collected, and attach the covariance dict under a new key in the returned dict (or store it as an instance attribute).
- **VALIDATE**: `python -c "from src.graph.trading_graph import TradingGraph; print('import ok')"`

### Task 8: UPDATE initial state in `src/graph/propagation.py`
- **IMPLEMENT**:
  - Replace `"trader_investment_plan": ""` with `"composite_signal": ""` in `create_initial_state`.
  - Initialize new state fields to empty strings (if any new report fields are added).
- **VALIDATE**: `python -c "from src.graph.propagation import Propagator; s = Propagator().create_initial_state('AAPL','2024-01-01'); print('composite_signal' in s)"`

### Task 9: UPDATE `src/graph/signal_processing.py`
- **IMPLEMENT**:
  - In `propose_portfolio_trades()` (line ~172), update docstring and parameter from `ticker_signals: Dict[str, str]` referencing `trader_investment_plan_text` to `ticker_signals: Dict[str, str]` referencing `composite_signal_json`.
  - The method currently passes the raw signal text to the LLM. Change it to parse each `composite_signal` JSON and include `mu_final`, `sigma_final`, `conviction_final`, and `weighting_rationale` in the prompt. This gives the Portfolio Manager structured data instead of prose.
  - Alternatively, if the prompt rewrite is too large, keep passing the JSON string and let the LLM parse it (simpler, less code change).
  - `process_signal()` now reads from `CompositeSignal.trade_proposal` (structured) instead of parsing free-text for BUY/SELL/HOLD. The `CompositeSignal` already contains the direction, so extraction is simpler.
  - `propose_trade()` can use `mu_final`, `conviction_final`, and the trade proposal to generate more precise Alpaca order parameters.
- **VALIDATE**: `python -c "from src.graph.signal_processing import SignalProcessor; print('import ok')"`

### Task 10: UPDATE `src/graph/reflection.py`
- **IMPLEMENT**:
  - Find `reflect_trader()` (around line 76) which accesses `current_state["trader_investment_plan"]`.
  - Change it to read `current_state.get("composite_signal", "")`.
  - Parse the JSON and use `mu_final` / `conviction_final` as the "decision" to reflect on.
  - `_extract_current_situation()` now reads structured `ResearchReport` JSON fields instead of old free-form report fields.
  - Reflection prompt updated to reference composite signals, computed metrics, and thesis narratives.
  - `reflect_trader()` uses `composite_signal` as the decision to reflect on.
- **VALIDATE**: `python -c "from src.graph.reflection import Reflector; print('import ok')"`

### Task 11: UPDATE CLI in `cli/main.py`
- **IMPLEMENT**: Replace every reference to `trader_investment_plan` with `composite_signal`. Key locations (from grep):
  - Line 69: `REPORT_SECTIONS` dict — change `"trader_investment_plan"` key to `"composite_signal"`, change finalizing agent name from `"Trader"` to `"Synthesis Agent"`.
  - Line 175: `self.report_sections["composite_signal"] = "Synthesis Signal"`
  - Lines 214–216: report rendering for composite_signal.
  - Lines 668–673: file writing and markdown section header — write `composite_signal` JSON (pretty-printed) instead of free text.
  - Lines 709–715: `Markdown(final_state["composite_signal"])` in rich display.
  - Lines 1039–1041: streaming chunk capture — `chunk.get("composite_signal")`.
  - Line 1053: `graph.process_signal(final_state["trader_investment_plan"])` → `graph.process_signal(final_state.get("composite_signal", ""))` (or better, parse composite JSON and extract decision).
  - Line 1111: reference in some other function.
  - Lines 1514–1516: another streaming capture.
  - Line 1571: `state.get("trader_investment_plan", "")` → `state.get("composite_signal", "")`.
  - Analyst selection updated: offer `["fundamental", "technical", "macro", "sentiment"]` instead of old types
  - `MessageBuffer` progress tracking updated for new agent names and 3-phase workflow stages
  - Streaming output shows per-horizon progress (e.g., "Fundamental Analyst: long_term thesis → code validation → ✓")
  - Results display shows structured output: composite mu, conviction, key metrics, conflicts
  - Config prompts include `code_agent_model` and `code_agent_base_url` options
- **PATTERN**: The CLI currently renders `trader_investment_plan` as markdown. For `composite_signal` (JSON), pretty-print it with `json.dumps(parsed, indent=2)` before displaying.
- **VALIDATE**: `python -c "from cli.main import app; print('import ok')"` (or at least syntax-check with `python -m py_compile cli/main.py`)

### Task 12: UPDATE Config in `src/default_config.py`
- **IMPLEMENT**: Add the following new defaults:
  ```python
  "code_agent_model": "qwen2.5-coder:32b",       # Ollama model for code generation
  "code_agent_base_url": "http://localhost:11434",  # Ollama endpoint
  "code_agent_max_iterations": 5,                   # Max generate-execute-fix cycles
  "research_depth": "shallow",                        # "shallow" (1 pass), "medium" (up to 2), "deep" (up to 3)
  "horizons_enabled": ["long_term", "medium_term", "short_term"],
  "horizon_lookback": {"long_term": 365, "medium_term": 90, "short_term": 10},
  ```
- **VALIDATE**: `python -c "from src.default_config import DEFAULT_CONFIG; print('code_agent_model' in DEFAULT_CONFIG)"`

### Task 13: HANDLE the deleted `social_media_analyst.py`
- **IMPLEMENT**: The working tree has `src/agents/analysts/social_media_analyst.py` deleted, but `setup.py` still calls `create_social_media_analyst`. This will cause `ImportError` on graph setup. Either:
  - **Option A**: Restore the file from git (`git checkout HEAD -- src/agents/analysts/social_media_analyst.py`), or
  - **Option B**: Remove `"social"` from the default `selected_analysts` in `TradingGraph.__init__` and `setup_graph()`, and update `src/agents/__init__.py` to remove the import.
  
  **Recommended**: Option A (restore the file) unless the user explicitly wants it gone. The feature spec assumes 4 analysts exist. Add a note in the plan.
- **VALIDATE**: `python -c "from src.agents import create_social_media_analyst; print('ok')"` (if restoring) or ensure no import errors if removing.

---

## TESTING STRATEGY

### Unit Tests
- **Schema validation**: Instantiate `CompositeSignal` with valid and invalid data. Assert `ValidationError` on bad weights (e.g., sum != 1.0). Note: Pydantic won't auto-enforce sum-to-1; that is done in code. Add a helper `normalize_weights()` and test it.
- **Synthesis agent node**: Mock `llm.invoke` to return a canned weights JSON. Mock `code_agent.execute_plan` to return deterministic math results. Verify the returned `composite_signal` JSON parses into a `CompositeSignal` with expected `mu_final`.
- **Covariance computation**: Feed 3 mock tickers with known `sigma_final` values. Assert the returned covariance matrix is symmetric, PSD, and diagonal entries equal `σ^2`.
- **Analyst resilience**: Mock an analyst node that raises an exception twice. Assert the report field remains empty and synthesis still produces a valid signal with redistributed weights.

### Integration Tests
- Run the full graph for a single ticker with `--debug` and verify `composite_signal` is present in the final state.
- Run `propagate_multi` with 2 tickers and verify the covariance matrix is attached.
- Verify CLI streaming mode correctly captures `composite_signal` chunks.
- Verify reflection reads composite signal and produces meaningful critique.

### Edge Cases
- **Missing analyst**: Only 2 of 3 analysts selected. Weights for missing analysts should be 0.0 and the remaining weights should still sum to 1.0.
- **Malformed LLM JSON**: LLM returns weights that don't parse. Fallback to equal weights.
- **Negative sigma**: Code agent returns invalid sigma. Fallback to a small positive epsilon (0.01).
- **Empty memory**: `memory.get_memories` returns `[]`. Phase 1 should proceed without past context.
- **Single ticker in propagate_multi**: Covariance matrix should be 1x1.
- **Analyst failure after retries**: Report field stays empty; synthesis agent compensates.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```bash
python -m py_compile src/agents/utils/schemas.py
python -m py_compile src/agents/utils/agent_states.py
python -m py_compile src/agents/trader/trader.py
python -m py_compile src/agents/__init__.py
python -m py_compile src/graph/setup.py
python -m py_compile src/graph/trading_graph.py
python -m py_compile src/graph/conditional_logic.py
python -m py_compile src/graph/propagation.py
python -m py_compile src/graph/signal_processing.py
python -m py_compile src/graph/reflection.py
python -m py_compile cli/main.py
python -m py_compile src/default_config.py
```

### Level 2: Unit Tests
```bash
python -m pytest tests/ -x -q --tb=short
```
(Note: existing tests are sparse; at minimum verify no import regressions.)

### Level 3: Integration / Smoke Test
```bash
python -c "
from src.graph.trading_graph import TradingGraph
from src.agents.utils.schemas import CompositeSignal
import json

# Verify schemas import
print('Schemas OK')

# Verify graph can be instantiated (won't run without API keys)
# graph = TradingGraph(selected_analysts=['market', 'fundamentals'])
# print('Graph init OK')
"
```

### Level 4: Manual Validation
1. Run CLI for a single ticker with `--debug`.
2. Inspect `results/{ticker}/{date}/report.log` — should contain `CompositeSignal` JSON instead of prose trade plan.
3. Inspect `eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_*.json` — should have `composite_signal` key.

---

## ACCEPTANCE CRITERIA
- [ ] `CompositeSignal`, `CrossSignalConflict`, `AnalystWeights`, `HorizonWeights` schemas exist and validate correctly.
- [ ] `AgentState` contains `composite_signal` and no longer contains `trader_investment_plan`.
- [ ] `create_synthesis_agent` produces a JSON-serialized `CompositeSignal` from 4 analyst reports via 3-phase workflow (LLM weights → LLM conflicts → code math + validation).
- [ ] Per-horizon analyst weights sum to 1.0 (±1e-6) enforced in code.
- [ ] Cross-signal conflicts are documented in `CompositeSignal.cross_signal_conflicts`.
- [ ] `mu_final`, `sigma_final`, and `conviction_final` are populated deterministically from the fixed formula.
- [ ] `FinancialSituationMemory` is queried during Phase 1 with a situation string built from reports.
- [ ] `propagate_multi()` computes and returns a cross-asset covariance matrix after all individual syntheses complete.
- [ ] All references to `trader_investment_plan` across `trading_graph.py`, `propagation.py`, `signal_processing.py`, `reflection.py`, and `cli/main.py` are updated.
- [ ] No `ImportError` on graph initialization (i.e., `social_media_analyst.py` issue resolved).
- [ ] CLI renders the `composite_signal` JSON in a human-readable format.
- [ ] Analyst node wrappers catch exceptions, retry with exponential backoff (max 2 retries), and skip failing analysts gracefully.
- [ ] Synthesis Agent compensates for missing analysts by redistributing weights and noting the omission in `weighting_rationale`.
- [ ] `default_config.py` includes new fields: `code_agent_model`, `code_agent_base_url`, `code_agent_max_iterations`, `research_depth`, `horizons_enabled`, `horizon_lookback`.

## COMPLETION CHECKLIST
- [ ] All tasks completed in order
- [ ] All validation commands pass
- [ ] Full test suite passes (or at least no new regressions)
- [ ] Ready for `/commit`
