"""
Synthesis Agent — structured composite signal generation.

Replaces the free-text Trader node with a 3-phase deterministic + LLM-assisted
synthesis that consumes analyst ResearchReport outputs and emits a
CompositeSignal.
"""

from __future__ import annotations

import functools
import json
import math
from datetime import datetime
import re
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.utils.schemas import (
    AgentType,
    AnalystWeights,
    CompositeSignal,
    CrossSignalConflict,
    HorizonThesis,
    HorizonValues,
    HorizonWeights,
    ResearchReport,
)


# ── Constants ────────────────────────────────────────────────────────────────

AGENT_TYPE_MAP = {
    "market_report": AgentType.MARKET,
    "fundamentals_report": AgentType.FUNDAMENTAL,
}

AGENT_TYPE_KEYS = {
    AgentType.FUNDAMENTAL: "fundamental",
    AgentType.MARKET: "market",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_json_block(text: str) -> dict:
    """Extract the first JSON object or array from text."""
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
            if isinstance(d, (dict, list)):
                return d
        except (json.JSONDecodeError, ValueError):
            continue

    return {}


def _normalize_weights(weights: dict, horizons: tuple) -> dict:
    """Normalize per-horizon analyst weights to sum to 1.0 (±1e-6)."""
    normalized = {}
    for horizon in horizons:
        w = weights.get(horizon, {})
        if not isinstance(w, dict):
            w = {}
        vals = {
            "fundamental": float(w.get("fundamental", 0.0)),
            "market": float(w.get("market", 0.0)),
        }
        total = sum(vals.values())
        if total > 0:
            vals = {k: v / total for k, v in vals.items()}
        else:
            vals = {k: 1.0 / len(vals) for k in vals}
        normalized[horizon] = vals
    return normalized


def _default_equal_weights(horizons: tuple) -> dict:
    """Return equal weights for all analysts across the selected horizons."""
    eq = 0.5
    return {h: {"fundamental": eq, "market": eq} for h in horizons}


def _build_horizon_weights_example(horizons: tuple) -> str:
    """Build the example weights JSON block for the selected horizons."""
    eq = '{"fundamental": 0.5, "market": 0.5}'
    lines = [f'    "{h}": {eq}' for h in horizons]
    return "\n".join(lines)


def _build_horizon_blend_example(horizons: tuple) -> str:
    """Build the example horizon_blend_weights JSON for the selected horizons."""
    n = len(horizons)
    eq = round(1.0 / n, 2)
    parts = [f'"{h}": {eq}' for h in horizons]
    return ", ".join(parts)


def _build_horizon_list(horizons: tuple) -> str:
    """Comma-separated horizon names for prompt text."""
    return ", ".join(horizons)


# ── Synthesis Agent Factory ──────────────────────────────────────────────────


def create_synthesis_agent(llm, horizons=("long_term", "medium_term", "short_term")):
    """Create a synthesis node function for the outer AgentState graph.

    Parameters
    ----------
    llm      : LangChain chat model for Phase 1 (weights) and Phase 2 (conflicts)
    horizons : Tuple of horizon strings to include in synthesis.
               Default: ("long_term", "medium_term", "short_term")
    """
    # Safety: ensure all three HorizonValues fields get a value
    _all_horizons = ("long_term", "medium_term", "short_term")

    def synthesis_node(state, name):
        company_name = state["company_of_interest"]

        # ── Step A: Deserialize reports ─────────────────────────────────────
        reports: Dict[AgentType, ResearchReport] = {}
        for key, agent_type in AGENT_TYPE_MAP.items():
            raw = state.get(key, "")
            if raw:
                try:
                    reports[agent_type] = ResearchReport.model_validate_json(raw)
                except Exception:
                    pass

        if not reports:
            # No reports available — emit a neutral signal
            mu_defaults = {h: 0.0 for h in _all_horizons}
            sigma_defaults = {h: 0.01 for h in _all_horizons}
            blend_defaults = {h: round(1.0 / len(horizons), 2) for h in _all_horizons}
            composite = CompositeSignal(
                ticker=company_name,
                timestamp=datetime.now(),
                mu_composite=HorizonValues(**mu_defaults),
                sigma_composite=HorizonValues(**sigma_defaults),
                mu_final=0.0,
                sigma_final=0.01,
                conviction_final=0.0,
                analyst_weights=HorizonWeights(),
                horizon_blend_weights=HorizonValues(**blend_defaults),
                weighting_rationale="No analyst reports available; defaulting to neutral signal.",
                cross_signal_conflicts=[],
                unresolved_penalty=0.0,
                source_reports=[],
            )
            return {
                "messages": [
                    HumanMessage(content="No reports available — neutral signal.")
                ],
                "composite_signal": composite.model_dump_json(),
                "sender": name,
            }

        # ── Phase 1: Dynamic Weight Assignment (LLM) ────────────────────────
        reports_json = []
        for agent_type, report in reports.items():
            entry: dict = {"analyst": agent_type.value}
            # Scalar per-horizon values
            for field, src in (
                ("mu", report.mu),
                ("sigma", report.sigma_contribution),
                ("conviction", report.conviction),
            ):
                entry[field] = {h: getattr(src, h, 0.0) for h in horizons}
            # Trace IDs for mu / sigma per horizon
            entry["mu_trace_id"] = {h: getattr(report.mu_trace_id, h, "") for h in horizons}
            entry["sigma_trace_id"] = {h: getattr(report.sigma_trace_id, h, "") for h in horizons}
            # Key catalysts with trace IDs
            entry["key_catalysts"] = [
                {"catalyst": c.catalyst, "term": c.term, "metric_name": c.metric_name, "trace_id": c.computation_trace_id}
                for c in report.key_catalysts[:3]
            ]
            # Key risks with trace IDs
            entry["key_risks"] = [
                {"risk": r.risk, "term": r.term, "metric_name": r.metric_name, "trace_id": r.computation_trace_id}
                for r in report.key_risks[:3]
            ]
            # Build trace_id -> metric_name lookup from computed_metrics
            _trace_to_metric = {
                m.computation_trace_id: m.metric_name
                for m in report.computed_metrics
                if m.computation_trace_id
            }
            # Computation traces available
            entry["available_traces"] = [
                {
                    "trace_id": t.trace_id,
                    "metric_name": _trace_to_metric.get(t.trace_id, "unknown"),
                    "description": t.code[:80] + "..." if len(t.code) > 80 else t.code,
                }
                for t in report.computation_traces[:5]
            ]
            reports_json.append(entry)

        horizon_names = _build_horizon_list(horizons)
        weights_example = _build_horizon_weights_example(horizons)
        blend_example = _build_horizon_blend_example(horizons)

        phase1_prompt = f"""You are a senior portfolio manager synthesizing multi-analyst research into a unified signal.

Your task is to assign per-analyst, per-horizon weights that sum to 1.0 for each horizon.

Available analysts and their signals:
{json.dumps(reports_json, indent=2)}

Instructions:
1. Assign weights for each horizon ({horizon_names}).
2. Each horizon's weights must sum to exactly 1.0.
3. Provide a concise rationale (2-3 sentences) explaining your weighting logic. When referencing any metric, catalyst, or risk, cite it inline using the exact format [metric_name | trace:<computation_trace_id>] (e.g., "market bullish momentum [rsi | trace:abc-123] outweighs fundamental overvaluation concern [dcf_intrinsic_value | trace:def-456]").
4. If an analyst type is missing, assign it weight 0.0 and redistribute.
5. Also provide horizon_blend_weights (how much to weight each selected horizon when blending into a single final signal). These must also sum to 1.0.

Respond with a single JSON object in this exact structure:
{{
  "rationale": "...",
  "weights": {{
{weights_example}
  }},
  "horizon_blend_weights": {{{blend_example}}}
}}

Output ONLY the JSON object. No markdown fences, no extra text."""

        phase1_result = llm.invoke([
            SystemMessage(content="You are a senior portfolio manager synthesizing multi-analyst research."),
            HumanMessage(content=phase1_prompt),
        ])
        phase1_json = _extract_json_block(phase1_result.content)

        weights_dict = _default_equal_weights(horizons)
        horizon_blend = {h: round(1.0 / len(horizons), 2) for h in horizons}
        rationale = "Fallback to equal weights due to parse failure."

        if isinstance(phase1_json, dict):
            rationale = phase1_json.get("rationale", rationale)
            raw_weights = phase1_json.get("weights")
            if isinstance(raw_weights, dict):
                weights_dict = _normalize_weights(raw_weights, horizons)
            raw_blend = phase1_json.get("horizon_blend_weights")
            if isinstance(raw_blend, dict):
                blend_vals = {
                    h: float(raw_blend.get(h, horizon_blend[h])) for h in horizons
                }
                total = sum(blend_vals.values())
                if total > 0:
                    horizon_blend = {k: v / total for k, v in blend_vals.items()}
                else:
                    horizon_blend = {h: round(1.0 / len(horizons), 2) for h in horizons}

        # ── Phase 2: Cross-Signal Consistency Check (LLM) ───────────────────
        phase2_prompt = f"""You are a risk analyst reviewing multi-analyst research for contradictions.

Analyst signals:
{json.dumps(reports_json, indent=2)}

Your task:
1. Identify contradictions between analysts (e.g., fundamental says bullish, market says bearish).
2. For each contradiction, specify:
   - analyst_a and analyst_b (use exactly: fundamental, market)
   - horizon ({horizon_names})
   - conflict_description (1 sentence). When referencing any metric, catalyst, or risk, cite it inline using the exact format [metric_name | trace:<computation_trace_id>] (e.g., "fundamental bearish DCF [dcf_intrinsic_value | trace:abc-123] contradicts market bullish RSI [rsi | trace:def-456]").
   - resolution_status: "resolved" if one analyst's reasoning clearly outweighs the other, else "unresolved"
   - conviction_penalty: a negative float between -1.0 and 0.0 (e.g., -0.15). More severe = larger magnitude.
3. IMPORTANT: Only report genuinely distinct contradictions. If the same core disagreement appears in multiple forms (e.g., mu disagreement and risk disagreement for the same analyst pair on the same horizon), report it ONCE with the most severe penalty. Do not double-count the same underlying conflict.

Respond with a single JSON object:
{{
  "conflicts": [
    {{
      "analyst_a": "fundamental",
      "analyst_b": "market",
      "horizon": "{horizons[-1]}",
      "conflict_description": "Fundamental DCF implies 20% upside while market RSI shows overbought exhaustion.",
      "resolution_status": "unresolved",
      "conviction_penalty": -0.12
    }}
  ]
}}

If no contradictions exist, return {{"conflicts": []}}.
Output ONLY the JSON object. No markdown fences, no extra text."""

        phase2_result = llm.invoke([
            SystemMessage(content="You are a risk analyst reviewing multi-analyst research for contradictions."),
            HumanMessage(content=phase2_prompt),
        ])
        phase2_json = _extract_json_block(phase2_result.content)

        conflicts: List[CrossSignalConflict] = []
        unresolved_penalty = 0.0
        if isinstance(phase2_json, dict):
            raw_conflicts = phase2_json.get("conflicts", [])
            if isinstance(raw_conflicts, list):
                for c in raw_conflicts:
                    if not isinstance(c, dict):
                        continue
                    try:
                        a_type = AgentType(c.get("analyst_a", ""))
                        b_type = AgentType(c.get("analyst_b", ""))
                    except ValueError:
                        continue
                    penalty = float(c.get("conviction_penalty", 0.0))
                    conflicts.append(
                        CrossSignalConflict(
                            analyst_a=a_type,
                            analyst_b=b_type,
                            horizon=c.get("horizon", ""),
                            conflict_description=c.get("conflict_description", ""),
                            resolution_status=c.get("resolution_status", "unresolved"),
                            conviction_penalty=max(-1.0, min(0.0, penalty)),
                        )
                    )
                    if c.get("resolution_status") != "resolved":
                        unresolved_penalty += abs(penalty)

        unresolved_penalty = min(unresolved_penalty, 1.0)

        # ── Phase 2.5: Composite Thesis Generation (LLM) ─────────────────────
        _thesis_prompt = f"""You are a senior portfolio manager writing a unified investment thesis by blending multi-analyst research.

Analyst theses:
{json.dumps(reports_json, indent=2)}

Assigned analyst weights:
{json.dumps(weights_dict, indent=2)}

Identified conflicts:
{json.dumps([{"analyst_a": c.analyst_a.value, "analyst_b": c.analyst_b.value, "horizon": c.horizon, "description": c.conflict_description} for c in conflicts], indent=2)}

Your task:
1. Write a single unified investment thesis for each horizon ({horizon_names}).
2. Blend the analyst narratives according to their assigned weights. Higher-weighted analysts should dominate the narrative.
3. When referencing any metric, catalyst, or risk, cite it inline using the exact format [metric_name | trace:<computation_trace_id>].
4. Acknowledge unresolved conflicts briefly but do not let them derail the overall direction.

Respond with a single JSON object:
{{
  "composite_thesis": {{
    "long_term": "...",
    "medium_term": "...",
    "short_term": "..."
  }}
}}

If a horizon has no signal, return an empty string for that horizon.
Output ONLY the JSON object. No markdown fences, no extra text."""

        _thesis_result = llm.invoke([
            SystemMessage(content="You are a senior portfolio manager writing a unified investment thesis."),
            HumanMessage(content=_thesis_prompt),
        ])
        _thesis_json = _extract_json_block(_thesis_result.content)
        _composite_thesis = {h: "" for h in _all_horizons}
        if isinstance(_thesis_json, dict):
            _raw_thesis = _thesis_json.get("composite_thesis", {})
            if isinstance(_raw_thesis, dict):
                for h in _all_horizons:
                    _composite_thesis[h] = str(_raw_thesis.get(h, ""))[:500]

        # ── Phase 3: Composite Signal Math (deterministic) ──────────────────
        mu_comp: Dict[str, float] = {}
        sigma_comp: Dict[str, float] = {}

        for horizon in horizons:
            mu_h = 0.0
            sigma_sq = 0.0
            w = weights_dict.get(horizon, {})
            for agent_type, report in reports.items():
                key = AGENT_TYPE_KEYS.get(agent_type, agent_type.value)
                weight = w.get(key, 0.0)
                mu_val = getattr(report.mu, horizon, 0.0)
                sigma_val = getattr(report.sigma_contribution, horizon, 0.01)
                mu_h += weight * mu_val
                sigma_sq += (weight**2) * (sigma_val**2)
            mu_comp[horizon] = max(-1.0, min(1.0, mu_h))
            sigma_comp[horizon] = max(0.001, math.sqrt(sigma_sq))

        # Fill missing horizons with zeros so HorizonValues schema is satisfied
        for h in _all_horizons:
            if h not in mu_comp:
                mu_comp[h] = 0.0
            if h not in sigma_comp:
                sigma_comp[h] = 0.01

        # Blend horizons into final mu / sigma
        mu_final = sum(horizon_blend[h] * mu_comp[h] for h in horizons)
        sigma_final = math.sqrt(
            sum((horizon_blend[h] ** 2) * (sigma_comp[h] ** 2) for h in horizons)
        )
        sigma_final = max(0.001, sigma_final)

        # Conviction: scaled by final mu magnitude, reduced by unresolved penalties
        # Cap penalty at 0.5 so multiple conflicts cannot completely zero conviction
        raw_conviction = abs(mu_final)
        effective_penalty = min(unresolved_penalty, 0.5)
        conviction_final = max(0.0, min(1.0, raw_conviction * (1.0 - effective_penalty)))

        # Validate weight sums
        weights_ok = all(
            abs(sum(weights_dict.get(h, {}).values()) - 1.0) < 1e-6 for h in horizons
        )
        provenance_parts = ["weights_sum_ok" if weights_ok else "weights_sum_fail"]
        if sigma_final > 0:
            provenance_parts.append("sigma_ok")
        if abs(mu_final) <= 1.0:
            provenance_parts.append("mu_bounds_ok")

        # Validate weight sums
        weights_ok = all(
            abs(sum(weights_dict.get(h, {}).values()) - 1.0) < 1e-6 for h in horizons
        )
        provenance_parts = ["weights_sum_ok" if weights_ok else "weights_sum_fail"]
        if sigma_final > 0:
            provenance_parts.append("sigma_ok")
        if abs(mu_final) <= 1.0:
            provenance_parts.append("mu_bounds_ok")
        computation_provenance = ", ".join(provenance_parts)

        # ── Build CompositeSignal ─────────────────────────────────────────────
        # Populate all three HorizonWeights fields; missing horizons get default 0/0
        def _aw(weights_dict: dict, h: str) -> AnalystWeights:
            w = weights_dict.get(h, {})
            return AnalystWeights(
                fundamental=float(w.get("fundamental", 0.0)),
                market=float(w.get("market", 0.0)),
            )

        analyst_weights = HorizonWeights(
            long_term=_aw(weights_dict, "long_term"),
            medium_term=_aw(weights_dict, "medium_term"),
            short_term=_aw(weights_dict, "short_term"),
        )

        # Fill blend values for all horizons
        full_blend = {h: horizon_blend.get(h, 0.0) for h in _all_horizons}

        composite = CompositeSignal(
            ticker=company_name,
            timestamp=datetime.now(),
            mu_composite=HorizonValues(**mu_comp),
            sigma_composite=HorizonValues(**sigma_comp),
            mu_final=mu_final,
            sigma_final=sigma_final,
            conviction_final=conviction_final,
            analyst_weights=analyst_weights,
            horizon_blend_weights=HorizonValues(**full_blend),
            weighting_rationale=rationale,
            composite_thesis=HorizonThesis(**_composite_thesis),
            cross_signal_conflicts=conflicts,
            unresolved_penalty=unresolved_penalty,
            source_reports=[at.value for at in reports.keys()],
            computation_provenance=computation_provenance,
        )

        composite_json = composite.model_dump_json()

        result_msg = HumanMessage(
            content=f"Synthesis complete for {company_name}. "
            f"mu_final={mu_final:.4f}, sigma_final={sigma_final:.4f}, "
            f"conviction={conviction_final:.4f}, conflicts={len(conflicts)}."
        )

        return {
            "messages": [result_msg],
            "composite_signal": composite_json,
            "sender": name,
        }

    return functools.partial(synthesis_node, name="Synthesis Agent")


if __name__ == "__main__":
    import json
    import pathlib
    import sys

    # ── Ensure src/ is on the path when run directly ──────────────────────────
    _project_root = str(pathlib.Path(__file__).resolve().parents[3])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from src.llm_clients import create_llm_client
    from src.agents.code_agent.code_agent import CodeValidationAgent
    from src.agents.analysts.market_analyst import create_market_analyst
    from src.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
    from src.agents.utils.schemas import CompositeSignal, ResearchReport

    # ── Configuration ─────────────────────────────────────────────────────────
    TICKER = "AAPL"
    TRADE_DATE = "2026-04-24"
    RESEARCH_DEPTH = "shallow"
    RANDOM_SEED = 42

    llm_client = create_llm_client(
        provider="ollama",
        model="minimax-m2.7:cloud",
        base_url="http://localhost:11434/v1",
        max_retries=10,
        reasoning_effort="high",
        temperature=0,
        seed=RANDOM_SEED,
    )
    reasoning_llm = llm_client.get_llm()

    # ── Create analyst code agents ────────────────────────────────────────────
    market_code_agent = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        timeout=60,
        max_iterations=5,
        analyst_type="market",
        project_root=_project_root,
        verbose=True,
    )
    fundamentals_code_agent = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        timeout=60,
        max_iterations=5,
        analyst_type="fundamental",
        project_root=_project_root,
        verbose=True,
    )

    # ── Create analyst nodes ──────────────────────────────────────────────────
    market_node = create_market_analyst(reasoning_llm, market_code_agent, RESEARCH_DEPTH)
    fundamentals_node = create_fundamentals_analyst(
        reasoning_llm, fundamentals_code_agent, RESEARCH_DEPTH
    )

    init_state = {
        "company_of_interest": TICKER,
        "trade_date": TRADE_DATE,
    }

    # ── Run analysts in parallel ────────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n{'=' * 60}")
    print(f"Running analysts in parallel for {TICKER} on {TRADE_DATE} ...")
    print(f"{'=' * 60}")

    def _run_market():
        print("  [Market] starting ...")
        result = market_node(init_state)
        report_json = result.get("market_report", "{}")
        try:
            report = ResearchReport.model_validate_json(report_json)
            print(f"  [Market] done — mu={report.mu.long_term:+.4f}, conviction={report.conviction.long_term:.2f}")
            return "market", report_json, report
        except Exception as exc:
            print(f"  [Market] WARNING: Failed to parse report: {exc}")
            return "market", report_json, None

    def _run_fundamentals():
        print("  [Fundamentals] starting ...")
        result = fundamentals_node(init_state)
        report_json = result.get("fundamentals_report", "{}")
        try:
            report = ResearchReport.model_validate_json(report_json)
            print(f"  [Fundamentals] done — mu={report.mu.long_term:+.4f}, conviction={report.conviction.long_term:.2f}")
            return "fundamentals", report_json, report
        except Exception as exc:
            print(f"  [Fundamentals] WARNING: Failed to parse report: {exc}")
            return "fundamentals", report_json, None

    market_report_json = "{}"
    fundamentals_report_json = "{}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_market): "market",
            executor.submit(_run_fundamentals): "fundamentals",
        }
        for future in as_completed(futures):
            analyst_key, report_json, report = future.result()
            if analyst_key == "market":
                market_report_json = report_json
            else:
                fundamentals_report_json = report_json

    # ── Prepare state for synthesis ─────────────────────────────────────────
    synthesis_state = {
        "company_of_interest": TICKER,
        "trade_date": TRADE_DATE,
        "market_report": market_report_json,
        "fundamentals_report": fundamentals_report_json,
    }

    # ── Run synthesis agent ───────────────────────────────────────────────────
    synthesis_node = create_synthesis_agent(reasoning_llm)

    print(f"\n{'=' * 60}")
    print(f"Running SYNTHESIS for {TICKER} on {TRADE_DATE} ...")
    print(f"{'=' * 60}")
    result = synthesis_node(synthesis_state)

    composite_json = result.get("composite_signal", "{}")
    try:
        composite = CompositeSignal.model_validate_json(composite_json)
    except Exception as exc:
        print(f"Failed to parse CompositeSignal: {exc}")
        print("Raw JSON:")
        print(composite_json)
        raise SystemExit(1)

    print(f"\n{'=' * 60}")
    print(f"SYNTHESIS RESULT — {TICKER} @ {TRADE_DATE}")
    print(f"{'=' * 60}")
    print(f"mu_final      = {composite.mu_final:+.4f}")
    print(f"sigma_final   = {composite.sigma_final:.4f}")
    print(f"conviction    = {composite.conviction_final:.4f}")
    print(f"rationale     = {composite.weighting_rationale}")
    print(f"\ncomposite thesis (long_term):  {composite.composite_thesis.long_term[:200]}...")
    print(f"composite thesis (medium_term): {composite.composite_thesis.medium_term[:200]}...")
    print(f"composite thesis (short_term): {composite.composite_thesis.short_term[:200]}...")
    print(f"conflicts     = {len(composite.cross_signal_conflicts)}")
    print(f"source_reports= {composite.source_reports}")
    print(f"{'=' * 60}")
    print("\nFull CompositeSignal:")
    print(json.dumps(composite.model_dump(mode="json"), indent=2))
    print(f"{'=' * 60}")