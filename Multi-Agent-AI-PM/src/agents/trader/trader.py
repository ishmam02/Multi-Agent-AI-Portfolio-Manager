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
import statistics
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
    "news_report": AgentType.NEWS,
}

AGENT_TYPE_KEYS = {
    AgentType.FUNDAMENTAL: "fundamental",
    AgentType.MARKET: "market",
    AgentType.NEWS: "news",
}

# Research-depth config: samples, critique rounds, conflict threshold
DEPTH_CONFIG = {
    "shallow": {"samples": 2, "critique_rounds": 0, "conflict_threshold": 0.25},
    "medium": {"samples": 3, "critique_rounds": 1, "conflict_threshold": 0.20},
    "deep": {"samples": 5, "critique_rounds": 2, "conflict_threshold": 0.15},
}

_CONSENSUS_STD_THRESHOLD = 0.15
_REFINE_CAP = 0.10


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
            "news": float(w.get("news", 0.0)),
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
    eq = 1.0 / 3.0
    return {h: {"fundamental": eq, "market": eq, "news": eq} for h in horizons}


def _compute_base_weights(reports: Dict[AgentType, ResearchReport], horizons: tuple) -> dict:
    """
    Heuristic base weights proportional to each analyst's average conviction.
    Fully deterministic fallback when LLM consensus is unstable.
    """
    base = {}
    for horizon in horizons:
        conv = {}
        for agent_type, report in reports.items():
            key = AGENT_TYPE_KEYS.get(agent_type, agent_type.value)
            conv[key] = max(0.0, getattr(report.conviction, horizon, 0.0))
        total = sum(conv.values())
        if total > 0:
            base[horizon] = {k: v / total for k, v in conv.items()}
        else:
            eq = 1.0 / 3.0
            base[horizon] = {"fundamental": eq, "market": eq, "news": eq}
    return base


def _build_horizon_weights_example(horizons: tuple) -> str:
    """Build the example weights JSON block for the selected horizons."""
    eq = '{"fundamental": 0.33, "market": 0.33, "news": 0.34}'
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


def _reports_json(
    reports: Dict[AgentType, ResearchReport],
    horizons: tuple,
    confidence_rationale: dict | None = None,
) -> list:
    """Serialize reports into the JSON structure used by LLM prompts."""
    out = []
    for agent_type, report in reports.items():
        entry: dict = {"analyst": agent_type.value}
        for field, src in (
            ("mu", report.mu),
            ("sigma", report.sigma_contribution),
            ("conviction", report.conviction),
        ):
            entry[field] = {h: getattr(src, h, 0.0) for h in horizons}
        entry["mu_trace_id"] = {h: getattr(report.mu_trace_id, h, "") for h in horizons}
        entry["sigma_trace_id"] = {h: getattr(report.sigma_trace_id, h, "") for h in horizons}
        entry["key_catalysts"] = [
            {
                "catalyst": c.catalyst,
                "term": c.term,
                "metric_name": c.metric_name,
                "trace_id": c.computation_trace_id,
            }
            for c in report.key_catalysts[:3]
        ]
        entry["key_risks"] = [
            {
                "risk": r.risk,
                "term": r.term,
                "metric_name": r.metric_name,
                "trace_id": r.computation_trace_id,
            }
            for r in report.key_risks[:3]
        ]
        _trace_to_metric = {
            m.computation_trace_id: m.metric_name
            for m in report.computed_metrics
            if m.computation_trace_id
        }
        entry["available_traces"] = [
            {
                "trace_id": t.trace_id,
                "metric_name": _trace_to_metric.get(t.trace_id, "unknown"),
                "description": t.code[:80] + "..." if len(t.code) > 80 else t.code,
            }
            for t in report.computation_traces[:5]
        ]
        # Inject news-analyst confidence rationale so the LLM can see *why*
        # conviction is high/low when assigning weights.
        if agent_type == AgentType.NEWS and confidence_rationale:
            entry["confidence_rationale"] = {
                h: confidence_rationale.get(h, "") for h in horizons
            }
        out.append(entry)
    return out


# ── Phase 1: Self-Consistency Weight Sampling ───────────────────────────────


def _sample_weights(
    llm, reports_json: list, horizons: tuple, samples: int
) -> List[dict]:
    """Run Phase 1 weight assignment N times and collect results."""
    horizon_names = _build_horizon_list(horizons)
    weights_example = _build_horizon_weights_example(horizons)
    blend_example = _build_horizon_blend_example(horizons)

    phase1_prompt = f"""You are a senior portfolio manager synthesizing multi-analyst research into a unified signal.

Your task is to assign per-analyst, per-horizon weights that sum to 1.0 for each horizon.

Available analysts and their signals:
{{reports_json}}

Instructions:
1. Assign weights for each horizon ({horizon_names}).
2. Each horizon's weights must sum to exactly 1.0.
3. Consider all available analyst types (fundamental, market, news). Give more weight to analysts with higher conviction and clearer causal chains.
4. If an analyst's confidence_rationale mentions weak evidence, conflicting signals, thin data, or ungrounded estimates, PENALIZE that analyst's weight for that horizon (reduce by 0.10-0.30). Conversely, if the rationale describes strong, consistent evidence with price-data backing, REWARD that analyst's weight (increase by 0.05-0.15).
5. Provide a concise rationale (2-3 sentences) explaining your weighting logic. When referencing any metric, catalyst, or risk, cite it inline using the exact format [metric_name | trace:<computation_trace_id>].
6. If an analyst type is missing, assign it weight 0.0 and redistribute.
7. Also provide horizon_blend_weights (how much to weight each selected horizon when blending into a single final signal). These must also sum to 1.0.

Respond with a single JSON object in this exact structure:
{{
  "rationale": "...",
  "weights": {{
{weights_example}
  }},
  "horizon_blend_weights": {{{blend_example}}}
}}

Output ONLY the JSON object. No markdown fences, no extra text."""

    collected: List[dict] = []
    for _ in range(samples):
        result = llm.invoke([
            SystemMessage(content="You are a senior portfolio manager synthesizing multi-analyst research."),
            HumanMessage(content=phase1_prompt.format(reports_json=json.dumps(reports_json, indent=2))),
        ])
        parsed = _extract_json_block(result.content)
        if isinstance(parsed, dict) and "weights" in parsed:
            collected.append(parsed)
    return collected


def _consensus_weights(
    samples: List[dict],
    base_weights: dict,
    horizons: tuple,
) -> tuple:
    """
    Compute median weights across self-consistency samples.
    Returns (weights_dict, blend_dict, rationale, used_fallback).
    Falls back to base_weights if sample std > threshold.
    """
    if not samples:
        return base_weights, {h: round(1.0 / len(horizons), 2) for h in horizons}, "Fallback to base weights — no LLM samples succeeded.", True

    # Collect per-horizon, per-analyst weight lists
    analyst_keys = ["fundamental", "market", "news"]
    weight_samples: Dict[str, Dict[str, List[float]]] = {h: {k: [] for k in analyst_keys} for h in horizons}
    blend_samples: Dict[str, List[float]] = {h: [] for h in horizons}
    rationales: List[str] = []

    for s in samples:
        if not isinstance(s, dict):
            continue
        rationales.append(s.get("rationale", ""))
        raw_weights = s.get("weights", {})
        if isinstance(raw_weights, dict):
            for h in horizons:
                wh = raw_weights.get(h, {})
                if isinstance(wh, dict):
                    for k in analyst_keys:
                        try:
                            weight_samples[h][k].append(float(wh.get(k, 0.0)))
                        except (TypeError, ValueError):
                            pass
        raw_blend = s.get("horizon_blend_weights", {})
        if isinstance(raw_blend, dict):
            for h in horizons:
                try:
                    blend_samples[h].append(float(raw_blend.get(h, 0.0)))
                except (TypeError, ValueError):
                    pass

    # Median weights
    median_weights: dict = {}
    max_std = 0.0
    for h in horizons:
        median_weights[h] = {}
        for k in analyst_keys:
            vals = weight_samples[h][k]
            if vals:
                median_weights[h][k] = statistics.median(vals)
                if len(vals) > 1:
                    max_std = max(max_std, statistics.stdev(vals))
            else:
                median_weights[h][k] = 0.0

    median_weights = _normalize_weights(median_weights, horizons)

    # Median blend
    median_blend: dict = {}
    for h in horizons:
        vals = blend_samples[h]
        median_blend[h] = statistics.median(vals) if vals else round(1.0 / len(horizons), 2)
    total_blend = sum(median_blend.values())
    if total_blend > 0:
        median_blend = {k: v / total_blend for k, v in median_blend.items()}
    else:
        median_blend = {h: round(1.0 / len(horizons), 2) for h in horizons}

    # Fallback check
    used_fallback = max_std > _CONSENSUS_STD_THRESHOLD
    if used_fallback:
        weights_out = base_weights
        blend_out = {h: round(1.0 / len(horizons), 2) for h in horizons}
        rationale = (
            f"LLM weight samples showed high variance (max std={max_std:.3f}); "
            f"falling back to conviction-proportional base weights. "
            f"Original rationale samples: {' | '.join(r for r in rationales if r)[:300]}"
        )
    else:
        weights_out = median_weights
        blend_out = median_blend
        rationale = "Consensus LLM weights. " + " | ".join(r for r in rationales if r)[:300]

    return weights_out, blend_out, rationale, used_fallback


def _critique_weights(
    llm, consensus_weights: dict, reports_json: list, horizons: tuple
) -> str:
    """Debate round: ask a risk officer to critique the PM's weight assignment."""
    critique_prompt = f"""You are a senior risk officer reviewing a portfolio manager's analyst weight assignment.

Analyst signals:
{json.dumps(reports_json, indent=2)}

Assigned weights:
{json.dumps(consensus_weights, indent=2)}

Your task: identify biases in the PM's weighting.
- Did they over-weight the analyst with the most confident tone rather than the best evidence?
- Did they ignore a strong contrarian signal?
- Did they fail to penalize an analyst whose conviction was low or whose causal chain was weak?
- Did they fail to account for the confidence_rationale?  If an analyst admits weak or conflicting evidence in its confidence_rationale, the PM should have reduced its weight.
- Are any horizon weights dangerously concentrated (>0.70 on one analyst)?

Respond with a concise critique (2-4 sentences). Be specific about which analyst/horizon needs adjustment and why.
Output ONLY plain text. No markdown fences, no JSON."""

    result = llm.invoke([
        SystemMessage(content="You are a senior risk officer reviewing analyst weight assignments."),
        HumanMessage(content=critique_prompt),
    ])
    return str(result.content or "").strip()


def _refine_weights(
    llm,
    consensus_weights: dict,
    critique: str,
    reports_json: list,
    horizons: tuple,
) -> tuple:
    """
    Apply capped adjustments based on critique.
    Returns (refined_weights, refined_blend, rationale).
    """
    horizon_names = _build_horizon_list(horizons)
    weights_example = _build_horizon_weights_example(horizons)
    blend_example = _build_horizon_blend_example(horizons)

    refine_prompt = f"""You are a senior portfolio manager refining your analyst weight assignment after receiving risk-officer feedback.

Analyst signals:
{json.dumps(reports_json, indent=2)}

Your current weights:
{json.dumps(consensus_weights, indent=2)}

Risk officer critique:
{critique}

Instructions:
1. Adjust weights for each horizon ({horizon_names}) based on the critique.
2. Each horizon's analyst weights must still sum to exactly 1.0.
3. You may adjust each analyst's weight by AT MOST ±{_REFINE_CAP:.2f} from the current value. Do not make large swings.
4. Provide a concise rationale (1-2 sentences) for the adjustments.
5. Also provide horizon_blend_weights summing to 1.0.

Respond with a single JSON object in this exact structure:
{{
  "rationale": "...",
  "weights": {{
{weights_example}
  }},
  "horizon_blend_weights": {{{blend_example}}}
}}

Output ONLY the JSON object. No markdown fences, no extra text."""

    result = llm.invoke([
        SystemMessage(content="You are a senior portfolio manager refining analyst weights."),
        HumanMessage(content=refine_prompt),
    ])
    parsed = _extract_json_block(result.content)

    if isinstance(parsed, dict) and "weights" in parsed:
        raw_weights = parsed.get("weights")
        raw_blend = parsed.get("horizon_blend_weights")
        rationale = parsed.get("rationale", "Refined weights after critique.")
        if isinstance(raw_weights, dict):
            weights = _normalize_weights(raw_weights, horizons)
        else:
            weights = consensus_weights
        if isinstance(raw_blend, dict):
            blend_vals = {h: float(raw_blend.get(h, round(1.0 / len(horizons), 2))) for h in horizons}
            total = sum(blend_vals.values())
            blend = {k: v / total for k, v in blend_vals.items()} if total > 0 else {h: round(1.0 / len(horizons), 2) for h in horizons}
        else:
            blend = {h: round(1.0 / len(horizons), 2) for h in horizons}
        return weights, blend, rationale

    return consensus_weights, {h: round(1.0 / len(horizons), 2) for h in horizons}, "Refinement failed; keeping consensus."


# ── Phase 2: Deterministic Conflict Detection + LLM Labelling ────────────────


def _detect_conflicts(
    reports: Dict[AgentType, ResearchReport],
    threshold: float,
    horizons: tuple,
) -> List[dict]:
    """
    Detect cross-analyst conflicts using deterministic rules.
    Returns list of conflict stubs (no description yet).
    """
    analyst_list = list(reports.keys())
    conflicts = []
    for i, a_type in enumerate(analyst_list):
        for b_type in analyst_list[i + 1:]:
            for h in horizons:
                mu_a = getattr(reports[a_type].mu, h, 0.0)
                mu_b = getattr(reports[b_type].mu, h, 0.0)
                sigma_a = getattr(reports[a_type].sigma_contribution, h, 0.01)
                sigma_b = getattr(reports[b_type].sigma_contribution, h, 0.01)

                # Rule 1: sign disagreement + magnitude gap
                sign_disagree = (mu_a >= 0) != (mu_b >= 0)
                denom = max(abs(mu_a), abs(mu_b), 0.001)
                relative_gap = abs(mu_a - mu_b) / denom

                # Rule 2: sigma divergence (>2x ratio)
                sigma_ratio = max(sigma_a, sigma_b) / max(min(sigma_a, sigma_b), 0.001)
                sigma_divergence = sigma_ratio > 2.0

                if (sign_disagree and relative_gap > threshold) or sigma_divergence:
                    conflicts.append({
                        "analyst_a": AGENT_TYPE_KEYS.get(a_type, a_type.value),
                        "analyst_b": AGENT_TYPE_KEYS.get(b_type, b_type.value),
                        "horizon": h,
                        "mu_a": mu_a,
                        "mu_b": mu_b,
                        "sigma_a": sigma_a,
                        "sigma_b": sigma_b,
                    })
    return conflicts


def _label_conflicts(
    llm, conflict_stubs: List[dict], reports_json: list, horizons: tuple
) -> List[CrossSignalConflict]:
    """Ask LLM to write conflict descriptions and resolution status for flagged conflicts."""
    if not conflict_stubs:
        return []

    label_prompt = f"""You are a risk analyst reviewing flagged contradictions between analyst signals.

Analyst signals:
{json.dumps(reports_json, indent=2)}

Flagged conflicts (detected by deterministic rules):
{json.dumps(conflict_stubs, indent=2)}

For each flagged conflict, write:
- conflict_description (1 sentence). When referencing any metric, catalyst, or risk, cite it inline using [metric_name | trace:<computation_trace_id>].
- resolution_status: "resolved" if one analyst's reasoning clearly outweighs the other, else "unresolved"
- conviction_penalty: a negative float between -1.0 and 0.0 (e.g., -0.15). More severe = larger magnitude.

IMPORTANT: Only report genuinely distinct contradictions. If the same core disagreement appears in multiple forms, report it ONCE with the most severe penalty. Do not double-count.

Respond with a single JSON object:
{{
  "conflicts": [
    {{
      "analyst_a": "fundamental",
      "analyst_b": "market",
      "horizon": "{horizons[-1]}",
      "conflict_description": "...",
      "resolution_status": "unresolved",
      "conviction_penalty": -0.12
    }}
  ]
}}

If no contradictions exist, return {{"conflicts": []}}.
Output ONLY the JSON object. No markdown fences, no extra text."""

    result = llm.invoke([
        SystemMessage(content="You are a risk analyst reviewing multi-analyst research for contradictions."),
        HumanMessage(content=label_prompt),
    ])
    parsed = _extract_json_block(result.content)

    conflicts: List[CrossSignalConflict] = []
    if isinstance(parsed, dict):
        raw_conflicts = parsed.get("conflicts", [])
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
    return conflicts


# ── Phase 2.5: Optional Composite Thesis ───────────────────────────────────────


def _generate_thesis(
    llm, reports_json: list, weights_dict: dict, conflicts: List[CrossSignalConflict], horizons: tuple
) -> dict:
    """Generate unified investment thesis per horizon. Purely narrative."""
    horizon_names = _build_horizon_list(horizons)
    thesis_prompt = f"""You are a senior portfolio manager writing a unified investment thesis by blending multi-analyst research.

Analyst theses:
{json.dumps(reports_json, indent=2)}

Assigned analyst weights:
{json.dumps(weights_dict, indent=2)}

Identified conflicts:
{json.dumps([{"analyst_a": c.analyst_a.value, "analyst_b": c.analyst_b.value, "horizon": c.horizon, "description": c.conflict_description} for c in conflicts], indent=2)}

Your task:
1. Write a single unified investment thesis for each horizon ({horizon_names}).
2. Blend the analyst narratives according to their assigned weights. Higher-weighted analysts should dominate the narrative.
3. When referencing any metric, catalyst, or risk, cite it inline using [metric_name | trace:<computation_trace_id>].
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

    result = llm.invoke([
        SystemMessage(content="You are a senior portfolio manager writing a unified investment thesis."),
        HumanMessage(content=thesis_prompt),
    ])
    parsed = _extract_json_block(result.content)
    _composite_thesis = {h: "" for h in ("long_term", "medium_term", "short_term")}
    if isinstance(parsed, dict):
        raw = parsed.get("composite_thesis", {})
        if isinstance(raw, dict):
            for h in ("long_term", "medium_term", "short_term"):
                _composite_thesis[h] = str(raw.get(h, ""))[:500]
    return _composite_thesis


# ── Synthesis Agent Factory ──────────────────────────────────────────────────


def create_synthesis_agent(
    llm,
    horizons=("long_term", "medium_term", "short_term"),
    research_depth="medium",
    generate_thesis=True,
):
    """Create a synthesis node function for the outer AgentState graph.

    Parameters
    ----------
    llm            : LangChain chat model for Phase 1 (weights) and Phase 2 (conflicts)
    horizons       : Tuple of horizon strings to include in synthesis.
    research_depth : "shallow" | "medium" | "deep" — controls self-consistency samples,
                     critique rounds, and conflict-detection sensitivity.
    generate_thesis: If False, skip Phase 2.5 (thesis generation) for speed/determinism.
    """
    depth_cfg = DEPTH_CONFIG.get(research_depth, DEPTH_CONFIG["medium"])
    samples = max(2, depth_cfg["samples"])  # enforce minimum > 1
    critique_rounds = depth_cfg["critique_rounds"]
    conflict_threshold = depth_cfg["conflict_threshold"]

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
            blend_defaults = {h: round(1.0 / len(horizons), 2) for h in horizons}
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

        # Serialize reports once for all LLM prompts
        confidence_rationale = state.get("confidence_rationale") or {}
        reports_json = _reports_json(reports, horizons, confidence_rationale)

        # ── Phase 1: Self-Consistency Weight Assignment ─────────────────
        base_weights = _compute_base_weights(reports, horizons)
        phase1_samples = _sample_weights(llm, reports_json, horizons, samples)
        weights_dict, horizon_blend, rationale, used_fallback = _consensus_weights(
            phase1_samples, base_weights, horizons
        )

        # Debate / critique rounds
        for _ in range(critique_rounds):
            critique = _critique_weights(llm, weights_dict, reports_json, horizons)
            weights_dict, horizon_blend, rationale = _refine_weights(
                llm, weights_dict, critique, reports_json, horizons
            )

        # ── Phase 2: Deterministic Conflict Detection + LLM Labelling ───
        conflict_stubs = _detect_conflicts(reports, conflict_threshold, horizons)
        conflicts = _label_conflicts(llm, conflict_stubs, reports_json, horizons)

        unresolved_penalty = 0.0
        for c in conflicts:
            if c.resolution_status != "resolved":
                unresolved_penalty += abs(c.conviction_penalty)
        unresolved_penalty = min(unresolved_penalty, 1.0)

        # ── Phase 2.5: Optional Composite Thesis ─────────────────────────
        _composite_thesis = {h: "" for h in _all_horizons}
        if generate_thesis:
            _composite_thesis = _generate_thesis(
                llm, reports_json, weights_dict, conflicts, horizons
            )

        # ── Phase 3: Composite Signal Math (deterministic) ───────────────
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
        if used_fallback:
            provenance_parts.append("consensus_fallback")
        computation_provenance = ", ".join(provenance_parts)

        # ── Build CompositeSignal ─────────────────────────────────────────────
        def _aw(weights_dict: dict, h: str) -> AnalystWeights:
            w = weights_dict.get(h, {})
            return AnalystWeights(
                fundamental=float(w.get("fundamental", 0.0)),
                market=float(w.get("market", 0.0)),
                news=float(w.get("news", 0.0)),
            )

        analyst_weights = HorizonWeights(
            long_term=_aw(weights_dict, "long_term"),
            medium_term=_aw(weights_dict, "medium_term"),
            short_term=_aw(weights_dict, "short_term"),
        )

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
            f"conviction={conviction_final:.4f}, conflicts={len(conflicts)}, "
            f"fallback={'yes' if used_fallback else 'no'}."
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
    _HORIZONS = ("long_term", "medium_term", "short_term")
    market_node = create_market_analyst(
        reasoning_llm, market_code_agent, RESEARCH_DEPTH, active_horizons=_HORIZONS
    )
    fundamentals_node = create_fundamentals_analyst(
        reasoning_llm, fundamentals_code_agent, RESEARCH_DEPTH, active_horizons=_HORIZONS
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
    synthesis_node = create_synthesis_agent(
        reasoning_llm, research_depth=RESEARCH_DEPTH
    )

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
