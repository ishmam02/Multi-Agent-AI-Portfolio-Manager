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
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.utils.schemas import (
    AgentType,
    AnalystWeights,
    CompositeSignal,
    CrossSignalConflict,
    HorizonValues,
    HorizonWeights,
    ResearchReport,
)


# ── Constants ────────────────────────────────────────────────────────────────

AGENT_TYPE_MAP = {
    "market_report": AgentType.MARKET,
    "news_report": AgentType.NEWS,
    "fundamentals_report": AgentType.FUNDAMENTAL,
}

AGENT_TYPE_KEYS = {
    AgentType.FUNDAMENTAL: "fundamental",
    AgentType.MARKET: "market",
    AgentType.NEWS: "news",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_json_block(text: str) -> dict:
    """Extract the first JSON object or array from text."""
    stripped = text.strip()
    # Direct parse
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Fenced code blocks
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    # Forward scan for { or [
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
            if isinstance(d, (dict, list)):
                return d
        except (json.JSONDecodeError, ValueError):
            continue

    return {}


def _normalize_weights(weights: dict) -> dict:
    """Normalize per-horizon analyst weights to sum to 1.0 (±1e-6)."""
    normalized = {}
    for horizon, w in weights.items():
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


def _default_equal_weights() -> dict:
    """Return equal weights for all analysts across all horizons."""
    return {
        "long_term": {"fundamental": 0.34, "market": 0.33, "news": 0.33},
        "medium_term": {"fundamental": 0.34, "market": 0.33, "news": 0.33},
        "short_term": {"fundamental": 0.34, "market": 0.33, "news": 0.33},
    }


# ── Synthesis Agent Factory ──────────────────────────────────────────────────


def create_synthesis_agent(llm, code_agent, memory):
    """Create a synthesis node function for the outer AgentState graph.

    Parameters
    ----------
    llm        : LangChain chat model for Phase 1 (weights) and Phase 2 (conflicts)
    code_agent : CodeValidationAgent for Phase 3 math (or None to compute inline)
    memory     : FinancialSituationMemory for contextual weighting
    """

    def synthesis_node(state, name):
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", datetime.now().strftime("%Y-%m-%d"))

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
            composite = CompositeSignal(
                ticker=company_name,
                timestamp=datetime.now(),
                mu_composite=HorizonValues(long_term=0.0, medium_term=0.0, short_term=0.0),
                sigma_composite=HorizonValues(long_term=0.01, medium_term=0.01, short_term=0.01),
                mu_final=0.0,
                sigma_final=0.01,
                conviction_final=0.0,
                analyst_weights=HorizonWeights(),
                horizon_blend_weights=HorizonValues(long_term=0.33, medium_term=0.33, short_term=0.34),
                weighting_rationale="No analyst reports available; defaulting to neutral signal.",
                cross_signal_conflicts=[],
                unresolved_penalty=0.0,
                source_reports=[],
                computation_provenance="default_fallback",
            )
            return {
                "messages": [HumanMessage(content="No reports available — neutral signal.")],
                "composite_signal": composite.model_dump_json(),
                "sender": name,
            }

        # ── Step B: Build situation string for memory ─────────────────────────
        situation_parts = []
        for agent_type, report in reports.items():
            thesis = report.investment_thesis
            situation_parts.append(
                f"{agent_type.value.upper()} THESIS:\n"
                f"  Long: {thesis.long_term}\n"
                f"  Medium: {thesis.medium_term}\n"
                f"  Short: {thesis.short_term}"
            )
            if report.key_catalysts:
                cats = "\n  ".join([c.catalyst for c in report.key_catalysts[:3]])
                situation_parts.append(f"  Catalysts: {cats}")
            if report.key_risks:
                risks = "\n  ".join([r.risk for r in report.key_risks[:3]])
                situation_parts.append(f"  Risks: {risks}")
        curr_situation = "\n\n".join(situation_parts)
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += f"Memory {i}: {rec['recommendation']}\n"

        # ── Phase 1: Dynamic Weight Assignment (LLM) ────────────────────────
        reports_json = []
        for agent_type, report in reports.items():
            reports_json.append(
                {
                    "analyst": agent_type.value,
                    "mu": {
                        "long_term": report.mu.long_term,
                        "medium_term": report.mu.medium_term,
                        "short_term": report.mu.short_term,
                    },
                    "sigma": {
                        "long_term": report.sigma_contribution.long_term,
                        "medium_term": report.sigma_contribution.medium_term,
                        "short_term": report.sigma_contribution.short_term,
                    },
                    "conviction": {
                        "long_term": report.conviction.long_term,
                        "medium_term": report.conviction.medium_term,
                        "short_term": report.conviction.short_term,
                    },
                    "thesis": {
                        "long_term": report.investment_thesis.long_term,
                        "medium_term": report.investment_thesis.medium_term,
                        "short_term": report.investment_thesis.short_term,
                    },
                }
            )

        phase1_prompt = f"""You are a senior portfolio manager synthesizing multi-analyst research into a unified signal.

Your task is to assign per-analyst, per-horizon weights that sum to 1.0 for each horizon.

Available analysts and their signals:
{json.dumps(reports_json, indent=2)}

Past context:
{past_memory_str or "(none)"}

Instructions:
1. Assign weights for each horizon (long_term, medium_term, short_term).
2. Each horizon's weights must sum to exactly 1.0.
3. Provide a concise rationale (2-3 sentences) explaining your weighting logic.
4. If an analyst type is missing, assign it weight 0.0 and redistribute.
5. Also provide horizon_blend_weights (how much to weight long/medium/short when blending into a single final signal). These three must also sum to 1.0.

Respond with a single JSON object in this exact structure:
{{
  "rationale": "...",
  "weights": {{
    "long_term": {{"fundamental": 0.34, "market": 0.33, "news": 0.33}},
    "medium_term": {{...}},
    "short_term": {{...}}
  }},
  "horizon_blend_weights": {{"long_term": 0.3, "medium_term": 0.4, "short_term": 0.3}}
}}

Output ONLY the JSON object. No markdown fences, no extra text."""

        phase1_result = llm.invoke([SystemMessage(content=phase1_prompt)])
        phase1_json = _extract_json_block(phase1_result.content)

        weights_dict = _default_equal_weights()
        horizon_blend = {"long_term": 0.33, "medium_term": 0.33, "short_term": 0.34}
        rationale = "Fallback to equal weights due to parse failure."

        if isinstance(phase1_json, dict):
            rationale = phase1_json.get("rationale", rationale)
            raw_weights = phase1_json.get("weights")
            if isinstance(raw_weights, dict):
                weights_dict = _normalize_weights(raw_weights)
            raw_blend = phase1_json.get("horizon_blend_weights")
            if isinstance(raw_blend, dict):
                blend_vals = {
                    "long_term": float(raw_blend.get("long_term", 0.33)),
                    "medium_term": float(raw_blend.get("medium_term", 0.33)),
                    "short_term": float(raw_blend.get("short_term", 0.34)),
                }
                total = sum(blend_vals.values())
                if total > 0:
                    horizon_blend = {k: v / total for k, v in blend_vals.items()}
                else:
                    horizon_blend = {"long_term": 0.33, "medium_term": 0.33, "short_term": 0.34}

        # ── Phase 2: Cross-Signal Consistency Check (LLM) ───────────────────
        phase2_prompt = f"""You are a risk analyst reviewing multi-analyst research for contradictions.

Analyst signals:
{json.dumps(reports_json, indent=2)}

Your task:
1. Identify contradictions between analysts (e.g., fundamental says bullish, market says bearish).
2. For each contradiction, specify:
   - analyst_a and analyst_b (use exactly: fundamental, market, news)
   - horizon (long_term, medium_term, or short_term)
   - conflict_description (1 sentence)
   - resolution_status: "resolved" if one analyst's reasoning clearly outweighs the other, else "unresolved"
   - conviction_penalty: a negative float between -1.0 and 0.0 (e.g., -0.15). More severe = larger magnitude.

Respond with a single JSON object:
{{
  "conflicts": [
    {{
      "analyst_a": "fundamental",
      "analyst_b": "market",
      "horizon": "short_term",
      "conflict_description": "Fundamental DCF implies 20% upside while market RSI shows overbought exhaustion.",
      "resolution_status": "unresolved",
      "conviction_penalty": -0.12
    }}
  ]
}}

If no contradictions exist, return {{"conflicts": []}}.
Output ONLY the JSON object. No markdown fences, no extra text."""

        phase2_result = llm.invoke([SystemMessage(content=phase2_prompt)])
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

        # ── Phase 3: Composite Signal Math (deterministic) ──────────────────
        # Compute per-horizon composite mu and sigma using the assigned weights.
        mu_comp = {}
        sigma_comp = {}
        horizons = ["long_term", "medium_term", "short_term"]

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
                sigma_sq += (weight ** 2) * (sigma_val ** 2)
            mu_comp[horizon] = max(-1.0, min(1.0, mu_h))
            sigma_comp[horizon] = max(0.001, math.sqrt(sigma_sq))

        # Blend horizons into final mu / sigma
        mu_final = sum(horizon_blend[h] * mu_comp[h] for h in horizons)
        sigma_final = math.sqrt(
            sum((horizon_blend[h] ** 2) * (sigma_comp[h] ** 2) for h in horizons)
        )
        sigma_final = max(0.001, sigma_final)

        # Conviction: scaled by final mu magnitude, reduced by unresolved penalties
        raw_conviction = abs(mu_final)
        conviction_final = max(0.0, min(1.0, raw_conviction - unresolved_penalty))

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
        analyst_weights = HorizonWeights(
            long_term=AnalystWeights(**weights_dict.get("long_term", {})),
            medium_term=AnalystWeights(**weights_dict.get("medium_term", {})),
            short_term=AnalystWeights(**weights_dict.get("short_term", {})),
        )

        composite = CompositeSignal(
            ticker=company_name,
            timestamp=datetime.now(),
            mu_composite=HorizonValues(**mu_comp),
            sigma_composite=HorizonValues(**sigma_comp),
            mu_final=mu_final,
            sigma_final=sigma_final,
            conviction_final=conviction_final,
            analyst_weights=analyst_weights,
            horizon_blend_weights=HorizonValues(**horizon_blend),
            weighting_rationale=rationale,
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
