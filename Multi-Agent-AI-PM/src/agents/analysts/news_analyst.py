"""
News Analyst — LLM-only tool-calling node with multi-pass research depth.

Strategically searches for important news events across the horizon lookback
(earnings calls, macro events, sector shifts) rather than dumping all articles
at once.  Supports shallow/medium/deep research depth for iterative refinement
and builds a full citation chain from the investment thesis.
"""

from __future__ import annotations

import json

from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.prompts import load_prompt
from src.agents.utils.core_stock_tools import get_stock_data as get_stock_data_tool
from src.agents.utils.news_data_tools import get_global_news as get_global_news_tool
from src.agents.utils.news_data_tools import (
    get_insider_transactions as get_insider_transactions_tool,
)
from src.agents.utils.news_data_tools import get_news as get_news_tool
from src.agents.utils.schemas import (
    AgentType,
    Catalyst,
    Citation,
    HorizonThesis,
    HorizonValues,
    ResearchReport,
    Risk,
)

# ── Module-level cache for canonical fetches (shared across parallel runs) ─────
_CANONICAL_CACHE: dict[tuple[str, str, str], str] = {}

# ── Structured output schema for LLM ───────────────────────────────────────────

from pydantic import BaseModel


class _NewsAnalystLLMOutput(BaseModel):
    """Schema for the LLM's structured output after tool calling."""

    mu: dict[str, float]
    sigma_contribution: dict[str, float]
    conviction: dict[str, float]
    investment_thesis: dict[str, str]
    confidence_rationale: dict[str, str]  # per-horizon: why is confidence high/low?
    citation_chain: list[dict]
    key_catalysts: list[dict]
    key_risks: list[dict]


# ── Load prompts ───────────────────────────────────────────────────────────────

_news_prompts = load_prompt("news_analyst")
SYSTEM_PROMPT: str = _news_prompts["SYSTEM_PROMPT"]  # type: ignore[assignment]
REFINE_PROMPT: str = _news_prompts.get("REFINE_PROMPT", "")  # type: ignore[assignment]


# ── Research depth → pass count ────────────────────────────────────────────────

_DEPTH_PASSES = {"shallow": 1, "medium": 2, "deep": 3}


# ── Node factory ───────────────────────────────────────────────────────────────


def create_news_analyst(
    llm,
    research_depth: str = "medium",
    horizons: tuple[str, ...] = ("long_term", "medium_term", "short_term"),
):
    """Create a news analyst node with multi-pass research and citation tracking.

    Parameters
    ----------
    llm            : LangChain chat model — tools are bound at invocation time.
    research_depth : "shallow" (1 pass) | "medium" (2 passes) | "deep" (3 passes).
    horizons       : Which horizons to analyse (default: all three).
    """

    tools = [
        get_news_tool,
        get_global_news_tool,
        get_stock_data_tool,
        get_insider_transactions_tool,
    ]
    max_passes = _DEPTH_PASSES.get(research_depth, 2)
    horizons_str = ", ".join(horizons)
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    def _tool_loop(messages: list, system_content: str) -> list:
        """Run the tool-calling loop: LLM calls tools, results fed back. Returns final messages."""
        chain = prompt.partial(system_message=system_content) | llm_with_tools

        response = chain.invoke({"messages": messages})
        tool_rounds = 0

        while (
            hasattr(response, "tool_calls") and response.tool_calls and tool_rounds < 5
        ):
            tool_rounds += 1
            messages.append(response)

            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                try:
                    if tool_name == "get_news":
                        result = get_news_tool.invoke(tool_args)
                    elif tool_name == "get_global_news":
                        result = get_global_news_tool.invoke(tool_args)
                    elif tool_name == "get_stock_data":
                        result = get_stock_data_tool.invoke(tool_args)
                    elif tool_name == "get_insider_transactions":
                        result = get_insider_transactions_tool.invoke(tool_args)
                    else:
                        result = f"Unknown tool: {tool_name}"
                except Exception as exc:
                    result = f"Tool error: {exc}"
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc.get("id", ""))
                )

            response = chain.invoke({"messages": messages})

        messages.append(response)
        return messages

    def _extract_structured_output(messages: list, system_content: str) -> dict | None:
        """Call the LLM with structured output to extract a validated dict.

        Falls back to text parsing if structured output is unavailable or fails.
        """
        try:
            structured_llm = llm.with_structured_output(_NewsAnalystLLMOutput)
            chain = prompt.partial(system_message=system_content) | structured_llm
            output: _NewsAnalystLLMOutput = chain.invoke({"messages": messages})
            return output.model_dump()
        except Exception:
            # Fallback: parse JSON from the last assistant message
            last = messages[-1]
            text = last.content if hasattr(last, "content") else str(last)
            return _parse_json_from_text(text)

    _CANONICAL_WINDOWS = [
        # (label, lookback_start_days, lookback_end_days, recency_weight)
        # recency_weight = 1 (older) to 3 (most recent).  More pages for higher weight.
        ("recent_sentiment", 14, 0, 3),
        ("two_week_ago", 28, 14, 2),
        ("medium_lookback", 60, 30, 2),
        ("earlier_medium", 120, 90, 1),
        ("long_baseline", 380, 350, 1),
        ("two_year_baseline", 750, 720, 1),
    ]

    def _prefetch_canonical(ticker: str, trade_date: str) -> list:
        """Fetch all canonical news windows for the ticker+date.

        The same ticker+date always yields the same windows, anchoring
        sigma and reducing run-to-run variance.  Recency weighting gives
        more coverage to recent windows and less to older ones.
        """
        td = datetime.strptime(trade_date, "%Y-%m-%d")
        prefetched: list = []
        for label, lookback_start, lookback_end, weight in _CANONICAL_WINDOWS:
            start = (td - timedelta(days=lookback_start)).strftime("%Y-%m-%d")
            end = (td - timedelta(days=lookback_end)).strftime("%Y-%m-%d")
            cache_key = (ticker, start, end)
            cached = _CANONICAL_CACHE.get(cache_key)
            if cached is not None:
                news = cached
            else:
                try:
                    news = get_news_tool.invoke(
                        {"ticker": ticker, "start_date": start, "end_date": end}
                    )
                except Exception as exc:
                    news = f"Error fetching {label}: {exc}"
                _CANONICAL_CACHE[cache_key] = news
            prefetched.append(
                SystemMessage(
                    content=(
                        f"=== CANONICAL WINDOW: {label} ({start} to {end}) [weight={weight}] ===\n"
                        f"{news}\n"
                        f"=== END {label} ==="
                    )
                )
            )
        return prefetched

    def news_analyst_node(state: dict) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        regime_context = state.get("regime_context", "")

        # ── Pre-fetch canonical windows (unified with recency weighting) ──────
        canonical_messages = _prefetch_canonical(ticker, trade_date)

        # ── Pass 1: initial research + preliminary thesis ──────────────────
        initial_system = SYSTEM_PROMPT.replace(
            "HORIZON_PLACEHOLDER", horizons_str
        ).replace("DEPTH_PLACEHOLDER", research_depth)

        if regime_context:
            initial_system = (
                f"MARKET REGIME CONTEXT: {regime_context}\n\n"
                "Use this regime context to calibrate your expectations. "
                "In bear markets or crashes, actively look for NEGATIVE signals and downside risks. "
                "Do NOT default to bullish assumptions.\n\n"
                + initial_system
            )

        messages = [
            HumanMessage(
                content=(
                    f"Analyze {ticker} as of {trade_date}.  Active horizons: {horizons_str}.\n"
                    f"Research depth: {research_depth} ({max_passes} pass(es)).\n"
                    f"Below are pre-fetched canonical news windows.  Use them as your baseline "
                    f"evidence before deciding what additional targeted searches to run."
                )
            )
        ]
        messages.extend(canonical_messages)
        messages = _tool_loop(messages, initial_system)
        current_thesis_json = _extract_structured_output(messages, initial_system)

        # ── Passes 2+: refinement loop ─────────────────────────────────────
        for p in range(2, max_passes + 1):
            refine_system = (
                f"{REFINE_PROMPT}\n\nActive horizons: {horizons_str}.\n"
                f"This is pass {p} of {max_passes}."
            )

            previous = (
                json.dumps(current_thesis_json, indent=2)
                if current_thesis_json
                else "(no thesis yet)"
            )

            messages.append(
                HumanMessage(
                    content=(
                        f"=== YOUR PREVIOUS THESIS (end of pass {p - 1}) ===\n{previous}\n\n"
                        f"=== REFINEMENT PASS {p} of {max_passes} ===\n"
                        f"Review your previous thesis.  Identify WHAT IS MISSING or WEAK:\n"
                        f"- Are there claims without article evidence?  Search for supporting articles.\n"
                        f"- Are conviction values well-calibrated or overconfident?\n"
                        f"- Did you miss any important events in the horizon lookback?\n"
                        f"- Are the citation_chain entries complete (claim → source → source_uuid)?\n\n"
                        f"Search for additional articles to fill gaps, then output your REFINED "
                        f"ResearchReport JSON with complete citation_chain."
                    )
                )
            )

            messages = _tool_loop(messages, refine_system)
            refined = _extract_structured_output(messages, refine_system)
            if refined:
                current_thesis_json = refined

        # ── Self-criticism / red-team pass ─────────────────────────────────
        # This pass ONLY critiques conviction calibration and sigma guesswork.
        # mu direction is LOCKED unless a factual error is found.
        if current_thesis_json:
            critique_system = (
                "You are a calibration reviewer. Your ONLY job is to check whether "
                "conviction and sigma_contribution are properly calibrated — NOT to "
                "second-guess the mu direction.\n\n"
                "Rules:\n"
                "1. DO NOT flip the sign of mu unless you find a clear factual error "
                "   in the evidence (e.g. a claim contradicts the source article).\n"
                "2. DO adjust conviction DOWN if the evidence is thin or conflicting.\n"
                "3. DO adjust sigma UP if the price evidence shows larger historical "
                "   moves than initially estimated, or DOWN if moves were smaller.\n"
                "4. DO adjust sigma DOWN if the evidence is very consistent across "
                "   multiple sources (low volatility of opinion = low sigma).\n\n"
                "If no calibration error is found, return the original JSON unchanged."
            )
            previous = json.dumps(current_thesis_json, indent=2)
            messages.append(
                HumanMessage(
                    content=(
                        f"=== CALIBRATION REVIEW ===\n{previous}\n\n"
                        "Review conviction and sigma calibration only. "
                        "If well-calibrated, return the original JSON unchanged."
                    )
                )
            )
            messages = _tool_loop(messages, critique_system)
            critiqued = _extract_structured_output(messages, critique_system)
            if critiqued:
                current_thesis_json = critiqued

        # ── Build final ResearchReport ──────────────────────────────────────
        report_json = _build_report(current_thesis_json, ticker, trade_date, horizons)
        confidence_rationale = (
            current_thesis_json.get("confidence_rationale", {})
            if current_thesis_json
            else {}
        )
        return {
            "news_report": report_json,
            "confidence_rationale": confidence_rationale,
        }

    return news_analyst_node


# ── JSON parsing ───────────────────────────────────────────────────────────────


def _parse_json_from_text(text: str) -> dict | None:
    """Extract the LLM's JSON object from its response. Returns None on failure."""
    import re

    decoder = json.JSONDecoder()
    stripped = text.strip()

    if stripped.startswith("{"):
        try:
            d = json.loads(stripped)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
        try:
            d, _ = decoder.raw_decode(stripped, 0)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            pass

    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        try:
            d = json.loads(m.group(1).strip())
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue

    best: dict | None = None
    best_size = -1
    for m in re.finditer(r"\{", text):
        try:
            d, _ = decoder.raw_decode(text, m.start())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict) and len(d) > best_size:
            best = d
            best_size = len(d)

    return best


# ── Report assembly ────────────────────────────────────────────────────────────


def _is_active_horizon(thesis: dict, key: str) -> bool:
    """Return True if the horizon has a non-empty thesis."""
    return bool(thesis.get(key, ""))


def _build_report(
    d: dict | None, ticker: str, trade_date: str, active_horizons: tuple[str, ...] = ()
) -> str:
    """Build a validated ResearchReport JSON string from the LLM's output dict."""
    if not d:
        return _zero_report(ticker, trade_date)

    try:
        mu = d.get("mu", {})
        sigma = d.get("sigma_contribution", d.get("sigma", {}))
        conviction = d.get("conviction", {})
        thesis = d.get("investment_thesis", {})

        def _hf(hv: dict, key: str) -> float:
            try:
                return float(hv.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0

        def _hs(hv: dict, key: str) -> str:
            v = hv.get(key, "")
            return str(v) if v else ""

        # Determine active horizons
        active = {
            "long_term": "long_term" in active_horizons or _is_active_horizon(thesis, "long_term"),
            "medium_term": "medium_term" in active_horizons or _is_active_horizon(thesis, "medium_term"),
            "short_term": "short_term" in active_horizons or _is_active_horizon(thesis, "short_term"),
        }

        def _coerce_sigma(val: float, key: str) -> float:
            if active.get(key, False):
                return max(val, 0.01)
            return val

        def _coerce_conviction(val: float, key: str) -> float:
            if active.get(key, False):
                return max(val, 0.05)
            return val

        # Build citation chain from LLM output
        citation_chain: list[Citation] = []
        for c in d.get("citation_chain", []) or []:
            if isinstance(c, dict):
                citation_chain.append(
                    Citation(
                        claim=str(c.get("claim", "")),
                        source=str(c.get("source", "")),
                        source_uuid=str(c.get("source_uuid", "")),
                    )
                )

        report = ResearchReport(
            ticker=ticker,
            agent_type=AgentType.NEWS,
            timestamp=datetime.strptime(trade_date, "%Y-%m-%d"),
            metrics_selected=[],
            mu=HorizonValues(
                long_term=_hf(mu, "long_term"),
                medium_term=_hf(mu, "medium_term"),
                short_term=_hf(mu, "short_term"),
            ),
            mu_trace_id=None,
            sigma_contribution=HorizonValues(
                long_term=_coerce_sigma(_hf(sigma, "long_term"), "long_term"),
                medium_term=_coerce_sigma(_hf(sigma, "medium_term"), "medium_term"),
                short_term=_coerce_sigma(_hf(sigma, "short_term"), "short_term"),
            ),
            sigma_trace_id=None,
            computed_metrics=[],
            investment_thesis=HorizonThesis(
                long_term=_hs(thesis, "long_term"),
                medium_term=_hs(thesis, "medium_term"),
                short_term=_hs(thesis, "short_term"),
            ),
            conviction=HorizonValues(
                long_term=_coerce_conviction(_hf(conviction, "long_term"), "long_term"),
                medium_term=_coerce_conviction(_hf(conviction, "medium_term"), "medium_term"),
                short_term=_coerce_conviction(_hf(conviction, "short_term"), "short_term"),
            ),
            key_catalysts=_coerce_catalysts(d.get("key_catalysts")),
            key_risks=_coerce_risks(d.get("key_risks")),
            source_uuids=[c.source_uuid for c in citation_chain if c.source_uuid],
            computation_traces=[],
            citation_chain=citation_chain,
            contributing_factors=[],
        )
        return report.model_dump_json()
    except Exception:
        return _zero_report(ticker, trade_date)


def _coerce_catalysts(raw: object) -> list[Catalyst]:
    out: list[Catalyst] = []
    for item in (raw or []) if isinstance(raw, list) else []:
        if isinstance(item, dict):
            out.append(
                Catalyst(
                    catalyst=str(item.get("catalyst", "")),
                    term=str(item.get("term", "long_term")),
                    claim=str(item.get("claim", "")),
                    source_uuid=str(item.get("source_uuid", "")),
                )
            )
    return out


def _coerce_risks(raw: object) -> list[Risk]:
    out: list[Risk] = []
    for item in (raw or []) if isinstance(raw, list) else []:
        if isinstance(item, dict):
            out.append(
                Risk(
                    risk=str(item.get("risk", "")),
                    term=str(item.get("term", "long_term")),
                    claim=str(item.get("claim", "")),
                    source_uuid=str(item.get("source_uuid", "")),
                )
            )
    return out


def _zero_report(ticker: str, trade_date: str) -> str:
    return ResearchReport(
        ticker=ticker,
        agent_type=AgentType.NEWS,
        timestamp=datetime.strptime(trade_date, "%Y-%m-%d"),
        metrics_selected=[],
        mu=HorizonValues(long_term=0.0, medium_term=0.0, short_term=0.0),
        mu_trace_id=None,
        sigma_contribution=HorizonValues(
            long_term=0.01, medium_term=0.01, short_term=0.01
        ),
        sigma_trace_id=None,
        computed_metrics=[],
        investment_thesis=HorizonThesis(long_term="", medium_term="", short_term=""),
        conviction=HorizonValues(long_term=0.0, medium_term=0.0, short_term=0.0),
        key_catalysts=[],
        key_risks=[],
        source_uuids=[],
        computation_traces=[],
        citation_chain=[],
        contributing_factors=[],
    ).model_dump_json()


# ── Test ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    import pathlib
    import statistics
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _project_root = str(pathlib.Path(__file__).resolve().parents[3])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _log = logging.getLogger("news_analyst_test")

    # ── Config ──────────────────────────────────────────────────────────
    TICKER = "WMT"
    TRADE_DATE = "2025-03-14"
    DEPTH = "shallow"
    NUM_RUNS = 30
    MAX_WORKERS = 10  # parallel analyses (adjust to your GPU / context budget)
    RANDOM_SEED = 42

    # ── Build shared LLM + node ───────────────────────────────────────
    _log.info("Creating LLM client (ollama / minimax-m2.7:cloud) …")
    from src.llm_clients import create_llm_client

    llm = create_llm_client(
        provider="ollama",
        model="minimax-m2.7:cloud",
        base_url="http://localhost:11434/v1",
        max_retries=10,
        reasoning_effort="high",
        temperature=0,
        seed=RANDOM_SEED,
    ).get_llm()

    _log.info("Creating news-analyst node (depth=%s) …", DEPTH)
    node = create_news_analyst(llm, DEPTH)
    init_state = {"company_of_interest": TICKER, "trade_date": TRADE_DATE}

    # ── Parallel runner ───────────────────────────────────────────────
    def _run_once(run_idx: int) -> dict | None:
        try:
            result = node(init_state)
            raw_json = result.get("news_report", "{}")
            from src.agents.utils.schemas import ResearchReport

            report = ResearchReport.model_validate_json(raw_json)
            return {
                "idx": run_idx,
                "ok": True,
                "mu_l": report.mu.long_term,
                "mu_m": report.mu.medium_term,
                "mu_s": report.mu.short_term,
                "sigma_l": report.sigma_contribution.long_term,
                "sigma_m": report.sigma_contribution.medium_term,
                "sigma_s": report.sigma_contribution.short_term,
                "conv_l": report.conviction.long_term,
                "conv_m": report.conviction.medium_term,
                "conv_s": report.conviction.short_term,
                "report": report,
                "confidence_rationale": result.get("confidence_rationale", {}),
            }
        except Exception as exc:
            return {"idx": run_idx, "ok": False, "error": str(exc)}

    _log.info(
        "Running %d analyses in parallel (max_workers=%d) …",
        NUM_RUNS,
        MAX_WORKERS,
    )
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_once, i): i for i in range(1, NUM_RUNS + 1)}
        for fut in as_completed(futures):
            res = fut.result()
            idx = res["idx"]
            if res["ok"]:
                print(
                    f"  Run {idx:02d}/{NUM_RUNS}  OK  "
                    f"mu=[{res['mu_l']:+.4f},{res['mu_m']:+.4f},{res['mu_s']:+.4f}]  "
                    f"conv=[{res['conv_l']:.2f},{res['conv_m']:.2f},{res['conv_s']:.2f}]"
                )
                results.append(res)
            else:
                print(f"  Run {idx:02d}/{NUM_RUNS}  FAIL  {res['error']}")

    # ── Deviation summary ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(
        f"  DEVIATION SUMMARY  ({len(results)}/{NUM_RUNS} OK, {TICKER} @ {TRADE_DATE})"
    )
    print(f"{'=' * 60}\n")

    for label, key in [
        ("mu (long)", "mu_l"),
        ("mu (medium)", "mu_m"),
        ("mu (short)", "mu_s"),
        ("sigma (long)", "sigma_l"),
        ("sigma (medium)", "sigma_m"),
        ("sigma (short)", "sigma_s"),
        ("conviction (long)", "conv_l"),
        ("conviction (medium)", "conv_m"),
        ("conviction (short)", "conv_s"),
    ]:
        vals = [r[key] for r in results]
        if len(vals) < 2:
            print(f"  {label:20s}: n={len(vals)}  (insufficient for stats)")
            continue
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals)
        spread = max(vals) - min(vals)
        print(
            f"  {label:20s}: n={len(vals):2d}  "
            f"mean={mean:+.6f}  stdev={stdev:.6f}  spread={spread:.6f}  "
            f"range=[{min(vals):+.6f}, {max(vals):+.6f}]"
        )

    # ── Pretty-print first successful report ──────────────────────────
    if results:
        report = results[0]["report"]
        print(f"\n{'=' * 60}")
        print("  SAMPLE REPORT (run 1)")
        print(f"{'=' * 60}")
        print(f"  ticker           : {report.ticker}")
        print(
            f"  mu               : L={report.mu.long_term:+.4f}  M={report.mu.medium_term:+.4f}  S={report.mu.short_term:+.4f}"
        )
        print(
            f"  sigma            : L={report.sigma_contribution.long_term:.4f}  M={report.sigma_contribution.medium_term:.4f}  S={report.sigma_contribution.short_term:.4f}"
        )
        print(
            f"  conviction       : L={report.conviction.long_term:+.2f}  M={report.conviction.medium_term:+.2f}  S={report.conviction.short_term:+.2f}"
        )
        cr = results[0].get("confidence_rationale", {})
        if cr:
            print("  confidence rationale:")
            for h in ("long_term", "medium_term", "short_term"):
                txt = cr.get(h, "")
                if txt:
                    print(
                        f"    • [{h[:1].upper()}] {txt[:100]}…"
                        if len(txt) > 100
                        else f"    • [{h[:1].upper()}] {txt}"
                    )
        print(f"  catalysts        : {len(report.key_catalysts)}")
        for cat in report.key_catalysts:
            print(f"    • [{cat.term}] {cat.catalyst}")
        print(f"  risks            : {len(report.key_risks)}")
        for rsk in report.key_risks:
            print(f"    • [{rsk.term}] {rsk.risk}")
        print(f"  citations        : {len(report.citation_chain)}")
        for cit in report.citation_chain[:5]:
            print(f"    • {cit.source_uuid}  {cit.source}")
        if len(report.citation_chain) > 5:
            print(f"    … and {len(report.citation_chain) - 5} more")
        print(f"{'=' * 60}")
        print("\nFull JSON of run 1:")
        print(report.model_dump_json(indent=2))
    print("\nDone.")
