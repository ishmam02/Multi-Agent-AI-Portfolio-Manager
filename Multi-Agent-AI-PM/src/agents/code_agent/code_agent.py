"""
CodeValidationAgent — Phase 2 autonomous coding sub-agent.

Launches the Claude Code CLI via ``ollama launch claude --print`` in
non-interactive mode.  The agent has access to:

  • Bash       — write and execute Python scripts in the isolated work directory
  • WebSearch  — search for library documentation or debugging help (fallback only)
  • WebFetch   — fetch specific documentation URLs (fallback only)

The agent always writes and runs code first.  WebSearch / WebFetch are used
only as a fallback when execution fails with an error the agent cannot resolve
from the traceback alone.

Invocation
──────────
  ollama launch claude --model <model> -- \\
      -p \\
      --append-system-prompt <system_prompt> \\
      --allowed-tools Bash,Read,Write,Edit,WebSearch,WebFetch,Skill \\
      --permission-mode bypassPermissions \\
      --no-session-persistence \\
      --output-format stream-json \\
      --add-dir <work_dir> \\
      <prompt>

Agent loop
──────────
  1. Agent receives the computation plan + scaffold metrics.py in the prompt.
  2. Agent overwrites metrics.py with a full implementation and runs it.
  3. If execution fails, the agent fixes the code and re-runs it.
  4. If the error is unresolvable from the traceback, the agent may use
     WebSearch / WebFetch to look up the specific API or formula.
  5. execute_plan() re-runs the final metrics.py, validates the output, and
     returns the result dict.  Falls back to scanning the CLI output for
     embedded JSON if re-execution fails.

Return schema
─────────────
  mu                  : float       — annualised expected return (e.g. 0.08)
  sigma               : float       — annualised volatility     (e.g. 0.22)
  computed_metrics    : list[dict]  — [{"metric_name": str, "value": Any}]
  computation_traces  : list[dict]  — [{"trace_id": str, "code": str,
                                        "inputs": dict, "output": Any}]
  code_succeeded      : bool
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from datetime import datetime
from typing import Any


# ── Project root (Multi-Agent-AI-PM/) ────────────────────────────────────────


def _find_project_root() -> str:
    """Walk up from this file looking for the directory containing pyproject.toml.

    Falls back to a 4-level dirname if the marker is not found (e.g. when
    running from an isolated copy).
    """
    path = os.path.abspath(__file__)
    for _ in range(10):
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
        if path == os.path.dirname(path):
            break
    # Fallback: assume code_agent/ → agents/ → src/ → Multi-Agent-AI-PM/
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


_PROJECT_ROOT = _find_project_root()

# ── File-based logging per analyst / horizon ─────────────────────────────────
# Log files: results/{ticker}/{date}/{analyst_type}/{horizon}/code_agent.log
# Cache of open loggers keyed by work_dir path.
_code_loggers: dict[str, logging.Logger] = {}


def _get_code_logger(work_dir: str) -> logging.Logger:
    """Return (or create) a file logger that writes to work_dir/code_agent.log."""
    if work_dir in _code_loggers:
        return _code_loggers[work_dir]

    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "code_agent.log")

    logger = logging.getLogger(f"code_agent.{work_dir}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(fh)

    _code_loggers[work_dir] = logger
    return logger


# ── Allowed-library registry ─────────────────────────────────────────────────

_COMMON_LIBS = ["math", "json", "datetime", "typing", "statistics"]

ANALYST_LIBRARIES: dict[str, list[str]] = {
    "fundamental": ["numpy", "pandas", "scipy"] + _COMMON_LIBS,
    "market": ["numpy", "pandas", "talib", "scipy", "scikit-learn"] + _COMMON_LIBS,
    "news": ["numpy", "pandas", "re", "collections", "itertools"] + _COMMON_LIBS,
}

# ── Load prompts from external markdown file ────────────────────────────────

from src.agents.prompts import load_prompt

_code_agent_prompts = load_prompt("code_agent")
_DOMAIN: dict[str, str] = _code_agent_prompts["DOMAIN"]  # type: ignore[assignment]
_TOOL_INSTRUCTIONS: str = _code_agent_prompts["TOOL_INSTRUCTIONS"]  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# CodeValidationAgent
# ─────────────────────────────────────────────────────────────────────────────


class CodeValidationAgent:
    """
    Phase 2 sub-agent: launches ``ollama launch claude`` to autonomously
    compute financial metrics via code generation, execution, web-assisted
    debugging, and iterative self-correction.

    Instantiate one agent per analyst type; the system prompt is tailored to
    the domain and the set of allowed Python libraries.

    Example
    -------
    >>> agent = CodeValidationAgent(
    ...     model="qwen3-coder:latest",
    ...     analyst_type="fundamental",
    ... )
    >>> result = agent.execute_plan(computation_plan, data)
    >>> result["code_succeeded"]
    True
    """

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        max_iterations: int = 8,
        analyst_type: str = "fundamental",
        verbose: bool = False,
        project_root: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model          : Model name passed to `ollama launch claude --model`.
        timeout        : Per-subprocess timeout multiplier (seconds).
        max_iterations : Controls the subprocess deadline:
                         deadline = timeout × max_iterations × 6.
        analyst_type   : One of "fundamental", "market".
        verbose        : If True, stream every agent event to stdout in real time.
        project_root   : Explicit path to Multi-Agent-AI-PM/.  If None, auto-
                         detected via pyproject.toml marker.
        """
        if analyst_type.lower() not in ANALYST_LIBRARIES:
            raise ValueError(
                f"analyst_type must be one of {list(ANALYST_LIBRARIES)}, "
                f"got '{analyst_type}'."
            )

        self.model = model
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.analyst_type = analyst_type.lower()
        self.verbose = verbose
        self._project_root = project_root or _PROJECT_ROOT

        self._allowed_libs: list[str] = ANALYST_LIBRARIES[self.analyst_type]
        self._current_work_dir: str = ""  # set during _run_agent

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, tag: str, text: str, *, truncate: int = 0) -> None:
        """Print a timestamped log line when verbose=True AND write to file logger."""
        ts = datetime.now().strftime("%H:%M:%S")
        body = text[:truncate] if truncate > 0 else text

        # Always write to file logger if work_dir is set
        if self._current_work_dir:
            logger = _get_code_logger(self._current_work_dir)
            logger.info("[%s] %s", tag, body)

        # Also print to stdout if verbose
        # if not self.verbose:
        #     return
        # first, *rest = body.splitlines()
        # if rest:
        #     pad = " " * (len(ts) + len(tag) + 5)
        #     body = first + "\n" + textwrap.indent("\n".join(rest), pad)
        # print(f"[{ts}] [{tag}] {body}", flush=True)

    def _log_block(self, header: str, body: str) -> None:
        """Print a full block of text with a header divider AND write to file logger."""
        ts = datetime.now().strftime("%H:%M:%S")
        sep = "─" * 60

        # Always write to file logger if work_dir is set
        if self._current_work_dir:
            logger = _get_code_logger(self._current_work_dir)
            logger.info("┌%s", sep)
            logger.info("│ %s", header)
            logger.info("└%s", sep)
            for line in body.splitlines():
                logger.info("%s", line)
            logger.info("─%s─", sep)

        # Also print to stdout if verbose
        # if not self.verbose:
        #     return
        # print(f"\n[{ts}] ┌{sep}", flush=True)
        # print(f"[{ts}] │ {header}", flush=True)
        # print(f"[{ts}] └{sep}", flush=True)
        # print(body, flush=True)
        # print(f"[{ts}] ─{sep}─\n", flush=True)

    # ── Scaffolding ───────────────────────────────────────────────────────────

    @staticmethod
    def _classify_str(text: str) -> str:
        """Classify a string payload as 'csv', 'kv' (key-value), or 'text'.

        csv  — tabular comma-separated data (e.g. OHLCV, financial statements)
        kv   — label: value pairs (e.g. fundamentals overview)
        text — free-form prose / markdown (e.g. news articles)
        """
        content_lines = [
            line
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not content_lines:
            return "text"
        # Prose / markdown: has ## section headers
        if any(line.lstrip().startswith("##") for line in content_lines):
            return "text"
        # Tabular CSV: first content line has at least 1 comma (header row)
        if content_lines[0].count(",") >= 1:
            return "csv"
        # Key-value: most lines match "Short Label: value" pattern
        kv_hits = sum(
            1
            for line in content_lines
            if re.match(r"^[A-Za-z][A-Za-z0-9 ()\-_%/]*:\s*\S", line)
        )
        if kv_hits / len(content_lines) >= 0.6:
            return "kv"
        return "text"

    @staticmethod
    def _is_financial_statement_csv(text: str) -> bool:
        """Detect financial-statement CSVs produced by yfinance.

        These CSVs have:
          • first header = 'Metric' (or empty before fix) followed by date columns
          • first data rows = metric names like 'Total Revenue', 'Net Income', etc.
          • columns are reporting periods (dates), not time-series observations
        """
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False
        header = lines[0]
        # Header starts with 'Metric' (after fix) or empty first field, then dates
        parts = header.split(",")
        if len(parts) < 2:
            return False
        first_field = parts[0].strip()
        if first_field not in ("", "Metric"):
            return False
        # At least one column looks like a date (YYYY-MM-DD)
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        return any(date_pattern.match(p.strip()) for p in parts[1:])

    @staticmethod
    def _scaffold_metrics_file(work_dir: str, data: dict) -> tuple[str, list[str]]:
        """Write all data files to *work_dir* and return (script_path, created_files).

        Each value in *data* is written to an appropriately typed file:
          str (csv tabular)  → <key>.csv   — loaded as pd.read_csv(index_col=0)
          str (key-value)    → <key>.csv   — 2-column metric/value CSV
          str (prose/text)   → <key>.txt   — plain text, read with open()
          dict               → <key>.json  — loaded with json.load()
          list[dict]         → <key>.csv   — DictWriter CSV
          other              → <key>.json
        """
        import csv as csv_mod

        created_files: list[str] = []
        # Each entry: (varname, filename, load_snippet)
        load_entries: list[tuple[str, str, str]] = []
        primary_df_var: str | None = None  # first tabular-CSV variable name

        for key, val in data.items():
            if isinstance(val, str):
                kind = CodeValidationAgent._classify_str(val)

                if kind == "csv":
                    cleaned = "\n".join(
                        line
                        for line in val.splitlines()
                        if not line.strip().startswith("#")
                    ).strip()
                    fname = f"{key}.csv"
                    with open(
                        os.path.join(work_dir, fname), "w", encoding="utf-8"
                    ) as fh:
                        fh.write(cleaned + "\n")
                    varname = "df" if primary_df_var is None else f"{key}_df"
                    if primary_df_var is None:
                        primary_df_var = varname

                    # Financial-statement CSVs have metric names as index and date
                    # strings as columns — do NOT parse_dates on the index.
                    is_fs = CodeValidationAgent._is_financial_statement_csv(cleaned)
                    if is_fs:
                        snippet = (
                            f'{varname} = pd.read_csv("{fname}", index_col=0)\n'
                            f"# index = metric names (e.g. 'Total Revenue', 'Net Income')\n"
                            f"# columns = reporting-period dates (strings like '2025-12-31')\n"
                        )
                    else:
                        snippet = (
                            f'{varname} = pd.read_csv("{fname}", index_col=0, parse_dates=True)\n'
                            f"try:\n    {varname}.sort_index(inplace=True)\nexcept Exception:\n    pass\n"
                        )

                elif kind == "kv":
                    rows = []
                    for line in val.splitlines():
                        if line.strip().startswith("#") or not line.strip():
                            continue
                        if ":" in line:
                            metric, _, value = line.partition(":")
                            rows.append(
                                {"metric": metric.strip(), "value": value.strip()}
                            )
                    fname = f"{key}.csv"
                    with open(
                        os.path.join(work_dir, fname), "w", newline="", encoding="utf-8"
                    ) as fh:
                        writer = csv_mod.DictWriter(fh, fieldnames=["metric", "value"])
                        writer.writeheader()
                        writer.writerows(rows)
                    varname = f"{key}_df"
                    snippet = f'{varname} = pd.read_csv("{fname}")\n'

                else:  # text / prose
                    fname = f"{key}.txt"
                    with open(
                        os.path.join(work_dir, fname), "w", encoding="utf-8"
                    ) as fh:
                        fh.write(val.strip() + "\n")
                    varname = f"{key}_text"
                    snippet = f'{varname} = open("{fname}").read()\n'

            elif isinstance(val, dict):
                fname = f"{key}.json"
                with open(os.path.join(work_dir, fname), "w", encoding="utf-8") as fh:
                    json.dump(val, fh, indent=2)
                varname = f"{key}_data"
                snippet = (
                    f'with open("{fname}") as _f:\n    {varname} = json.load(_f)\n'
                )

            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                fname = f"{key}.csv"
                with open(
                    os.path.join(work_dir, fname), "w", newline="", encoding="utf-8"
                ) as fh:
                    writer = csv_mod.DictWriter(fh, fieldnames=val[0].keys())
                    writer.writeheader()
                    writer.writerows(val)
                varname = "df" if primary_df_var is None else f"{key}_df"
                if primary_df_var is None:
                    primary_df_var = varname
                snippet = f'{varname} = pd.read_csv("{fname}")\n'

            else:
                fname = f"{key}.json"
                with open(os.path.join(work_dir, fname), "w", encoding="utf-8") as fh:
                    json.dump(val, fh, indent=2)
                varname = f"{key}_data"
                snippet = (
                    f'with open("{fname}") as _f:\n    {varname} = json.load(_f)\n'
                )

            created_files.append(fname)
            load_entries.append((varname, fname, snippet))

        # Fallback: no data at all
        if not load_entries:
            empty_csv = "data.csv"
            with open(os.path.join(work_dir, empty_csv), "w", encoding="utf-8") as fh:
                fh.write("")
            created_files.append(empty_csv)
            primary_df_var = "df"
            load_entries.append(("df", empty_csv, 'df = pd.read_csv("data.csv")\n'))

        if primary_df_var is None:
            # All data was non-tabular; fall back to first variable for stubs
            primary_df_var = load_entries[0][0]

        load_block = "".join(snippet for _, _, snippet in load_entries)

        script_path = os.path.join(work_dir, "metrics.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(
                "import inspect\n"
                "import json\n"
                "import uuid\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "np.random.seed(42)\n"
                "\n"
                f"{load_block}"
                "\n"
                "\n"
                "def build_computation_trace(func, inputs, output):\n"
                '    """Build a trace dict for one metric function.\n'
                "\n"
                "    Parameters\n"
                "    ----------\n"
                "    func   : the callable that produced the result\n"
                "    inputs : dict of key inputs / data slices used\n"
                "    output : the value returned by func\n"
                "\n"
                "    Returns\n"
                "    -------\n"
                "    dict with trace_id, code (full function source), inputs, output\n"
                '    """\n'
                "    return {\n"
                '        "trace_id": str(uuid.uuid4()),\n'
                '        "code": inspect.getsource(func),\n'
                '        "inputs": inputs,\n'
                '        "output": output,\n'
                "    }\n"
                "\n"
                "\n"
                "def build_computed_metric(metric_name, value, trace):\n"
                '    """Build a computed metric dict linked to its computation trace.\n'
                "\n"
                "    Parameters\n"
                "    ----------\n"
                "    metric_name            : str — e.g. 'rsi', 'dcf_fair_value'\n"
                "    value                  : the computed result (float, dict, etc.)\n"
                "    trace                  : the dict returned by build_computation_trace\n"
                '    """\n'
                "    return {\n"
                '        "metric_name": metric_name,\n'
                '        "value": value,\n'
                '        "computation_trace_id": trace["trace_id"],\n'
                "    }\n"
                "\n"
                "\n"
                f"def compute_mu({primary_df_var}):\n"
                "    # TODO: replace with real implementation\n"
                "    return 0.0\n"
                "\n"
                "\n"
                f"def compute_sigma({primary_df_var}):\n"
                "    # TODO: replace with real implementation\n"
                "    return 0.01\n"
                "\n"
                "\n"
                "def main():\n"
                f"    mu_val = compute_mu({primary_df_var})\n"
                "    mu_trace = build_computation_trace(\n"
                f'        compute_mu, {{"rows": len({primary_df_var})}}, mu_val\n'
                "    )\n"
                "    mu_metric = build_computed_metric(\n"
                '        "mu", mu_val,\n'
                "        mu_trace,\n"
                "    )\n"
                "\n"
                f"    sigma_val = compute_sigma({primary_df_var})\n"
                "    sigma_trace = build_computation_trace(\n"
                f'        compute_sigma, {{"rows": len({primary_df_var})}}, sigma_val\n'
                "    )\n"
                "    sigma_metric = build_computed_metric(\n"
                '        "sigma", sigma_val,\n'
                "        sigma_trace,\n"
                "    )\n"
                "\n"
                "    computed_metrics = [mu_metric, sigma_metric]\n"
                "    computation_traces = [mu_trace, sigma_trace]\n"
                "\n"
                "    # metrics_selected: list every metric you computed with interpretation and rationale\n"
                "    metrics_selected = [\n"
                "        {\n"
                '            "metric_name": "mu",\n'
                '            "metric_interpretation": "Annualised expected return",\n'
                '            "metric_rationale": "Primary directional signal",\n'
                '            "computation_instruction": "See compute_mu function",\n'
                "        },\n"
                "        {\n"
                '            "metric_name": "sigma",\n'
                '            "metric_interpretation": "Annualised volatility",\n'
                '            "metric_rationale": "Uncertainty in return estimate",\n'
                '            "computation_instruction": "See compute_sigma function",\n'
                "        },\n"
                "    ]\n"
                "\n"
                "    result = {\n"
                '        "mu": mu_val,\n'
                '        "mu_trace_id": mu_trace["trace_id"],\n'
                '        "sigma": sigma_val,\n'
                '        "sigma_trace_id": sigma_trace["trace_id"],\n'
                '        "computed_metrics": computed_metrics,\n'
                '        "computation_traces": computation_traces,\n'
                '        "metrics_selected": metrics_selected,\n'
                "    }\n"
                "    print(json.dumps(result, indent=2))\n"
                "\n"
                "\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            )
        created_files.append("metrics.py")
        return script_path, created_files

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_system_prompt(
        self, domain_prompt: str | None = None, work_dir: str = ""
    ) -> str:
        domain = domain_prompt or _DOMAIN[self.analyst_type]
        tools = _TOOL_INSTRUCTIONS.format(
            allowed_libs=", ".join(self._allowed_libs),
            work_dir=work_dir,
        )
        return f"{domain}\n\n{tools}"

    @staticmethod
    def _build_user_prompt(
        computation_plan: dict,
        data: dict,
        data_files: list[str] | None = None,
    ) -> str:
        # List the data files available in the working directory instead of
        # dumping the full raw data into the prompt — this drastically reduces
        # context size and speeds up every turn.
        if data_files:
            file_list = "\n".join(f"  • {f}" for f in data_files)
            data_section = (
                f"=== DATA FILES (already in your working directory) ===\n"
                f"{file_list}\n"
                f"Data keys: {list(data.keys())}\n"
            )
        else:
            data_section = f"=== RAW DATA ===\n{json.dumps(data, indent=2)}\n"

        # Format computation_plan List[Metrics] for clarity
        metrics_list = computation_plan.get("computation_plan", [])
        if metrics_list:
            metrics_lines = "\n".join(
                f"  {i + 1}. {m.get('metric_name', '?')}\n"
                f"       interpretation: {m.get('metric_interpretation', '')}\n"
                f"       rationale:      {m.get('metric_rationale', '')}\n"
                f"       instruction:    {m.get('computation_instruction', '')}"
                for i, m in enumerate(metrics_list)
                if isinstance(m, dict)
            )
            plan_section = (
                f"=== COMPUTATION PLAN ({len(metrics_list)} metrics) ===\n"
                f"{metrics_lines}\n"
            )
            top_instruction = (
                "Compute the financial metrics defined in the plan below.\n\n"
            )
        else:
            # Plan-less mode: the code agent must analyse data, plan metrics, then code.
            additional = computation_plan.get("additional_instructions", "")
            revision = computation_plan.get("revision_count", 0)
            plan_payload_str = json.dumps(computation_plan, indent=2)
            plan_section = f"=== PLANNING CONTEXT ===\n{plan_payload_str}\n"
            if additional:
                plan_section += f"\n=== ADDITIONAL INSTRUCTIONS ===\n{additional}\n"
            if revision > 0:
                plan_section += (
                    f"\n=== RETRY ATTEMPT {revision} ===\n"
                    "Your previous code failed to execute or produced invalid output. "
                    "SIMPLIFY your plan: use fewer metrics, simpler formulas, and "
                    "avoid complex data parsing. Focus on the core signals only.\n"
                )
            top_instruction = (
                "You are an autonomous quantitative analyst.\n"
                "STEP 1 — QUICKLY SCAN the data files in your working directory. "
                "Use a fast header check (e.g. `head -n 3` or Read the first few lines) "
                "to see available columns and metric names. Do NOT read full data contents "
                "or perform deep exploratory analysis. You only need headers and a handful of "
                "rows to orient yourself.\n"
                "STEP 2 — DECIDE which metrics to compute based on available columns. "
                "Choose models and parameters that fit THIS specific stock, but do not use "
                "a one-size-fits-all formula.\n"
                "STEP 3 — IMMEDIATELY WRITE clean, vectorised Python code in metrics.py "
                "that computes all chosen metrics, including mu and sigma.\n"
                "STEP 4 — RUN `python3 metrics.py` and iterate until it exits 0 with valid JSON.\n\n"
            )

        return (
            f"{top_instruction}"
            f"{plan_section}\n"
            f"{data_section}\n"
            "IMPLEMENT FIRST: immediately write the full metrics.py using Bash and "
            "run it.  All libraries are pre-installed.  Do NOT search the web or "
            "install packages before writing code.  Only use WebSearch if execution "
            "fails with an error you cannot resolve from the traceback.\n"
            "Keep iterating until `python3 metrics.py` exits 0 and prints valid JSON."
        )

    # ── Result extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _has_result_keys(d: Any) -> bool:
        """Return True if *d* looks like a metrics result dict."""
        return isinstance(d, dict) and (
            "mu" in d or "computed_metrics" in d or "horizons" in d
        )

    @staticmethod
    def _extract_result_json(text: str) -> dict | None:
        """
        Extract the metrics result JSON from *text*.

        Tries in order:
          1. Direct ``json.loads`` on stripped text (clean stdout).
          2. Fenced ```result``` / ```json``` blocks via ``raw_decode``.
          3. Forward scan for every ``{`` position via ``raw_decode``,
             keeping the last valid match.

        ``json.JSONDecoder.raw_decode`` is used instead of manual brace
        counting so that ``{`` / ``}`` inside JSON string values (e.g. the
        full source code in ``computation_traces[].code``) are handled correctly.

        Every candidate is validated against the expected schema keys
        (``mu`` or ``computed_metrics``) before being accepted.
        Returns None if no valid result JSON is found.
        """
        _valid = CodeValidationAgent._has_result_keys
        decoder = json.JSONDecoder()

        # 1. Direct parse — clean stdout from `python3 metrics.py`
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                d = json.loads(stripped)
                if _valid(d):
                    return d
            except json.JSONDecodeError:
                pass

        # 2. Fenced ```result ... ``` or ```json ... ``` blocks
        for m in re.finditer(r"```(?:result|json)\s*", text):
            try:
                d, _ = decoder.raw_decode(text, m.end())
                if isinstance(d, dict) and _valid(d):
                    return d
            except (json.JSONDecodeError, ValueError):
                continue

        # 3. Forward scan for { positions — keep the dict with the MOST keys.
        #    raw_decode respects JSON string quoting, so braces inside
        #    string values (e.g. full_code_trace.code) are skipped.
        #    The outer result dict has 7+ keys; nested inner dicts (traces,
        #    inputs, etc.) have fewer.  Picking the largest avoids extracting
        #    a nested trace dict instead of the full result.
        best_valid: dict | None = None
        best_size = -1
        for m in re.finditer(r"\{", text):
            try:
                d, _ = decoder.raw_decode(text, m.start())
                if isinstance(d, dict) and _valid(d) and len(d) > best_size:
                    best_valid = d
                    best_size = len(d)
            except (json.JSONDecodeError, ValueError):
                continue
        return best_valid

    # ── Result validation ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _validate(self, raw: dict) -> dict:
        metrics = []
        for m in raw.get("computed_metrics", []):
            if isinstance(m, dict) and "metric_name" in m:
                entry: dict[str, Any] = {
                    "metric_name": str(m["metric_name"]),
                    "value": m.get("value"),
                }
                if "computation_trace_id" in m:
                    entry["computation_trace_id"] = str(m["computation_trace_id"])
                if "term" in m:
                    entry["term"] = str(m["term"])
                metrics.append(entry)
        traces = [
            {
                "trace_id": str(t.get("trace_id", uuid.uuid4())),
                "code": str(t.get("code", "")),
                "inputs": t.get("inputs", {}),
                "output": t.get("output"),
            }
            for t in raw.get("computation_traces", [])
            if isinstance(t, dict)
        ]
        # Extract metrics_selected (the plan the code agent created)
        metrics_selected: list[dict] = []
        for m in raw.get("metrics_selected", []):
            if isinstance(m, dict) and "metric_name" in m:
                metrics_selected.append(
                    {
                        "metric_name": str(m["metric_name"]),
                        "metric_interpretation": str(
                            m.get("metric_interpretation", "")
                        ),
                        "metric_rationale": str(m.get("metric_rationale", "")),
                        "computation_instruction": str(
                            m.get("computation_instruction", "")
                        ),
                    }
                )
        # Support multi-horizon output: pass through "horizons" dict if present
        result: dict[str, Any] = {
            "computed_metrics": metrics,
            "computation_traces": traces,
            "metrics_selected": metrics_selected,
            "code_succeeded": True,
        }
        if "horizons" in raw and isinstance(raw["horizons"], dict):
            result["horizons"] = raw["horizons"]
        else:
            result["mu"] = self._safe_float(raw.get("mu"), 0.0)
            result["sigma"] = max(self._safe_float(raw.get("sigma", 0.01)), 1e-6)
        if "mu_trace_id" in raw:
            result["mu_trace_id"] = str(raw["mu_trace_id"])
        if "sigma_trace_id" in raw:
            result["sigma_trace_id"] = str(raw["sigma_trace_id"])
        return result

    def _failure_fallback(self, reason: str) -> dict:
        return {
            "mu": 0.0,
            "sigma": 0.01,
            "computed_metrics": [],
            "computation_traces": [
                {
                    "trace_id": str(uuid.uuid4()),
                    "code": "",
                    "inputs": {},
                    "output": f"FAILED: {reason}",
                }
            ],
            "metrics_selected": [],
            "code_succeeded": False,
        }

    # ── Subprocess re-execution ───────────────────────────────────────────────

    def _rerun_metrics_file(self, work_dir: str) -> tuple[str, str, int]:
        """
        Re-execute work_dir/metrics.py in place so any helper files or
        installed packages the agent left in work_dir are still available.
        Falls back to the newest .py file in work_dir if metrics.py is absent.
        Returns (stdout, stderr, returncode).
        """
        script = os.path.join(work_dir, "metrics.py")
        if not os.path.isfile(script):
            # Fallback: pick the most recently modified .py file the agent wrote
            candidates = sorted(
                (
                    p
                    for p in (os.path.join(work_dir, f) for f in os.listdir(work_dir))
                    if p.endswith(".py") and os.path.isfile(p)
                ),
                key=os.path.getmtime,
                reverse=True,
            )
            if not candidates:
                return "", "No .py file found in work_dir.", 1
            script = candidates[0]
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=work_dir,
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", f"Re-execution timed out after {self.timeout}s.", 1
        except Exception as exc:
            return "", f"Subprocess error: {exc}", 1

    # ── Claude CLI subprocess call ────────────────────────────────────────────

    def _run_agent(
        self, computation_plan: dict, data: dict, domain_prompt: str | None = None
    ) -> tuple[str, str]:
        """
        Launch ``ollama launch claude`` in non-interactive (--print) mode.

        Streams ``--output-format stream-json`` line by line; logs each event
        when verbose=True.  The ``result`` event's ``result`` field becomes
        ``result_text`` and is returned to the caller for JSON extraction.

        Returns
        -------
        (result_text, work_dir)
            result_text : text from the CLI ``result`` event (may be empty).
            work_dir    : directory where metrics.py and data files live.
        """
        ticker = computation_plan.get("ticker", "UNKNOWN")
        analysis_date = computation_plan.get("analysis_date", "unknown")
        active_horizons = computation_plan.get("active_horizons", [])
        horizon = computation_plan.get("horizon")
        if horizon is None:
            if isinstance(active_horizons, (list, tuple)) and len(active_horizons) == 1:
                horizon = active_horizons[0]
            else:
                horizon = "unknown"
        # Use "multi" when multiple horizons are computed in one pass
        if isinstance(active_horizons, (list, tuple)) and len(active_horizons) > 1:
            work_dir_horizon = "multi"
        else:
            work_dir_horizon = horizon
        work_dir = os.path.join(
            self._project_root,
            "results",
            ticker,
            analysis_date,
            self.analyst_type,
            work_dir_horizon,
            uuid.uuid4().hex[:8],
        )
        os.makedirs(work_dir, exist_ok=True)
        self._current_work_dir = work_dir
        self._log(
            "INIT",
            f"project_root={self._project_root}  work_dir={work_dir}  model={self.model}",
        )

        # Pre-create metrics.py + data files so the agent edits rather than
        # writes from scratch.
        script_path, data_files = self._scaffold_metrics_file(work_dir, data)
        self._log("SCAFFOLD", f"Created {script_path} + sidecar data files")

        # ── Off-load long domain prompts to a file so the system prompt stays short.
        #    The agent reads the file on its first turn; after that every turn
        #    processes only the short system prompt.
        short_domain: str | None = domain_prompt
        if domain_prompt and len(domain_prompt) > 2000:
            instructions_path = os.path.join(work_dir, "instructions.txt")
            with open(instructions_path, "w", encoding="utf-8") as fh:
                fh.write(domain_prompt)
            short_domain = (
                f"You are a quantitative {self.analyst_type} analyst. "
                f"Read {instructions_path} for detailed methodology before coding."
            )
            self._log("PROMPT", f"Off-loaded long domain prompt to {instructions_path}")

        # ── Gather context the agent would normally spend turns discovering ──
        dir_listing = "\n".join(f"  {f}" for f in sorted(os.listdir(work_dir)))

        # Build concise data preview for CSV files: headers / metric names ONLY.
        # No actual data values are sent — the agent can Read the file if it needs them.
        data_preview = ""
        for fname in sorted(data_files):
            if not fname.endswith(".csv") or fname == "metrics.py":
                continue
            fpath = os.path.join(work_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue

            lines = text.splitlines()
            if not lines:
                continue

            # Every CSV: show the header row (column names).
            data_preview += f"\n=== {fname} HEADER ===\n{lines[0]}\n"

            # Financial-statement CSVs: also list the metric names (first column of
            # each data row) so the agent knows what metrics are available.
            if CodeValidationAgent._is_financial_statement_csv(text):
                metrics = []
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    first_col = line.split(",")[0].strip()
                    if first_col:
                        metrics.append(first_col)
                if metrics:
                    data_preview += f"METRICS: {', '.join(metrics)}\n"

        # Add instructions.txt reference if it was off-loaded
        instructions_ref = ""
        instructions_path = os.path.join(work_dir, "instructions.txt")
        if os.path.isfile(instructions_path):
            instructions_ref = (
                f"\n=== METHODOLOGY ===\n"
                f"Read {instructions_path} for the full computation methodology "
                "(formulas, metrics, output format) before writing any code.\n"
            )

        prompt = (
            self._build_user_prompt(computation_plan, data, data_files=data_files)
            + f"\n\n=== WORKING DIRECTORY: {work_dir} ===\n{dir_listing}"
            + data_preview
            + instructions_ref
            + f"\n\n=== SCAFFOLD FILE ===\n"
            f"{script_path} already exists with `build_computation_trace`, "
            f"`build_computed_metric`, and `compute_mu` / `compute_sigma` stubs. "
            f"Read it first, then overwrite it with your full implementation.\n"
            + f"\nIMPORTANT: Start immediately. Read {script_path} first, then overwrite "
            f"it with a heredoc (`cat > {script_path} << 'PYEOF' ... PYEOF`) and run "
            f"it with `python3 {script_path}`.\n"
            "CRITICAL: You MUST use ONLY the exact absolute path above for metrics.py. "
            "Do NOT use a relative path like `cat > metrics.py`. Do NOT guess or invent "
            "any other directory. The scaffold already exists at that exact path; overwrite it there.\n"
            "Keep these helpers exactly as-is:\n"
            "  • `build_computation_trace(func, inputs, output)`\n"
            "  • `build_computed_metric(metric_name, value, trace)`\n\n"
            "Your implementation must:\n"
            "  1. Replace `compute_mu` and `compute_sigma` with real implementations.\n"
            "     For multi-horizon runs, use per-horizon names like `compute_mu_long_term`,\n"
            "     `compute_sigma_long_term`, `compute_mu_medium_term`, etc.\n"
            "  2. Add one function per additional metric requested in the plan.\n"
            "  3. CRITICAL: `computed_metrics` MUST contain an entry for EVERY metric "
            "you compute, INCLUDING mu and sigma. Do NOT omit them.\n"
            "     In `main()`, for EVERY metric (including mu and sigma):\n"
            "       a. Call the metric function to get its value.\n"
            "       b. Build its trace: `trace = build_computation_trace(func, inputs, value)`\n"
            "       c. Build its metric: `metric = build_computed_metric(name, value, trace)`\n"
            "       d. Append trace to `computation_traces` and metric to `computed_metrics`.\n"
        )

        # Dynamically insert the correct output format based on active_horizons
        if isinstance(active_horizons, (list, tuple)) and len(active_horizons) > 1:
            prompt += (
                "  5. Print the final result using the MULTI-HORIZON format:\n"
                "       result = {\n"
                '           "horizons": {\n'
            )
            for h in active_horizons:
                prompt += (
                    f'               "{h}": {{\n'
                    f'                   "mu": <float>,\n'
                    f'                   "mu_trace_id": "<uuid4>",\n'
                    f'                   "sigma": <float>,\n'
                    f'                   "sigma_trace_id": "<uuid4>"\n'
                    f"               }},\n"
                )
            prompt += (
                "           },\n"
                '           "computed_metrics": computed_metrics,   # each entry MUST include "term": "long_term" etc.\n'
                '           "computation_traces": computation_traces,\n'
                '           "metrics_selected": metrics_selected,\n'
                "       }\n"
                "       print(json.dumps(result, indent=2))\n"
            )
        else:
            h = (
                active_horizons[0]
                if isinstance(active_horizons, (list, tuple))
                and len(active_horizons) == 1
                else "long_term"
            )
            prompt += (
                "  5. Print the final result using the SINGLE-HORIZON format:\n"
                "       result = {\n"
                '           "mu": mu_val,\n'
                '           "mu_trace_id": mu_trace["trace_id"],\n'
                '           "sigma": sigma_val,\n'
                '           "sigma_trace_id": sigma_trace["trace_id"],\n'
                '           "computed_metrics": computed_metrics,\n'
                '           "computation_traces": computation_traces,\n'
                '           "metrics_selected": metrics_selected,\n'
                "       }\n"
                "       print(json.dumps(result, indent=2))\n"
            )

        prompt += (
            "  6. BEFORE declaring success, verify that `computed_metrics` contains "
            "entries for `mu` and `sigma` (and any other metrics you computed).\n"
            "     If they are missing, add them — do NOT return an empty computed_metrics list.\n"
            "  7. Nothing else should be printed to stdout.\n"
            f"ALL file paths MUST be absolute under {work_dir}.\n"
            f"Iterate with Bash until `python3 {script_path}` exits 0 with no errors."
        )

        system_prompt = self._build_system_prompt(short_domain, work_dir=work_dir)
        self._log_block("SYSTEM PROMPT", system_prompt)
        self._log_block("USER PROMPT", prompt)

        # ── Build the claude CLI command ──────────────────────────────────────
        allowed_tools = ",".join(
            [
                "Bash",
                "Read",
                "Write",
                "Edit",
                "WebSearch",
                "WebFetch",
                "Skill",
            ]
        )
        disallowed_tools = ",".join(
            [
                "Task",
                "TaskOutput",
                "TaskStop",
                "Glob",
                "Grep",
                "NotebookEdit",
                "TodoWrite",
                "AskUserQuestion",
                "EnterPlanMode",
                "ExitPlanMode",
                "EnterWorktree",
                "ExitWorktree",
                "CronCreate",
                "CronDelete",
                "CronList",
                "RemoteTrigger",
            ]
        )
        # ollama launch claude --model <model> -- <claude-cli-args> <prompt>
        # Everything after "--" is forwarded verbatim to the claude CLI.
        # prompt MUST be the final positional argument after all flags.
        # verbose=True  → stream-json so tokens print live
        # verbose=False → default text output, read all at once
        cmd = [
            "ollama",
            "launch",
            "claude",
            "--model",
            self.model,
            "--",
            "-p",
            "--bare",
            "--append-system-prompt",
            system_prompt,
            "--allowed-tools",
            allowed_tools,
            "--disallowed-tools",
            disallowed_tools,
            "--permission-mode",
            "bypassPermissions",
            "--no-session-persistence",
            "--add-dir",
            work_dir,
        ]
        # Always use --verbose + stream-json so the CLI operates in
        # headless streaming mode regardless of self.verbose.
        # self.verbose only controls whether events are printed to stdout.
        cmd += ["--verbose", "--output-format", "stream-json"]
        if self.verbose:
            cmd.append("--verbose")
        cmd.append(prompt)  # positional prompt arg — must be last

        max_retries = 2  # retry up to 2 times on timeout before giving up
        result_text = ""
        deadline = self.timeout * self.max_iterations * 6
        # Per-line idle timeout: if the subprocess produces no output for this
        # long, assume it is hung and kill it.  This guards against the
        # blocking-readline problem where proc.wait(timeout=...) is never
        # reached because the for-loop on stdout never exits.
        line_idle_timeout = self.timeout * 4  # e.g. 120s × 4 = 480s per line

        for attempt in range(1, max_retries + 2):  # 1..max_retries+1
            t_start = time.monotonic()
            self._log(
                "CLI",
                f"Launching: ollama launch claude --model {self.model}  "
                f"cwd={work_dir}  attempt={attempt}/{max_retries + 1}",
            )

            timed_out = False
            try:
                with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    cwd=work_dir,
                ) as proc:
                    try:
                        # ── Reader thread: push stdout lines onto a queue ────
                        line_q: queue.Queue[str | None] = queue.Queue()

                        def _reader() -> None:
                            try:
                                for raw_bytes in proc.stdout:
                                    if isinstance(raw_bytes, bytes):
                                        line_q.put(
                                            raw_bytes.decode("utf-8", errors="replace")
                                        )
                                    else:
                                        line_q.put(raw_bytes)
                            finally:
                                line_q.put(None)  # sentinel: EOF

                        reader_t = threading.Thread(target=_reader, daemon=True)
                        reader_t.start()

                        # ── Stream-JSON: parse each event ────────────────────
                        while True:
                            try:
                                raw_line = line_q.get(timeout=line_idle_timeout)
                            except queue.Empty:
                                # No output for line_idle_timeout — subprocess hung
                                self._log(
                                    "CLI",
                                    f"No output for {line_idle_timeout}s — "
                                    f"killing subprocess (attempt {attempt})",
                                )
                                timed_out = True
                                break
                            if raw_line is None:
                                break  # EOF — subprocess closed stdout

                            # Also enforce overall wall-clock deadline
                            if time.monotonic() - t_start > deadline:
                                self._log(
                                    "CLI",
                                    f"Wall-clock deadline {deadline}s exceeded — "
                                    f"killing subprocess (attempt {attempt})",
                                )
                                timed_out = True
                                break

                            line = raw_line.rstrip("\n")
                            try:
                                evt = json.loads(line)
                            except json.JSONDecodeError:
                                self._log("STREAM", line)
                                continue

                            evt_type = evt.get("type", "")
                            if evt_type == "result":
                                result_text = evt.get("result", "")
                                stop_reason = evt.get("stop_reason", "")
                                is_error = evt.get("is_error", False)
                                self._log(
                                    "RESULT",
                                    f"is_error={is_error}  "
                                    f"stop_reason={stop_reason}  "
                                    f"turns={evt.get('num_turns')}",
                                )
                                # If the assistant finished normally there is no
                                # reason to keep the subprocess alive — it may idle
                                # for minutes before closing stdout. Break immediately
                                # and let proc.wait(timeout=30) clean up.
                                if stop_reason == "end_turn" and not is_error:
                                    break
                            elif evt_type in ("system", "user"):
                                self._log(
                                    f"EVT:{evt_type}",
                                    json.dumps(evt, default=str),
                                )
                            else:
                                self._log(f"EVT:{evt_type}", line)

                        if timed_out:
                            proc.kill()
                            _, stderr_out = proc.communicate()
                            elapsed = time.monotonic() - t_start
                            self._log("CLI", f"Timed out after {elapsed:.1f}s")
                            if stderr_out:
                                self._log("CLI", f"stderr: {stderr_out}")
                        else:
                            proc.wait(timeout=30)
                            elapsed = time.monotonic() - t_start
                            self._log(
                                "CLI", f"Done in {elapsed:.1f}s  rc={proc.returncode}"
                            )
                            if proc.returncode != 0:
                                stderr_out = proc.stderr.read()
                                if stderr_out:
                                    self._log("CLI", f"stderr: {stderr_out}")

                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()  # drain pipes so __exit__ doesn't deadlock
                        elapsed = time.monotonic() - t_start
                        self._log("CLI", f"Timed out after {elapsed:.1f}s")
                        timed_out = True

            except Exception as exc:
                self._log("CLI", f"Subprocess error: {exc}")
                timed_out = True

            # If we got a result or didn't time out, we're done
            if not timed_out or result_text:
                total_wall = time.monotonic() - t_start
                self._log("DONE", f"Total wall time: {total_wall:.1f}s")
                break

            # Timed out with no result — retry if attempts remain
            remaining = max_retries + 1 - attempt
            if remaining > 0:
                self._log(
                    "CLI",
                    f"Timeout with no result — retrying ({remaining} attempt(s) left)",
                )
            else:
                self._log("CLI", "All retry attempts exhausted — giving up")

        total_wall = time.monotonic() - t_start
        self._log("DONE", f"Total wall time: {total_wall:.1f}s")

        return result_text, work_dir

    # ── Public interface ──────────────────────────────────────────────────────

    async def aexecute_plan(
        self, computation_plan: dict, data: dict, domain_prompt: str | None = None
    ) -> dict:
        """
        Async version of execute_plan.  Use this when calling from an async
        context (e.g. async LangGraph nodes).

        Runs the blocking ``claude --print`` subprocess in a thread executor so
        it does not block the event loop.

        Result extraction uses a two-tier strategy:
          1. Re-run metrics.py deterministically and parse its stdout.
          2. Scan the claude CLI result_text for embedded JSON.

        Parameters
        ----------
        computation_plan : dict
        data : dict
        domain_prompt : str, optional

        Returns
        -------
        dict — see module docstring for the full schema.
        """
        try:
            result_text, work_dir = await asyncio.to_thread(
                self._run_agent, computation_plan, data, domain_prompt
            )
        except Exception as exc:
            import traceback

            self._log_block("ERROR — claude CLI exception", traceback.format_exc())
            self._current_work_dir = ""
            return self._failure_fallback(f"claude CLI error: {exc}")

        result = self._extract_from_result(result_text, work_dir)
        self._current_work_dir = ""
        return result

    def execute_plan(
        self, computation_plan: dict, data: dict, domain_prompt: str | None = None
    ) -> dict:
        """
        Synchronous entry point — for use from synchronous LangGraph nodes or
        plain Python callers.

        Parameters
        ----------
        computation_plan : dict
        data : dict
        domain_prompt : str, optional

        Returns
        -------
        dict with keys:
          mu                  : float
          sigma               : float
          computed_metrics    : list[dict]   [{metric_name, value}, ...]
          computation_traces  : list[dict]   [{trace_id, code, inputs, output}, ...]
          code_succeeded      : bool
        """
        try:
            result_text, work_dir = self._run_agent(
                computation_plan, data, domain_prompt
            )
        except Exception as exc:
            self._current_work_dir = ""
            return self._failure_fallback(f"claude CLI error: {exc}")
        result = self._extract_from_result(result_text, work_dir)
        self._current_work_dir = ""
        return result

    def _extract_from_result(self, result_text: str, work_dir: str) -> dict:
        """
        Two-tier extraction shared by execute_plan and aexecute_plan.

        Tier 1: re-run metrics.py deterministically.
        Tier 2: scan result_text for embedded JSON.
        """
        self._log(
            "EXTRACT", f"result_text={len(result_text)} chars  work_dir={work_dir}"
        )
        # ── Guard: detect unmodified stub ──────────────────────────────────
        metrics_path = os.path.join(work_dir, "metrics.py")
        is_stub = False
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as fh:
                    src = fh.read()
                is_stub = "# TODO: replace with real implementation" in src
            except Exception:
                pass
        if is_stub:
            self._log(
                "EXTRACT",
                "metrics.py is still the unmodified stub — agent never wrote code",
            )
            return self._failure_fallback(
                "Code agent did not write metrics.py (still stub)"
            )

        # ── Tier 1: re-run metrics.py deterministically ──────────────────────
        stdout, stderr, rc = self._rerun_metrics_file(work_dir)
        self._log_block("TIER1 — re-run metrics.py", f"rc={rc}")
        self._log_block("TIER1 stdout", stdout or "(empty)")
        self._log_block("TIER1 stderr", stderr or "(empty)")
        if rc == 0:
            parsed = self._extract_result_json(stdout)
            if parsed is not None:
                self._log("EXTRACT", f"Tier 1 SUCCESS: {list(parsed.keys())}")
                return self._validate(parsed)
            self._log("TIER1", "rc=0 but JSON extraction failed")

        # ── Tier 2: scan claude CLI result_text for embedded JSON ─────────────
        self._log_block("TIER2 — result_text", result_text or "(empty)")
        parsed = self._extract_result_json(result_text)
        if parsed is not None:
            self._log("EXTRACT", f"Tier 2 SUCCESS: {list(parsed.keys())}")
            return self._validate(parsed)

        self._log_block(
            "ALL TIERS FAILED",
            f"Tier 1: rc={rc}\nstderr={stderr}\nstdout={stdout}\n\n"
            f"Tier 2: result_text={result_text}",
        )
        return self._failure_fallback(
            f"All extraction tiers failed. "
            f"metrics.py rc={rc}, stderr: {stderr}, stdout: {stdout}"
        )
