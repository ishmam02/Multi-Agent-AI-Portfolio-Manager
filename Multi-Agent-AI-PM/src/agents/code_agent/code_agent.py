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
        timeout: int = 60,
        max_iterations: int = 5,
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

    def _log(self, tag: str, text: str, *, truncate: int = 500) -> None:
        """Print a timestamped log line when verbose=True AND write to file logger."""
        ts = datetime.now().strftime("%H:%M:%S")
        body = text[:truncate] + ("…" if len(text) > truncate else "")

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
                "    result = {\n"
                '        "mu": mu_val,\n'
                '        "mu_trace_id": mu_trace["trace_id"],\n'
                '        "sigma": sigma_val,\n'
                '        "sigma_trace_id": sigma_trace["trace_id"],\n'
                '        "computed_metrics": computed_metrics,\n'
                '        "computation_traces": computation_traces,\n'
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
        else:
            plan_section = (
                f"=== COMPUTATION PLAN ===\n{json.dumps(computation_plan, indent=2)}\n"
            )

        return (
            "Compute the financial metrics defined in the plan below.\n\n"
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
        return isinstance(d, dict) and ("mu" in d or "computed_metrics" in d)

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

        # 3. Forward scan for { positions — keep last valid match.
        #    raw_decode respects JSON string quoting, so braces inside
        #    string values (e.g. full_code_trace.code) are skipped.
        last_valid: dict | None = None
        for m in re.finditer(r"\{", text):
            try:
                d, _ = decoder.raw_decode(text, m.start())
                if isinstance(d, dict) and _valid(d):
                    last_valid = d
            except (json.JSONDecodeError, ValueError):
                continue
        return last_valid

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
        result: dict[str, Any] = {
            "mu": self._safe_float(raw.get("mu"), 0.0),
            "sigma": max(self._safe_float(raw.get("sigma"), 0.01), 1e-6),
            "computed_metrics": metrics,
            "computation_traces": traces,
            "code_succeeded": True,
        }
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
                text=True,
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
        horizon = computation_plan.get("horizon", "unknown")
        work_dir = os.path.join(
            self._project_root,
            "results",
            ticker,
            analysis_date,
            self.analyst_type,
            horizon,
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

        # ── Gather context the agent would normally spend turns discovering ──
        dir_listing = "\n".join(f"  {f}" for f in sorted(os.listdir(work_dir)))

        primary_csv = next(
            (f for f in data_files if f.endswith(".csv") and f != "metrics.py"), None
        )
        data_preview = ""
        if primary_csv:
            try:
                with open(os.path.join(work_dir, primary_csv), encoding="utf-8") as fh:
                    lines = [fh.readline() for _ in range(12)]
                data_preview = (
                    f"\n=== HEAD OF {primary_csv} (first 12 rows) ===\n"
                    + "".join(lines).rstrip()
                )
            except Exception:
                pass

        with open(script_path, encoding="utf-8") as fh:
            scaffold_src = fh.read()

        prompt = (
            self._build_user_prompt(computation_plan, data, data_files=data_files)
            + f"\n\n=== WORKING DIRECTORY: {work_dir} ===\n{dir_listing}"
            + data_preview
            + f"\n\n=== CURRENT {script_path} (scaffold) ===\n{scaffold_src}"
            + f"\n\nIMPORTANT: All files above already exist in {work_dir}. "
            "Do NOT run ls, cat, or head — you already have everything you need above. "
            f"Immediately overwrite {script_path} with the full implementation using a "
            f"heredoc (`cat > {script_path} << 'PYEOF' ... PYEOF`) and run it with "
            f"`python3 {script_path}`.\n"
            "The scaffold already defines these helpers (keep them exactly as-is):\n"
            "  • `build_computation_trace(func, inputs, output)` — captures the "
            "function source and returns a trace dict with a trace_id.\n"
            "  • `build_computed_metric(metric_name, value, trace)` — returns a "
            "computed metric dict linked to the trace.\n\n"
            "It also loads the primary data as `df`.  Your implementation must:\n"
            "  1. Keep `build_computation_trace` and `build_computed_metric` exactly as-is.\n"
            "  2. Replace `compute_mu` and `compute_sigma` with real implementations.\n"
            "  3. Add one function per additional metric requested in the plan.\n"
            "  4. In `main()`, for EVERY metric (including mu and sigma):\n"
            "       a. Call the metric function to get its value.\n"
            "       b. Build its trace: `trace = build_computation_trace(func, inputs, value)`\n"
            "       c. Build its metric: `metric = build_computed_metric(name, value, trace)`\n"
            "       d. Append trace to `computation_traces` and metric to `computed_metrics`.\n"
            "  5. Print the final result using EXACTLY this structure:\n"
            "       result = {\n"
            '           "mu": mu_val,\n'
            '           "mu_trace_id": mu_trace["trace_id"],\n'
            '           "sigma": sigma_val,\n'
            '           "sigma_trace_id": sigma_trace["trace_id"],\n'
            '           "computed_metrics": computed_metrics,\n'
            '           "computation_traces": computation_traces,\n'
            "       }\n"
            "       print(json.dumps(result, indent=2))\n"
            "  6. Nothing else should be printed to stdout.\n"
            f"ALL file paths MUST be absolute under {work_dir}.\n"
            f"Iterate with Bash until `python3 {script_path}` exits 0 with no errors."
        )

        system_prompt = self._build_system_prompt(domain_prompt, work_dir=work_dir)
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
        line_idle_timeout = self.timeout * 5  # e.g. 60s × 5 = 300s per line

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
                    text=True,
                    cwd=work_dir,
                ) as proc:
                    try:
                        # ── Reader thread: push stdout lines onto a queue ────
                        line_q: queue.Queue[str | None] = queue.Queue()

                        def _reader() -> None:
                            try:
                                for raw in proc.stdout:
                                    line_q.put(raw)
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
                                self._log("STREAM", line[:300])
                                continue

                            evt_type = evt.get("type", "")
                            if evt_type == "result":
                                result_text = evt.get("result", "")
                                self._log(
                                    "RESULT",
                                    f"is_error={evt.get('is_error')}  "
                                    f"stop_reason={evt.get('stop_reason')}  "
                                    f"turns={evt.get('num_turns')}",
                                )
                            elif evt_type in ("system", "user"):
                                self._log(
                                    f"EVT:{evt_type}",
                                    json.dumps(evt, default=str)[:300],
                                )
                            else:
                                self._log(f"EVT:{evt_type}", line[:200])

                        if timed_out:
                            proc.kill()
                            _, stderr_out = proc.communicate()
                            elapsed = time.monotonic() - t_start
                            self._log("CLI", f"Timed out after {elapsed:.1f}s")
                            if stderr_out:
                                self._log("CLI", f"stderr: {stderr_out[:500]}")
                        else:
                            proc.wait(timeout=30)
                            elapsed = time.monotonic() - t_start
                            self._log(
                                "CLI", f"Done in {elapsed:.1f}s  rc={proc.returncode}"
                            )
                            if proc.returncode != 0:
                                stderr_out = proc.stderr.read()
                                if stderr_out:
                                    self._log("CLI", f"stderr: {stderr_out[:500]}")

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


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    import sys as _sys

    # Ensure the Multi-Agent-AI-PM directory is on sys.path so that
    # `src.*` imports resolve regardless of how this file is invoked.
    _repo_root = str(pathlib.Path(__file__).resolve().parents[3])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

    from src.agents.utils.core_stock_tools import get_stock_data

    # 1. Fetch AAPL OHLCV data for last year
    print("Fetching AAPL stock data for 2025 ...")
    raw_data = get_stock_data.invoke(
        {
            "symbol": "AAPL",
            "start_date": "2025-01-01",
            "end_date": "2026-03-30",
        }
    )
    print("Data fetched. Preview (first 400 chars):")
    print(str(raw_data)[:400])
    print()

    # 2. Quantitative market analyst computation plan (List[Metrics] format)
    computation_plan = {
        "ticker": "AAPL",
        "analysis_date": "2025-12-31",
        "horizon": "short_term",
        "analyst_type": "market",
        "computation_plan": [
            # ── Trend ────────────────────────────────────────────────────────
            {
                "metric_name": "EMA_20",
                "metric_interpretation": "20-period exponential moving average of Close; short-term trend baseline.",
                "metric_rationale": "Captures near-term price momentum and acts as dynamic support/resistance.",
                "computation_instruction": "df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean(); return float(df['EMA_20'].iloc[-1])",
            },
            {
                "metric_name": "EMA_50",
                "metric_interpretation": "50-period EMA of Close; intermediate-term trend baseline.",
                "metric_rationale": "Key medium-term trend filter used in regime classification.",
                "computation_instruction": "df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean(); return float(df['EMA_50'].iloc[-1])",
            },
            {
                "metric_name": "EMA_200",
                "metric_interpretation": "200-period EMA of Close; long-term trend baseline.",
                "metric_rationale": "Widely-watched level that separates bull/bear market regimes.",
                "computation_instruction": "df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean(); return float(df['EMA_200'].iloc[-1])",
            },
            {
                "metric_name": "ADX_14",
                "metric_interpretation": "14-period Average Directional Index; measures trend strength (0–100), not direction.",
                "metric_rationale": "Distinguishes trending from ranging markets; required for regime classification.",
                "computation_instruction": (
                    "Compute +DM = max(High-prev_High, 0), -DM = max(prev_Low-Low, 0) (zero if +DM > -DM for -DM and vice versa). "
                    "TR = max(High-Low, abs(High-prev_Close), abs(Low-prev_Close)). "
                    "Smooth all three with 14-period Wilder's EMA (ATR-style). "
                    "+DI = 100*smooth_+DM/ATR14, -DI = 100*smooth_-DM/ATR14. "
                    "DX = 100*abs(+DI - -DI)/(+DI + -DI). ADX = 14-period Wilder's EMA of DX. "
                    "Return {'ADX_14': float(adx[-1]), 'ADX_plus_DI': float(plus_di[-1]), 'ADX_minus_DI': float(minus_di[-1])}"
                ),
            },
            {
                "metric_name": "price_vs_EMA50_pct",
                "metric_interpretation": "Percentage deviation of last Close from EMA_50; positive = above trend.",
                "metric_rationale": "Quantifies how extended price is relative to the intermediate trend.",
                "computation_instruction": "return (df['Close'].iloc[-1] / ema_50[-1] - 1) * 100",
            },
            {
                "metric_name": "price_vs_EMA200_pct",
                "metric_interpretation": "Percentage deviation of last Close from EMA_200; positive = above long-term trend.",
                "metric_rationale": "Measures long-term overextension or undervaluation vs. the 200-EMA.",
                "computation_instruction": "return (df['Close'].iloc[-1] / ema_200[-1] - 1) * 100",
            },
            # ── Momentum ─────────────────────────────────────────────────────
            {
                "metric_name": "RSI_14",
                "metric_interpretation": "Wilder's 14-period Relative Strength Index [0,100]; >70 overbought, <30 oversold.",
                "metric_rationale": "Primary momentum oscillator; feeds into composite signal scoring for mu.",
                "computation_instruction": (
                    "delta = df['Close'].diff(). "
                    "gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(). "
                    "loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean(). "
                    "RS = gain / loss.replace(0, 1e-9). return float(100 - 100/(1+RS.iloc[-1]))"
                ),
            },
            {
                "metric_name": "MACD_line",
                "metric_interpretation": "MACD line = EMA(12) - EMA(26) of Close; positive = short-term momentum above medium.",
                "metric_rationale": "Trend-following momentum indicator; histogram sign feeds into composite scoring.",
                "computation_instruction": (
                    "ema12 = df['Close'].ewm(span=12, adjust=False).mean(). "
                    "ema26 = df['Close'].ewm(span=26, adjust=False).mean(). "
                    "macd = ema12 - ema26. signal = macd.ewm(span=9, adjust=False).mean(). "
                    "histogram = macd - signal. "
                    "return {'MACD_line': float(macd.iloc[-1]), 'MACD_signal': float(signal.iloc[-1]), 'MACD_histogram': float(histogram.iloc[-1])}"
                ),
            },
            {
                "metric_name": "stochastic_K",
                "metric_interpretation": "14-period Stochastic %K [0,100] = 100*(C-L14)/(H14-L14); >80 overbought, <20 oversold.",
                "metric_rationale": "Identifies overbought/oversold levels within the current trading range.",
                "computation_instruction": (
                    "L14 = df['Low'].rolling(14).min(). H14 = df['High'].rolling(14).max(). "
                    "K = 100*(df['Close']-L14)/(H14-L14+1e-9). D = K.rolling(3).mean(). "
                    "return {'stochastic_K': float(K.iloc[-1]), 'stochastic_D': float(D.iloc[-1])}"
                ),
            },
            {
                "metric_name": "ROC_10",
                "metric_interpretation": "10-period Rate of Change = (Close[-1]/Close[-11]-1)*100; percent price change over 10 bars.",
                "metric_rationale": "Short-term price momentum proxy; confirms or diverges from RSI signal.",
                "computation_instruction": "return float((df['Close'].iloc[-1] / df['Close'].iloc[-11] - 1) * 100)",
            },
            # ── Volatility ───────────────────────────────────────────────────
            {
                "metric_name": "BB_upper",
                "metric_interpretation": "Bollinger Band upper = 20-SMA + 2*20-day std(Close); dynamic resistance level.",
                "metric_rationale": "Price touching upper band signals overbought condition or strong breakout.",
                "computation_instruction": (
                    "sma20 = df['Close'].rolling(20).mean(). std20 = df['Close'].rolling(20).std(). "
                    "upper = sma20 + 2*std20. lower = sma20 - 2*std20. "
                    "width = (upper-lower)/sma20. pct_b = (df['Close']-lower)/(upper-lower+1e-9). "
                    "return {'BB_upper': float(upper.iloc[-1]), 'BB_lower': float(lower.iloc[-1]), "
                    "'BB_width': float(width.iloc[-1]), 'BB_pct_B': float(pct_b.iloc[-1])}"
                ),
            },
            {
                "metric_name": "ATR_14",
                "metric_interpretation": "14-period Average True Range; absolute daily volatility in price units.",
                "metric_rationale": "Used for position sizing and stop-loss placement; reflects current risk per bar.",
                "computation_instruction": (
                    "tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), "
                    "(df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1). "
                    "return float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])"
                ),
            },
            {
                "metric_name": "HV_20",
                "metric_interpretation": "20-day close-to-close historical volatility, annualised (decimal); e.g. 0.25 = 25% p.a.",
                "metric_rationale": "Baseline realised volatility; feeds into sigma and volatility regime classification.",
                "computation_instruction": (
                    "log_ret = np.log(df['Close']/df['Close'].shift(1)).dropna(). "
                    "return float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))"
                ),
            },
            {
                "metric_name": "garman_klass_vol",
                "metric_interpretation": "Garman-Klass OHLC volatility estimator, annualised; more efficient than close-to-close.",
                "metric_rationale": "Lower-variance vol estimator used as sigma for the output; floor at 0.01.",
                "computation_instruction": (
                    "u = np.log(df['High']/df['Open']). d = np.log(df['Low']/df['Open']). "
                    "c = np.log(df['Close']/df['Open']). "
                    "gk = (0.5*u**2 - (2*np.log(2)-1)*c**2). "
                    "return float(max(np.sqrt(gk.mean()*252), 0.01))"
                ),
            },
            # ── Volume ───────────────────────────────────────────────────────
            {
                "metric_name": "OBV",
                "metric_interpretation": "On-Balance Volume; cumulative volume signed by daily price direction.",
                "metric_rationale": "Detects volume-price divergence and accumulation/distribution pressure.",
                "computation_instruction": (
                    "direction = np.sign(df['Close'].diff().fillna(0)). "
                    "obv = (direction * df['Volume']).cumsum(). "
                    "return float(obv.iloc[-1])"
                ),
            },
            {
                "metric_name": "CMF_20",
                "metric_interpretation": "20-period Chaikin Money Flow [-1,+1]; positive = buying pressure dominates.",
                "metric_rationale": "Confirms OBV signal; both required for volume_regime classification.",
                "computation_instruction": (
                    "mfv = ((df['Close']-df['Low'])-(df['High']-df['Close']))/(df['High']-df['Low']+1e-9)*df['Volume']. "
                    "return float(mfv.rolling(20).sum() / df['Volume'].rolling(20).sum().replace(0,1e-9)).iloc[-1]"
                ),
            },
            # ── Regime classification ─────────────────────────────────────────
            {
                "metric_name": "trend_regime",
                "metric_interpretation": "Categorical trend state: 'strong_uptrend'|'weak_uptrend'|'ranging'|'weak_downtrend'|'strong_downtrend'.",
                "metric_rationale": "Feeds into composite signal score for mu calculation.",
                "computation_instruction": (
                    "adx_val, plus_di, minus_di = ADX_14, ADX_plus_DI, ADX_minus_DI (last values). "
                    "close = df['Close'].iloc[-1]. "
                    "if adx_val > 25 and plus_di > minus_di and close > ema50 and close > ema200: return 'strong_uptrend'. "
                    "elif 20 <= adx_val <= 25 and plus_di > minus_di and close > ema50: return 'weak_uptrend'. "
                    "elif adx_val < 20: return 'ranging'. "
                    "elif 20 <= adx_val <= 25 and minus_di > plus_di and close < ema50: return 'weak_downtrend'. "
                    "else: return 'strong_downtrend'"
                ),
            },
            {
                "metric_name": "volatility_regime",
                "metric_interpretation": "Categorical volatility state: 'low_vol'|'normal_vol'|'high_vol' based on HV_20 tercile.",
                "metric_rationale": "Contextualises current volatility relative to its own 252-day history.",
                "computation_instruction": (
                    "hv_series = log_ret.rolling(20).std() * np.sqrt(252). "
                    "t33, t67 = hv_series.rolling(252).quantile(0.33).iloc[-1], hv_series.rolling(252).quantile(0.67).iloc[-1]. "
                    "hv_now = hv_series.iloc[-1]. "
                    "return 'low_vol' if hv_now < t33 else ('high_vol' if hv_now > t67 else 'normal_vol')"
                ),
            },
            {
                "metric_name": "volume_regime",
                "metric_interpretation": "Categorical volume pressure: 'accumulation'|'distribution'|'neutral'.",
                "metric_rationale": "Volume confirmation for trend signals; feeds into composite score for mu.",
                "computation_instruction": (
                    "obv_slope = np.polyfit(range(20), obv_series.iloc[-20:].values, 1)[0]. "
                    "cmf_val = CMF_20 (last value). "
                    "if obv_slope > 0 and cmf_val > 0.05: return 'accumulation'. "
                    "elif obv_slope < 0 and cmf_val < -0.05: return 'distribution'. "
                    "else: return 'neutral'"
                ),
            },
            # ── mu / sigma ────────────────────────────────────────────────────
            {
                "metric_name": "mu",
                "metric_interpretation": "Annualised expected return derived from composite signal score; range [-0.40, +0.40].",
                "metric_rationale": "Primary output: signal-weighted return estimate for the short-term horizon.",
                "computation_instruction": (
                    "trend_score = +1 if trend_regime in ('strong_uptrend','weak_uptrend') else (-1 if 'downtrend' in trend_regime else 0). "
                    "mom_score = +1 if (50 < rsi <= 70 and macd_hist > 0) else (-1 if (30 <= rsi < 50 and macd_hist < 0) else 0). "
                    "vol_score = +1 if volume_regime=='accumulation' else (-1 if volume_regime=='distribution' else 0). "
                    "composite = trend_score + mom_score + vol_score. "
                    "return composite / 3 * 0.40"
                ),
            },
            {
                "metric_name": "sigma",
                "metric_interpretation": "Annualised volatility contribution; equals garman_klass_vol, floored at 0.01.",
                "metric_rationale": "Represents the uncertainty/risk dimension of the return estimate.",
                "computation_instruction": "return max(garman_klass_vol, 0.01)",
            },
        ],
    }

    # 3. Run the CodeValidationAgent against a local Ollama endpoint
    agent = CodeValidationAgent(
        model="minimax-m2.7:cloud",
        analyst_type="market",
        verbose=True,
    )

    print("Running CodeValidationAgent ...")
    result = agent.execute_plan(computation_plan, {"stock_data": raw_data})

    # 4. Print the result
    print("\n=== CodeValidationAgent Result ===")
    print(json.dumps(result, indent=2))
