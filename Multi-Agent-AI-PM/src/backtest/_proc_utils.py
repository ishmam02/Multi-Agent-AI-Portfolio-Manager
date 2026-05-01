"""
Process-group cleanup helpers for backtest entry points.

Backtests fan out via ``ProcessPoolExecutor`` and each worker eventually spawns
``ollama launch claude`` subprocesses through the code agent.  Killing only the
parent leaks the worker tree (workers become PPID=1 orphans and keep spawning
new claude subagents that create result directories).

The helpers here put each worker into its own process group so the entire
worker → ollama → claude tree can be killed with a single ``killpg``, and
install signal/atexit handlers on the main process that do exactly that on
SIGINT/SIGTERM/normal exit.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor


def init_worker_process_group() -> None:
    """ProcessPoolExecutor ``initializer`` — make each worker its own group leader.

    All subprocess.Popen children spawned by the worker (ollama launch claude,
    and in turn claude --bare) inherit this group, so a single ``killpg`` on the
    worker's PID tears down the whole subtree.
    """
    try:
        os.setpgrp()
    except OSError:
        pass


def _kill_executor_workers(executor: ProcessPoolExecutor, sig: int = signal.SIGKILL) -> None:
    """Send ``sig`` to every worker's process group.  Safe to call repeatedly."""
    procs = getattr(executor, "_processes", None)
    if not procs:
        return
    for proc in list(procs.values()):
        pid = getattr(proc, "pid", None)
        if pid is None:
            continue
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def install_executor_cleanup(executor: ProcessPoolExecutor) -> None:
    """Install SIGINT/SIGTERM/atexit handlers that kill the worker tree."""
    cleaned = {"done": False}

    def _cleanup() -> None:
        if cleaned["done"]:
            return
        cleaned["done"] = True
        _kill_executor_workers(executor, signal.SIGTERM)
        _kill_executor_workers(executor, signal.SIGKILL)

    def _signal_handler(signum: int, _frame) -> None:
        _cleanup()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup)
