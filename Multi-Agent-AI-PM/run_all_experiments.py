#!/usr/bin/env python3
"""
Sequential experiment runner.
Runs all 8 experiment phases one by one, waiting for each to complete.
Captures stdout/stderr to per-phase log files in experiment_results/.

Usage:
    python run_all_experiments.py
"""

import atexit
import datetime
import os
import pathlib
import signal
import subprocess
import sys


# Active child phase Popen — tracked so signal handlers can tear down the
# entire phase process tree (workers + ollama launch + claude --bare).
_active_proc: subprocess.Popen | None = None


def _kill_active_proc(sig: int = signal.SIGTERM) -> None:
    proc = _active_proc
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _signal_handler(signum: int, _frame) -> None:
    _kill_active_proc(signal.SIGTERM)
    # Give the phase a moment to do its own cleanup, then escalate.
    proc = _active_proc
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _kill_active_proc(signal.SIGKILL)
    sys.exit(128 + signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(lambda: _kill_active_proc(signal.SIGKILL))

EXPERIMENTS = [
    (
        "Phase 5: System Backtest",
        [
            sys.executable,
            "-m",
            "src.backtest.system_backtest",
            "--ticker",
            "AAPL,XOM,JNJ",
            "--workers",
            "3",
            "--research-depth",
            "shallow",
            "--results-dir",
            "experiment_results/system_backtest",
        ],
        "experiment_results/backtest_system.log",
    ),
    (
        "Phase 6: Fundamental Analyst Backtest",
        [
            sys.executable,
            "-m",
            "src.backtest.fundamentals_analyst_backtest",
            "--workers",
            "8",
            "--depth",
            "shallow",
        ],
        "experiment_results/backtest_fundamental.log",
    ),
    (
        "Phase 7: Market Analyst Backtest",
        [
            sys.executable,
            "-m",
            "src.backtest.market_analyst_backtest",
            "--workers",
            "8",
            "--depth",
            "shallow",
        ],
        "experiment_results/backtest_market.log",
    ),
    (
        "Phase 8: News Analyst Backtest",
        [
            sys.executable,
            "-m",
            "src.backtest.news_analyst_backtest",
            "--workers",
            "8",
            "--depth",
            "shallow",
        ],
        "experiment_results/backtest_news.log",
    ),
]


def run_phase(name: str, cmd: list[str], log_path: str) -> bool:
    """Run a single phase, capture output to log file, wait for completion."""
    print(f"\n{'=' * 70}")
    print(f"  STARTING {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_path}")
    print(f"  Time:    {datetime.datetime.now().isoformat()}")
    print(f"{'=' * 70}\n")

    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        log_file.write(f"# {name}\n")
        log_file.write(f"# Command: {' '.join(cmd)}\n")
        log_file.write(f"# Started: {datetime.datetime.now().isoformat()}\n")
        log_file.flush()

        global _active_proc
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        _active_proc = proc

        try:
            # Stream output to both console and log file in real time
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

            proc.wait()
        finally:
            _active_proc = None

        log_file.write(f"\n# Finished: {datetime.datetime.now().isoformat()}\n")
        log_file.write(f"# Exit code: {proc.returncode}\n")

    if proc.returncode != 0:
        print(f"\n  ERROR: {name} failed with exit code {proc.returncode}")
        return False

    print(f"\n  COMPLETED {name}")
    print(f"  Time: {datetime.datetime.now().isoformat()}")
    return True


def main():
    print("=" * 70)
    print("  VALOR EXPERIMENT SUITE — SEQUENTIAL RUNNER")
    print("  Running 8 phases sequentially. No timeouts.")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for i, (name, cmd, log_path) in enumerate(EXPERIMENTS, start=1):
        ok = run_phase(name, cmd, log_path)
        if ok:
            success_count += 1
        else:
            fail_count += 1
            print(f"\n  Phase {i} failed. Continuing to next phase...")

    print(f"\n{'=' * 70}")
    print("  ALL PHASES COMPLETE")
    print(f"  Success: {success_count}/{len(EXPERIMENTS)}")
    print(f"  Failed:  {fail_count}/{len(EXPERIMENTS)}")
    print(f"  Time:    {datetime.datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
