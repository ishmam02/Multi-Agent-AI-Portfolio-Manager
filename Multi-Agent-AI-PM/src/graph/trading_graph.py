# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
import time
from datetime import date
from typing import Dict, Any, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from langgraph.prebuilt import ToolNode

from src.llm_clients import create_llm_client
from src.llm_clients.rate_limiter import create_rate_limiter

from src.agents import *
from src.default_config import DEFAULT_CONFIG
from src.agents.utils.memory import FinancialSituationMemory
from src.agents.utils.agent_states import (
    AgentState,
)
from src.dataflows.config import set_config
from src.graph.signal_processing import SignalProcessor


# Import the new abstract tool methods from agent_utils
from src.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor
from src.graph.stock_screener import StockScreener
from src.dataflows.alpaca import get_open_positions


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        risk_profile=None,
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            risk_profile: User's investment risk profile dict
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []
        self.risk_profile = risk_profile

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        # Add rate limiter if configured (shared across both LLMs)
        rpm = self.config.get("rate_limit_rpm")
        if rpm:
            llm_kwargs["rate_limiter"] = create_rate_limiter(rpm)

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        # Initialize memories
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.trader_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict
        self._log_lock = threading.Lock()

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}

        max_retries = self.config.get("max_retries")
        if max_retries is not None:
            kwargs["max_retries"] = max_retries

        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, self.risk_profile
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["trader_investment_plan"])

    def propose_trade(
        self, full_signal: str, ticker: str, api_key: str, secret_key: str
    ) -> dict:
        """Propose a trade on Alpaca based on the analysis signal.

        Returns:
            Dict with 'reasoning' (str) and 'order_params' (dict | None).
        """
        return self.signal_processor.propose_trade(
            full_signal, ticker, api_key, secret_key
        )

    @staticmethod
    def execute_order(order_params: dict, api_key: str, secret_key: str) -> str:
        """Execute a previously proposed order on Alpaca.

        Returns:
            Order execution result string.
        """

        return SignalProcessor.execute_order(order_params, api_key, secret_key)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if an exception is a rate-limit / 429 error."""
        msg = str(exc).lower()
        return any(
            hint in msg
            for hint in ("rate limit", "rate_limit", "too many requests", "429")
        )

    def propagate_multi(
        self, tickers: List[str], trade_date: str
    ) -> Dict[str, Dict[str, Any]]:
        """Run the full agent graph on multiple tickers in parallel.

        Uses ThreadPoolExecutor to run graph.invoke() for each ticker
        with concurrency capped by config["max_concurrent_tickers"].
        Rate-limited tickers are retried with exponential backoff.

        Returns:
            Dict mapping ticker -> final_state
        """
        results: Dict[str, Dict[str, Any]] = {}
        lock = threading.Lock()
        max_workers = min(self.config.get("max_concurrent_tickers", 5), len(tickers))
        max_retries = self.config.get("max_retries", 3)

        def _run_single(ticker: str) -> tuple:
            for attempt in range(max_retries):
                try:
                    init_state = self.propagator.create_initial_state(
                        ticker, trade_date, self.risk_profile
                    )
                    args = self.propagator.get_graph_args(callbacks=self.callbacks)
                    final_state = self.graph.invoke(init_state, **args)
                    self._log_state(trade_date, final_state, ticker=ticker)
                    return ticker, final_state
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < max_retries - 1:
                        wait = 10 * (attempt + 1)
                        print(
                            f"[{ticker}] Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        raise
            # unreachable, but satisfies type checker
            raise RuntimeError(f"{ticker}: retries exhausted")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    t, state = future.result()
                    with lock:
                        results[t] = state
                except Exception as e:
                    print(f"Error analyzing {ticker}: {e}")
                    with lock:
                        results[ticker] = {"error": str(e)}

        return results

    def propagate_multi_streaming(
        self,
        tickers: List[str],
        trade_date: str,
        on_chunk_callback=None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run graph on multiple tickers in parallel with streaming for CLI.

        Args:
            tickers: List of ticker symbols.
            trade_date: Trading date string.
            on_chunk_callback: Called as on_chunk_callback(ticker, chunk)
                for each streamed chunk. Must be thread-safe.

        Returns:
            Dict mapping ticker -> final_state
        """
        results: Dict[str, Dict[str, Any]] = {}
        lock = threading.Lock()
        max_workers = min(self.config.get("max_concurrent_tickers", 5), len(tickers))

        max_retries = self.config.get("max_retries", 3)

        def _stream_single(ticker: str) -> tuple:
            for attempt in range(max_retries):
                try:
                    init_state = self.propagator.create_initial_state(
                        ticker, trade_date, self.risk_profile
                    )
                    args = self.propagator.get_graph_args(callbacks=self.callbacks)
                    trace = []
                    for chunk in self.graph.stream(init_state, **args):
                        if on_chunk_callback:
                            on_chunk_callback(ticker, chunk)
                        trace.append(chunk)
                    final_state = trace[-1] if trace else {}
                    self._log_state(trade_date, final_state, ticker=ticker)
                    return ticker, final_state
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < max_retries - 1:
                        wait = 10 * (attempt + 1)
                        print(
                            f"[{ticker}] Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        raise
            raise RuntimeError(f"{ticker}: retries exhausted")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_stream_single, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    t, state = future.result()
                    with lock:
                        results[t] = state
                except Exception as e:
                    print(f"Error streaming {ticker}: {e}")
                    with lock:
                        results[ticker] = {"error": str(e)}

        return results

    def discover_and_analyze(
        self,
        trade_date: str,
        api_key: str = None,
        secret_key: str = None,
        num_picks: int = 5,
    ) -> Dict[str, Any]:
        """Full pipeline: screen -> parallel research -> portfolio decisions.

        Args:
            trade_date: Trading date.
            api_key: Alpaca API key (for fetching portfolio).
            secret_key: Alpaca secret key.
            num_picks: Number of stocks to discover.

        Returns:
            Dict with 'discovered_tickers', 'analysis_results', and
            'portfolio_trades' keys.
        """

        # Get current portfolio from Alpaca
        portfolio_tickers = []
        if api_key and secret_key:
            try:
                positions = get_open_positions(api_key, secret_key)
                portfolio_tickers = [p["symbol"] for p in positions]
            except Exception as e:
                print(f"Could not fetch Alpaca positions: {e}")

        # Screen
        screener = StockScreener(self.config, llm=self.quick_thinking_llm)
        discovered = screener.screen_universe(
            portfolio_tickers,
            trade_date,
            num_picks=num_picks,
            risk_profile=self.risk_profile,
        )

        if not discovered:
            return {
                "discovered_tickers": [],
                "analysis_results": {},
                "portfolio_trades": None,
            }

        # Parallel research
        analysis_results = self.propagate_multi(discovered, trade_date)

        # Portfolio-level trade decisions
        portfolio_trades = None
        if api_key and secret_key:
            ticker_signals = {
                t: state.get("trader_investment_plan", "")
                for t, state in analysis_results.items()
                if "error" not in state
            }
            if ticker_signals:
                portfolio_trades = self.signal_processor.propose_portfolio_trades(
                    ticker_signals, api_key, secret_key
                )

        return {
            "discovered_tickers": discovered,
            "analysis_results": analysis_results,
            "portfolio_trades": portfolio_trades,
        }

    def _log_state(self, trade_date, final_state, ticker=None):
        """Log the final state to a JSON file.

        Args:
            trade_date: The trading date.
            final_state: The final state dict from graph execution.
            ticker: Ticker symbol for file path. Falls back to self.ticker.
        """
        ticker = ticker or self.ticker

        log_entry = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "trader_investment_plan": final_state["trader_investment_plan"],
        }

        with self._log_lock:
            self.log_states_dict[str(trade_date)] = log_entry

            # Save to file (per-ticker directory)
            if ticker:
                directory = Path(f"eval_results/{ticker}/TradingAgentsStrategy_logs/")
                directory.mkdir(parents=True, exist_ok=True)
                with open(directory / f"full_states_log_{trade_date}.json", "w") as f:
                    json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
