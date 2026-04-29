import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "ollama",
    "deep_think_llm": "minimax-m2.7:cloud",
    "quick_think_llm": "minimax-m2.7:cloud",
    "backend_url": "http://localhost:11434/v1",
    "max_retries": 10,
    "rate_limit_rpm": 60,  # Max LLM requests per minute (None = unlimited)
    # Provider-specific thinking configuration
    "google_thinking_level": None,  # "high", "minimal", etc.
    "openai_reasoning_effort": None,  # "medium", "high", "low"
    # Code agent settings
    "code_agent_model": "minimax-m2.7:cloud",
    "code_agent_base_url": "http://localhost:11434",
    "code_agent_timeout": 60,
    "code_agent_max_iterations": 5,
    # Research depth and horizons
    "research_depth": "shallow",
    "horizons_enabled": ["long_term", "medium_term", "short_term"],
    "horizon_lookback": {"long_term": 365, "medium_term": 90, "short_term": 10},
    # Cross-ticker covariance default correlation
    "cross_ticker_correlation": 0.3,
    # Graph execution settings
    "max_recur_limit": 100,
    "max_concurrent_tickers": 5,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",  # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",  # Options: alpha_vantage, yfinance, sec_edgar
        "news_data": "alpaca",  # Options: alpaca, alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Use SEC EDGAR for deep historical statements; yfinance for snapshot & earnings dates
        "get_income_statement": "sec_edgar",
        "get_balance_sheet": "sec_edgar",
        "get_cashflow": "sec_edgar",
    },
}
