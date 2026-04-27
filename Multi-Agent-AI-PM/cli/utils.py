import questionary
from typing import List, Optional, Tuple, Dict
from rich.console import Console

from cli.models import AnalystType, RiskProfile


console = Console()

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: (
            validate_date(x.strip())
            or "Please enter a valid date in YYYY-MM-DD format."
        ),
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("Shallow - Quick research with limited indicator coverage", 1),
        ("Medium - Balanced research depth and indicator coverage", 3),
        ("Deep - Comprehensive research with full indicator coverage", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    # Define shallow thinking llm engine options with their corresponding model names
    SHALLOW_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5 Mini - Cost-optimized reasoning", "gpt-5-mini"),
            ("GPT-5 Nano - Ultra-fast, high-throughput", "gpt-5-nano"),
            ("GPT-5.2 - Latest flagship", "gpt-5.2"),
            ("GPT-5.1 - Flexible reasoning", "gpt-5.1"),
            ("GPT-4.1 - Smartest non-reasoning, 1M context", "gpt-4.1"),
        ],
        "anthropic": [
            ("Claude Haiku 4.5 - Fast + extended thinking", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - Best for agents/coding", "claude-sonnet-4-5"),
            ("Claude Sonnet 4 - High-performance", "claude-sonnet-4-20250514"),
        ],
        "google": [
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, recommended", "gemini-2.5-flash"),
            ("Gemini 3 Pro - Reasoning-first", "gemini-3-pro-preview"),
            ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "xai": [
            (
                "Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx",
                "grok-4-1-fast-non-reasoning",
            ),
            (
                "Grok 4 Fast (Non-Reasoning) - Speed optimized",
                "grok-4-fast-non-reasoning",
            ),
            (
                "Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx",
                "grok-4-1-fast-reasoning",
            ),
            ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
        ],
        "openrouter": [
            (
                "NVIDIA Nemotron 3 Nano 30B (free)",
                "nvidia/nemotron-3-nano-30b-a3b:free",
            ),
            ("Z.AI GLM 4.5 Air (free)", "z-ai/glm-4.5-air:free"),
        ],
        "ollama": [
            ("Minimax-m2.7", "minimax-m2.7:cloud"),
            ("Kimi-k2-thinking (cloud)", "kimi-k2-thinking:cloud"),
            ("Kimi-k2.5 (cloud)", "kimi-k2.5:cloud"),
            ("Llama3.1:8b (8B, local)", "llama3.1:8b"),
            ("Qwen2.5:7b (7B, local)", "qwen2.5:7b"),
            ("Qwen2.5:1.5b (1.5B, local)", "qwen2.5:1.5b"),
            ("Gemma3:1b (1B, local)", "gemma3:1b"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
        ],
    }

    choice = questionary.select(
        "Select Your [Quick-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]No shallow thinking llm engine selected. Exiting...[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    # Define deep thinking llm engine options with their corresponding model names
    DEEP_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5.2 - Latest flagship", "gpt-5.2"),
            ("GPT-5.1 - Flexible reasoning", "gpt-5.1"),
            ("GPT-5 - Advanced reasoning", "gpt-5"),
            ("GPT-4.1 - Smartest non-reasoning, 1M context", "gpt-4.1"),
            ("GPT-5 Mini - Cost-optimized reasoning", "gpt-5-mini"),
            ("GPT-5 Nano - Ultra-fast, high-throughput", "gpt-5-nano"),
        ],
        "anthropic": [
            ("Claude Sonnet 4.5 - Best for agents/coding", "claude-sonnet-4-5"),
            ("Claude Opus 4.5 - Premium, max intelligence", "claude-opus-4-5"),
            ("Claude Opus 4.1 - Most capable model", "claude-opus-4-1-20250805"),
            ("Claude Haiku 4.5 - Fast + extended thinking", "claude-haiku-4-5"),
            ("Claude Sonnet 4 - High-performance", "claude-sonnet-4-20250514"),
        ],
        "google": [
            ("Gemini 3 Pro - Reasoning-first", "gemini-3-pro-preview"),
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, recommended", "gemini-2.5-flash"),
        ],
        "xai": [
            (
                "Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx",
                "grok-4-1-fast-reasoning",
            ),
            ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
            ("Grok 4 - Flagship model", "grok-4-0709"),
            (
                "Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx",
                "grok-4-1-fast-non-reasoning",
            ),
            (
                "Grok 4 Fast (Non-Reasoning) - Speed optimized",
                "grok-4-fast-non-reasoning",
            ),
        ],
        "openrouter": [
            ("Z.AI GLM 4.5 Air (free)", "z-ai/glm-4.5-air:free"),
            (
                "NVIDIA Nemotron 3 Nano 30B (free)",
                "nvidia/nemotron-3-nano-30b-a3b:free",
            ),
        ],
        "ollama": [
            ("Minimax-m2.7", "minimax-m2.7:cloud"),
            ("Kimi-k2-thinking (cloud)", "kimi-k2-thinking:cloud"),
            ("Kimi-k2.5 (cloud)", "kimi-k2.5:cloud"),
            ("Qwen2.5:7b (7B, local)", "qwen2.5:7b"),
            ("Llama3.1:8b (8B, local)", "llama3.1:8b"),
            ("Qwen2.5:1.5b (1.5B, local)", "qwen2.5:1.5b"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
            ("Gemma3:1b (1B, local)", "gemma3:1b"),
        ],
    }

    choice = questionary.select(
        "Select Your [Deep-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in DEEP_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No deep thinking llm engine selected. Exiting...[/red]")
        exit(1)

    return choice


def select_llm_provider() -> tuple[str, str]:
    """Select the OpenAI api url using interactive selection."""
    # Define OpenAI api options with their corresponding endpoints
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Google", "https://generativelanguage.googleapis.com/v1"),
        ("Anthropic", "https://api.anthropic.com/"),
        ("xAI", "https://api.x.ai/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://localhost:11434/v1"),
    ]

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]no OpenAI backend selected. Exiting...[/red]")
        exit(1)

    display_name, url = choice
    print(f"You selected: {display_name}\tURL: {url}")

    return display_name, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()


def configure_risk_profile() -> RiskProfile:
    """Prompt the user to configure their investment risk profile."""
    style = questionary.Style(
        [
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]
    )
    instruction = "\n- Use arrow keys to navigate\n- Press Enter to select"

    # Q1: Experience — inferred from actual activity, not self-label
    experience = questionary.select(
        "[1/6] How would you describe your investing background?",
        choices=[
            questionary.Choice(
                "No experience — I have never bought a stock, fund, or any investment",
                value="No experience",
            ),
            questionary.Choice(
                "Beginner — I've made a few trades or have a basic retirement/brokerage account",
                value="Beginner",
            ),
            questionary.Choice(
                "Intermediate — I actively follow markets and manage a multi-asset portfolio",
                value="Intermediate",
            ),
            questionary.Choice(
                "Expert — Seasoned investor with experience in advanced strategies (options, leverage, etc.)",
                value="Expert",
            ),
        ],
        instruction=instruction,
        style=style,
    ).ask()
    if experience is None:
        console.print("\n[red]No answer provided. Exiting...[/red]")
        exit(1)

    # Q2: Income — objective replenishment capacity
    income = questionary.select(
        "[2/6] What is your annual household income before taxes?",
        choices=[
            questionary.Choice("Under $25,000", value="Under $25k"),
            questionary.Choice("$25,000 – $49,999", value="$25k-$49k"),
            questionary.Choice("$50,000 – $99,999", value="$50k-$99k"),
            questionary.Choice("$100,000 – $199,999", value="$100k-$199k"),
            questionary.Choice("$200,000 – $499,999", value="$200k-$499k"),
            questionary.Choice("$500,000 or more", value="$500k+"),
        ],
        instruction=instruction,
        style=style,
    ).ask()
    if income is None:
        console.print("\n[red]No answer provided. Exiting...[/red]")
        exit(1)

    # Q3: Net worth — objective loss absorption capacity
    net_worth = questionary.select(
        "[3/6] What is your total net worth? (all assets minus all debts)",
        choices=[
            questionary.Choice("Under $5,000", value="Under $5k"),
            questionary.Choice("$5,000 – $24,999", value="$5k-$24k"),
            questionary.Choice("$25,000 – $49,999", value="$25k-$49k"),
            questionary.Choice("$50,000 – $99,999", value="$50k-$99k"),
            questionary.Choice("$100,000 – $249,999", value="$100k-$249k"),
            questionary.Choice("$250,000 – $499,999", value="$250k-$499k"),
            questionary.Choice("$500,000 – $999,999", value="$500k-$999k"),
            questionary.Choice("$1,000,000 or more", value="$1M+"),
        ],
        instruction=instruction,
        style=style,
    ).ask()
    if net_worth is None:
        console.print("\n[red]No answer provided. Exiting...[/red]")
        exit(1)

    # Q4: Goal — what outcome the investor is optimizing for
    goal = questionary.select(
        "[4/6] What matters most to you about this investment?",
        choices=[
            questionary.Choice(
                "Preserve Capital — protecting what I have comes before any return",
                value="Capital Preservation",
            ),
            questionary.Choice(
                "Balanced Growth — steady appreciation with limited downside exposure",
                value="Balanced Growth",
            ),
            questionary.Choice(
                "Aggressive Growth — maximize long-term returns; I accept significant volatility",
                value="Aggressive Growth",
            ),
        ],
        instruction=instruction,
        style=style,
    ).ask()
    if goal is None:
        console.print("\n[red]No answer provided. Exiting...[/red]")
        exit(1)

    # Q5: Risk tolerance — behavioral scenario reveals true psychological floor
    risk = questionary.select(
        "[5/6] Your portfolio drops 25% in a single month. What do you do?",
        choices=[
            questionary.Choice(
                "Sell everything immediately — I cannot stomach any further loss",
                value="Very Conservative",
            ),
            questionary.Choice(
                "Sell most positions — locking in losses feels safer than further decline",
                value="Conservative",
            ),
            questionary.Choice(
                "Hold and wait — unsettling, but I trust a long-term recovery",
                value="Moderate",
            ),
            questionary.Choice(
                "Hold and likely buy more — dips are opportunities, not threats",
                value="Aggressive",
            ),
            questionary.Choice(
                "Aggressively buy more — large drawdowns are the best entry points",
                value="Very Aggressive",
            ),
        ],
        instruction=instruction,
        style=style,
    ).ask()
    if risk is None:
        console.print("\n[red]No answer provided. Exiting...[/red]")
        exit(1)

    # Q6: Investment period — defines recovery window and strategy horizon
    period = questionary.select(
        "[6/6] When do you expect to need or access this investment?",
        choices=[
            questionary.Choice(
                "Within 1–5 years — short horizon; no time to recover from drawdowns",
                value="1-5 years",
            ),
            questionary.Choice(
                "5–10 years — medium horizon; can ride out one market cycle",
                value="5-10 years",
            ),
            questionary.Choice(
                "10–20 years — long horizon; multiple cycles available to recover",
                value="10-20 years",
            ),
            questionary.Choice(
                "20–40 years — very long; short-term swings are noise",
                value="20-40 years",
            ),
            questionary.Choice(
                "40–60 years — generational; aggressive growth is mathematically favored",
                value="40-60 years",
            ),
            questionary.Choice(
                "60+ years — ultra-long; virtually any drawdown is recoverable",
                value="60+ years",
            ),
        ],
        instruction=instruction,
        style=style,
    ).ask()
    if period is None:
        console.print("\n[red]No answer provided. Exiting...[/red]")
        exit(1)

    return RiskProfile(
        experience=experience,
        income=income,
        net_worth=net_worth,
        goal=goal,
        risk=risk,
        period=period,
    )


def ask_gemini_thinking_config() -> str | None:
    """Ask for Gemini thinking configuration.

    Returns thinking_level: "high" or "minimal".
    Client maps to appropriate API param based on model series.
    """
    return questionary.select(
        "Select Thinking Mode:",
        choices=[
            questionary.Choice("Enable Thinking (recommended)", "high"),
            questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style(
            [
                ("selected", "fg:green noinherit"),
                ("highlighted", "fg:green noinherit"),
                ("pointer", "fg:green noinherit"),
            ]
        ),
    ).ask()


def ask_alpaca_credentials() -> tuple[str, str]:
    """Prompt the user to enter Alpaca Paper Trading API credentials."""
    style = questionary.Style([("text", "fg:yellow"), ("highlighted", "noinherit")])

    api_key = questionary.password(
        "Enter your Alpaca API Key:",
        validate=lambda x: len(x.strip()) > 0 or "API key is required.",
        style=style,
    ).ask()
    if api_key is None:
        console.print("\n[red]No API key provided. Exiting...[/red]")
        exit(1)

    secret_key = questionary.password(
        "Enter your Alpaca Secret Key:",
        validate=lambda x: len(x.strip()) > 0 or "Secret key is required.",
        style=style,
    ).ask()
    if secret_key is None:
        console.print("\n[red]No secret key provided. Exiting...[/red]")
        exit(1)

    return api_key.strip(), secret_key.strip()


def select_code_agent_model(provider: str) -> str:
    """Select code-agent LLM using the same provider-then-model pattern as thinking agents."""
    # Re-use the shallow-thinker options — code agents are fast/cheap models
    SHALLOW_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5 Mini - Cost-optimized reasoning", "gpt-5-mini"),
            ("GPT-5 Nano - Ultra-fast, high-throughput", "gpt-5-nano"),
            ("GPT-5.2 - Latest flagship", "gpt-5.2"),
            ("GPT-5.1 - Flexible reasoning", "gpt-5.1"),
            ("GPT-4.1 - Smartest non-reasoning, 1M context", "gpt-4.1"),
        ],
        "anthropic": [
            ("Claude Haiku 4.5 - Fast + extended thinking", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - Best for agents/coding", "claude-sonnet-4-5"),
            ("Claude Sonnet 4 - High-performance", "claude-sonnet-4-20250514"),
        ],
        "google": [
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, recommended", "gemini-2.5-flash"),
            ("Gemini 3 Pro - Reasoning-first", "gemini-3-pro-preview"),
            ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "xai": [
            (
                "Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx",
                "grok-4-1-fast-non-reasoning",
            ),
            (
                "Grok 4 Fast (Non-Reasoning) - Speed optimized",
                "grok-4-fast-non-reasoning",
            ),
            (
                "Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx",
                "grok-4-1-fast-reasoning",
            ),
            ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
        ],
        "openrouter": [
            (
                "NVIDIA Nemotron 3 Nano 30B (free)",
                "nvidia/nemotron-3-nano-30b-a3b:free",
            ),
            ("Z.AI GLM 4.5 Air (free)", "z-ai/glm-4.5-air:free"),
        ],
        "ollama": [
            ("Minimax-m2.7", "minimax-m2.7:cloud"),
            ("Kimi-k2-thinking (cloud)", "kimi-k2-thinking:cloud"),
            ("Kimi-k2.5 (cloud)", "kimi-k2.5:cloud"),
            ("Llama3.1:8b (8B, local)", "llama3.1:8b"),
            ("Qwen2.5:7b (7B, local)", "qwen2.5:7b"),
            ("Qwen2.5:1.5b (1.5B, local)", "qwen2.5:1.5b"),
            ("Gemma3:1b (1B, local)", "gemma3:1b"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
        ],
    }

    choice = questionary.select(
        "Select Your [Code Agent LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No code agent model selected. Exiting...[/red]")
        exit(1)

    return choice


def ask_horizons() -> List[str]:
    """Prompt user to select which investment horizons to analyze."""
    choices = questionary.checkbox(
        "Select investment time horizons to analyze:",
        choices=[
            questionary.Choice("Long Term (1+ year)", value="long_term", checked=True),
            questionary.Choice(
                "Medium Term (3-12 months)", value="medium_term", checked=True
            ),
            questionary.Choice(
                "Short Term (< 3 months)", value="short_term", checked=True
            ),
        ],
        validate=lambda x: len(x) > 0 or "Select at least one horizon.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No horizons selected. Exiting...[/red]")
        exit(1)

    return choices


def ask_concurrency_limits() -> tuple[int, int, int]:
    """Prompt for max concurrent tickers, rate limit RPM, and max retries."""
    style = questionary.Style([("text", "fg:yellow"), ("highlighted", "noinherit")])

    max_concurrent = questionary.text(
        "Max concurrent tickers (portfolio mode):",
        default="5",
        validate=lambda x: (
            x.strip().isdigit() and 1 <= int(x.strip()) <= 20 or "Enter 1-20."
        ),
        style=style,
    ).ask()
    if max_concurrent is None:
        max_concurrent = "5"

    rate_limit = questionary.text(
        "Rate limit (requests per minute, 0=unlimited):",
        default="60",
        validate=lambda x: (
            x.strip().isdigit() and 0 <= int(x.strip()) <= 1000 or "Enter 0-1000."
        ),
        style=style,
    ).ask()
    if rate_limit is None:
        rate_limit = "60"

    max_retries = questionary.text(
        "Max retries on LLM failure:",
        default="10",
        validate=lambda x: (
            x.strip().isdigit() and 0 <= int(x.strip()) <= 50 or "Enter 0-50."
        ),
        style=style,
    ).ask()
    if max_retries is None:
        max_retries = "10"

    return int(max_concurrent), int(rate_limit), int(max_retries)


def ask_data_vendors() -> Dict[str, str]:
    """Prompt for preferred data vendors."""
    vendor = questionary.select(
        "Preferred stock data vendor:",
        choices=[
            questionary.Choice("yfinance (free, recommended)", value="yfinance"),
            questionary.Choice(
                "Alpha Vantage (requires API key)", value="alpha_vantage"
            ),
        ],
        style=questionary.Style(
            [
                ("selected", "fg:green noinherit"),
                ("highlighted", "fg:green noinherit"),
                ("pointer", "fg:green noinherit"),
            ]
        ),
    ).ask()

    if vendor is None:
        vendor = "yfinance"

    return {
        "core_stock_apis": vendor,
        "technical_indicators": vendor,
        "fundamental_data": vendor,
        "news_data": "alpaca",
    }


def select_analysis_mode() -> str:
    """Select between single stock analysis and portfolio discovery mode."""
    choice = questionary.select(
        "Select Analysis Mode:",
        choices=[
            questionary.Choice(
                "Single Stock — Enter one ticker for deep analysis",
                value="single",
            ),
            questionary.Choice(
                "Portfolio Discovery — Screen, research multiple stocks, and get portfolio-level trades",
                value="portfolio",
            ),
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No mode selected. Exiting...[/red]")
        exit(1)

    return choice


def get_num_picks() -> int:
    """Ask user how many stocks the screener should discover."""
    num = questionary.text(
        "How many stocks should the screener discover?",
        default="5",
        validate=lambda x: (
            (x.strip().isdigit() and 1 <= int(x.strip()) <= 20)
            or "Enter a number between 1 and 20."
        ),
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if num is None:
        console.print("\n[red]No number provided. Exiting...[/red]")
        exit(1)

    return int(num.strip())


def get_portfolio_tickers_manual() -> List[str]:
    """Get portfolio tickers from manual comma-separated input."""
    tickers_str = questionary.text(
        "Enter current portfolio tickers (comma-separated, or press Enter for empty):",
        default="",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if tickers_str is None:
        return []

    tickers_str = tickers_str.strip()
    if not tickers_str:
        return []

    return [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
