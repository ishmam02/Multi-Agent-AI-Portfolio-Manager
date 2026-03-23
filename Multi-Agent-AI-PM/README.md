# Multi-Agent AI Portfolio Manager

A multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents — from fundamental analysts, sentiment experts, and technical analysts, to trader and risk management teams — the platform collaboratively evaluates market conditions and informs trading decisions. Agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> This framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. It is not intended as financial, investment, or trading advice.

---

## Framework Overview

The framework decomposes complex trading tasks into specialized roles, ensuring a robust and scalable approach to market analysis and decision-making.

### Analyst Team

- **Fundamentals Analyst**: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- **Sentiment Analyst**: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- **News Analyst**: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- **Technical Analyst**: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team

- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent

- Composes reports from the analysts and researchers to make informed trading decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager

- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

---

## Installation and CLI

### Installation

Clone the repository:

```bash
git clone https://github.com/ishmamkabir/Multi-Agent-AI-Portfolio-Manager.git
cd Multi-Agent-AI-Portfolio-Manager
```

Create a virtual environment:

```bash
conda create -n ai-portfolio-manager python=3.13
conda activate ai-portfolio-manager
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Required APIs

The framework supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export OPENROUTER_API_KEY=...      # OpenRouter
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
```

For local models, configure Ollama with `llm_provider: "ollama"` in your config.

Alternatively, copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### CLI Usage

Run the CLI directly:

```bash
python -m cli.main
```

A screen will appear where you can select your desired tickers, date, LLMs, research depth, etc.

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

---

## Package Usage

### Implementation Details

Built with LangGraph for flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, OpenRouter, and Ollama.

### Python Usage

Import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function returns a decision:

```python
from src.graph.trading_graph import TradingAgentsGraph
from src.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

You can also customize the configuration:

```python
from src.graph.trading_graph import TradingAgentsGraph
from src.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # openai, google, anthropic, xai, openrouter, ollama
config["deep_think_llm"] = "gpt-5.2"     # Model for complex reasoning
config["quick_think_llm"] = "gpt-5-mini" # Model for quick tasks
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

---

## Contributing

Contributions are welcome! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better.
