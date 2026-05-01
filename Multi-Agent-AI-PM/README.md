# Neuro-Symbolic Multi-Agent Analyst Architecture

A research framework that deploys specialized LLM-powered analyst agents---Fundamental, Market (Technical), and News---to produce multi-horizon conviction signals through a neuro-symbolic workflow. Each analyst follows a three-phase pipeline: qualitative LLM reasoning over raw data, quantitative validation via autonomously generated Python code, and structured report synthesis with per-horizon expected returns and confidence scores. A Synthesis Agent dynamically weights analyst outputs to produce a composite signal.

This codebase accompanies the paper *Large Language Models in Quantitative Finance: A Survey and Empirical Study of Neuro-Symbolic Analyst Agents* and is designed as a structured, reproducible, and auditable research platform for studying how LLMs can augment---not replace---human judgment in financial decision-making.

## Disclaimer

This is a research framework. It is not financial, investment, or trading advice. Trading performance may vary based on model choice, temperature, data quality, and other non-deterministic factors.

---

## Architecture Overview

The system is structured as **Layer 2** of a proposed three-layer architecture. Layer 2 is the online execution layer: it receives a stock ticker and trade date, produces deeply researched multi-horizon conviction signals, and emits a composite BUY/SELL/HOLD decision.

### Analyst Agents

Three specialized analyst agents each produce independent Research Reports:

**Fundamental Analyst**

Reads SEC filings (10-K, 10-Q, 8-K), earnings call transcripts, and analyst consensus estimates. Forms a thesis on intrinsic value relative to market price, identifies catalysts, and assesses management quality through language analysis. Code validation executes: multi-stage DCF, residual income models, comparable company multiples, DuPont decomposition, Altman Z-score, Piotroski F-score, and any active fundamental alpha factors. Output: a Fundamental Research Report containing per-horizon expected returns, confidence intervals, the Investment Thesis, and full computation traces.

**Market Analyst (Technical)**

Interprets chart patterns, support/resistance levels, volume profiles, and market microstructure. Identifies regime context (trending vs. ranging, high vs. low volatility) that determines which technical signals are most relevant. Code validation executes: RSI, MACD, Bollinger Bands, ADX, Ichimoku, VWMA, ATR, realized and implied volatility surface analysis, and options skew signals. Output: per-horizon expected returns, volatility regime classification, and computation traces.

**News Analyst**

Directly reads and interprets news articles, social media sentiment, earnings call transcripts, and analyst reports. There is no separate NLP pipeline: the LLM itself is the sentiment engine. It applies validated sentiment scoring rubrics (structured prompts guiding evaluation of tone, surprise, uncertainty, and materiality). Code validates by cross-referencing LLM sentiment scores against quantitative proxies (abnormal trading volume, options implied volatility changes, short interest shifts). Output: Sentiment Score vectors at each horizon, with the complete citation chain from source article URL through LLM interpretation to validated score.

### Three-Phase Analyst Workflow

Every analyst follows an institutional equity-research workflow inspired by the Program-Aided Language Model (PAL) insight and the CodeAct finding that executable code actions elicit better agent behavior than free-text tool use.

**Phase 1: LLM Reasoning (Neural)**

The LLM reads the relevant data for each stock (SEC filings, price charts, news flow, macro data) and generates a structured Investment Thesis in natural language. This thesis articulates the qualitative case: what is the narrative, what is the catalyst, what could go wrong, and what conviction level does the analyst have? This is the step that pure deterministic code cannot perform.

**Phase 2: Code Validation (Symbolic)**

The LLM generates Python code that quantitatively validates (or invalidates) its own thesis. The code computes DCF valuations, scores technical indicators, calculates statistical measures, and produces numerical outputs (expected return, volatility, conviction scores). Every computation is tagged with Source UUIDs and Computation Traces. If the code output contradicts the thesis (for example, the LLM argued undervalued but the DCF confirms fair pricing), the contradiction is flagged and the thesis is revised.

**Phase 3: Structured Output**

The agent produces a Research Report containing:
- The Investment Thesis (qualitative narrative with citations)
- The Quantitative Validation (numerical expected return, volatility, conviction score with computation traces)
- The Contributing Alpha Factors (factor identifiers with their current information coefficient and decay status)
- The Key Risks (dissenting signals and downside scenarios)

### Multi-Horizon Analysis

Each analyst produces analysis at three frequencies, following institutional practice where portfolio managers maintain views across multiple time horizons:

**Long-Term (Quarterly, approximately 252 trading days):** Deep fundamental research, updated when new 10-K/10-Q filings arrive or material corporate events occur. This includes full DCF revaluation, competitive positioning assessment, management quality evaluation, and secular trend alignment. Results are cached and reused until the next material event trigger.

**Medium-Term (Weekly, approximately 63 trading days):** Earnings estimate revisions, sector rotation signals, credit spread changes, and institutional flow analysis. Updated weekly or when triggered by material news.

**Short-Term (Daily, approximately 10 trading days):** Technical momentum, intraday sentiment from news flow, options market signals, and short-interest changes. Computed fresh daily.

The final output for each stock composites all three horizons into a unified signal, with the Synthesis Agent determining the appropriate weighting based on the stock's characteristics. A utility stock may weight long-term 70%, while a momentum-driven tech stock may weight short-term 40%.

### Synthesis Agent (Dynamic Signal Aggregation)

Rather than using static weights to combine analyst outputs, a dedicated LLM Synthesis Agent dynamically determines the optimal weighting for each stock based on context. This agent reads all Research Reports for a given stock and produces:

**Dynamic Weight Assignment:** Explicit weights for fundamental, technical, and sentiment analysts per stock, per horizon, with a written justification. For a biotech stock awaiting FDA approval, sentiment and fundamental catalysts dominate. For a utility during a rate-hiking cycle, macro signals dominate.

**Cross-Signal Consistency Check:** The agent identifies and flags contradictions across analyst reports. If the Fundamental Analyst is bullish but the Market Analyst shows a breakdown, the Synthesis Agent documents the conflict and adjusts conviction accordingly rather than mechanically averaging.

**Composite Signal Generation:** The final composite signal for each horizon is a weighted combination of the three analyst expected returns. These per-horizon composites are blended into a unified final signal by the Synthesis Agent, which determines horizon weights based on the stock's characteristics. Code validates that the final signal is mathematically consistent. The full covariance matrix across the analyzed universe is computed deterministically from the individual analyst outputs.

The Synthesis Agent's reasoning becomes part of the provenance chain visible to the user.

### Code Validation Agent

The Code Validation Agent (CVA) is a deterministic wrapper around the LLM code-generation loop. Its purpose is to ensure that every quantitative claim in an analyst report is produced by executable, validated Python code rather than LLM hallucination.

**Workflow:**
1. The analyst's Phase 2 prompt is injected with the raw data files (CSV snapshots) and a system instruction requiring all numbers to derive from actual data.
2. The LLM generates a Python script (typically 100-400 lines) implementing the requested valuation or technical analysis.
3. The script is run in a subprocess with a 60-second timeout. Allowed libraries: numpy, pandas, math, statistics, talib.
4. The output is parsed as JSON and checked against a Pydantic schema (ResearchReport). Missing fields, type mismatches, or out-of-range values trigger a retry.
5. Up to 5 iterations are permitted. If all fail, the analyst returns an error record with partial outputs.

**Common failure modes and handling:**
- SyntaxError or ImportError: Captured from stderr, fed back to the LLM with the traceback.
- Timeout: If execution exceeds 60 seconds (typically due to infinite loops in LLM-generated code), the process is killed and a simplified prompt is issued.
- Schema Validation Failure: Pydantic reports exactly which fields are missing or invalid. The LLM receives this structured feedback.
- Contradiction Detection: If the code output contradicts the qualitative thesis, the contradiction is logged and the thesis is revised.

### Limitations of Layer 2

It is important to state clearly what Layer 2 does not do, because these omissions are precisely what the empirical results diagnose:

**No continuous alpha discovery.** The analysts reason over raw data---prices, filings, news---but they do not operate over a library of statistically validated predictive factors. The LLM must rediscover patterns from scratch on every run, which is both computationally expensive and statistically unreliable.

**No persistent memory.** Each trading session starts from a blank slate. The system does not remember why it bought a stock last month, whether that thesis has been validated or invalidated, or how its past predictions have performed. This makes it a trader, not an investor.

**No portfolio-level optimization.** The synthesis agent produces a signal per stock, but there is no cross-asset risk management, no turnover constraint, and no tax-aware rebalancing. The system treats each stock as an independent decision.

These three limitations motivate Layer 1 (Global R&D Factory for validated alpha factors) and Layer 3 (LLM Chief Investment Officer for persistent investment memory and thesis-driven deliberation).

---

## Tech Stack

- **Language:** Python 3.12+
- **Agent Orchestration:** LangGraph 0.3+, LangChain Core
- **LLM Providers:** OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama (local)
- **Local Model (default):** `minimax-m2.7:cloud` via Ollama v0.6.4
- **Data Sources:**
  - yFinance (daily OHLCV prices)
  - Alpha Vantage (fundamentals, earnings, technical indicators)
  - SEC EDGAR (10-K, 10-Q, 8-K filings with noise reduction and deduplication)
  - Alpaca News (real-time and historical news)
  - yFinance News (supplementary coverage)
- **Key Libraries:** pandas, numpy, pydantic, talib, sec-edgar-downloader, rich, typer, langgraph
- **Package:** `tradingagents` v0.2.0

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ishmamkabir/Multi-Agent-AI-Portfolio-Manager.git
cd Multi-Agent-AI-Portfolio-Manager
```

Create a virtual environment:

```bash
conda create -n ai-portfolio-manager python=3.12
conda activate ai-portfolio-manager
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Required API Keys

Set keys for your chosen providers:

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

---

## CLI Usage

Run the interactive CLI directly:

```bash
python -m cli.main
```

A screen will appear where you can select tickers, date, LLMs, research depth, and active analysts. The interface shows results as they load, letting you track each analyst's progress.

---

## Package Usage

### Quick Start

Import the `TradingGraph` and initialize with default config. The `.propagate()` function returns the final state and a decision:

```python
from src.graph.trading_graph import TradingGraph
from src.default_config import DEFAULT_CONFIG

ta = TradingGraph(debug=True, config=DEFAULT_CONFIG.copy())

final_state, decision = ta.propagate("AAPL", "2025-03-14")
print(decision)
```

### Custom Configuration

```python
from src.graph.trading_graph import TradingGraph
from src.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # openai, google, anthropic, xai, openrouter, ollama
config["deep_think_llm"] = "gpt-5.2"     # Model for complex reasoning
config["quick_think_llm"] = "gpt-5-mini" # Model for quick tasks
config["research_depth"] = "deep"        # shallow, medium, deep
config["horizons_enabled"] = ["long_term", "medium_term", "short_term"]

ta = TradingGraph(
    selected_analysts=["market", "fundamentals", "news"],
    debug=True,
    config=config
)
final_state, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `src/default_config.py` for all configuration options.

---

## Backtesting and Evaluation

The framework includes a comprehensive backtest suite for measuring determinism, directional accuracy, and cross-horizon coherence:

### Deviation Tests

Cross-run determinism tests (8 identical runs, temperature=0, fixed seed) measuring conviction consistency and expected return variance across runs. These quantify how reproducible the neuro-symbolic pipeline is when all randomness is controlled.

### Regime-Aware Backtests

Individual analyst backtests and full system backtests across seven historical market regimes (2010-2024):
- Post-GFC recovery (2010-2013)
- Mid bull (2014-2017)
- Volatile / trade war (2018-2019)
- COVID crash (Feb-Mar 2020)
- Recovery bull (2020-2021)
- 2022 bear market
- Post-bear bull (2022-2024)

### Per-Analyst Ablation Study

Decomposes the composite signal to identify which analyst drives accuracy and whether the synthesis layer helps or hurts. Each analyst's individual long-horizon prediction is evaluated against realized forward returns without synthesis.

### Evaluation Metrics

- **Directional Accuracy:** Fraction of predictions where the sign of the predicted return matches the sign of the actual forward return.
- **MAE and RMSE:** Mean absolute error and root mean squared error between predicted and actual returns.
- **Conviction Consistency:** Standard deviation of conviction scores across deviation-test runs.
- **Cross-Horizon Coherence:** Pearson correlation between short-, medium-, and long-term predictions for the same stock and date.

### Running Experiments

Run all experiments at once:

```bash
python run_all_experiments.py
```

Run individual backtests:

```bash
# Market analyst backtest
python src/backtest/market_analyst_backtest.py --ticker AAPL --workers 4

# Fundamental analyst backtest
python src/backtest/fundamentals_analyst_backtest.py --ticker AAPL --workers 4

# News analyst backtest
python src/backtest/news_analyst_backtest.py --ticker AAPL --workers 4

# Full system backtest
python src/backtest/system_backtest.py --ticker AAPL --workers 2 --research-depth shallow
```

### Key Empirical Findings

The evaluation reveals sobering results that motivate the three-layer architecture:

- **Directional accuracy near random-walk levels:** Analyst directional accuracy typically ranges between 45% and 55%, statistically indistinguishable from a coin flip.
- **High variance in deterministic code outputs:** Even with temperature 0 and fixed seeds, the Code Validation Agent produces non-negligible variation in computed metrics across runs.
- **Cross-horizon incoherence:** Short-term and long-term predictions for the same stock and date frequently disagree in sign.
- **Synthesis amplifies noise:** The composite signal consistently underperforms the best individual analyst in the ablation study. When the Synthesis Agent combines a strong signal with a weak or contradictory signal, the result is a muddled near-zero composite that loses directional information.

These failures are not engineering bugs; they are structural. They motivate the need for Layer 1 (validated predictive signals) and Layer 3 (investment discipline) to build a viable intelligent portfolio management system.

---

## Data Sources and Dataflows

The system integrates five primary data sources, each exposed through a unified dataflow interface:

**yFinance:** Daily OHLCV prices, adjusted for splits and dividends. Lookback windows scale with horizon: 252 days (long), 63 days (medium), 20 days (short).

**Alpha Vantage:** Fundamental data (income statement, balance sheet, cash flow), earnings calendar, analyst estimates, and technical indicators. API rate limits enforce 5 calls per minute on the free tier.

**SEC EDGAR:** 10-K, 10-Q, 8-K, and proxy statement filings retrieved via the official SEC API. Filings are parsed with noise reduction (deduplication of boilerplate, header and footer stripping) to minimize context-window consumption.

**Alpaca News:** Real-time and historical news articles with sentiment metadata. Articles are filtered by relevance score and deduplicated by URL hash.

**yFinance News:** Supplementary news feed for broader coverage, particularly for international tickers.

All dataflows implement a common interface with three methods: `fetch(ticker, start, end)`, `cache_key()`, and `validate()`. Caching is file-based (JSON/CSV in `dataflows/data_cache/`) with TTL rules: price data expires after 1 trading day, fundamental data after 1 calendar day, and news after 4 hours.

---

## Project Structure

```
.
├── cli/                          # Interactive CLI
│   ├── main.py                   # Entry point
│   ├── config.py                 # CLI-specific configuration
│   ├── models.py                 # CLI data models
│   ├── stats_handler.py          # LLM call tracking
│   ├── announcements.py          # Announcement fetching
│   ├── saved_config.py           # Config persistence
│   └── utils.py                  # CLI utilities
├── src/
│   ├── agents/
│   │   ├── analysts/
│   │   │   ├── base_analyst.py          # Shared analyst logic
│   │   │   ├── fundamentals_analyst.py  # Fundamental analyst agent
│   │   │   ├── market_analyst.py        # Market (technical) analyst agent
│   │   │   └── news_analyst.py          # News sentiment analyst agent
│   │   ├── code_agent/
│   │   │   ├── code_agent.py            # Code Validation Agent (Phase 2)
│   │   │   └── code_agent_old.py        # Previous iteration
│   │   ├── prompts/
│   │   │   ├── code_agent.md            # CVA system prompt
│   │   │   ├── fundamentals_analyst.md  # Fundamental analyst prompts
│   │   │   ├── market_analyst.md        # Market analyst prompts
│   │   │   └── news_analyst.md          # News analyst prompts
│   │   ├── trader/
│   │   │   └── trader.py                # Synthesis Agent (dynamic aggregation)
│   │   └── utils/
│   │       ├── agent_states.py          # Agent state definitions
│   │       ├── agent_utils.py           # Shared utilities
│   │       ├── core_stock_tools.py      # Core stock data tools
│   │       ├── fundamental_data_tools.py  # SEC/financial data tools
│   │       ├── memory.py                # Financial situation memory
│   │       ├── news_data_tools.py       # News retrieval tools
│   │       ├── schemas.py               # Pydantic schemas (ResearchReport, CompositeSignal)
│   │       └── technical_indicators_tools.py  # Technical indicator computation
│   ├── backtest/
│   │   ├── deviation_test.py            # Cross-run determinism tests
│   │   ├── fundamentals_analyst_backtest.py  # Fundamental analyst evaluation
│   │   ├── market_analyst_backtest.py   # Market analyst evaluation
│   │   ├── news_analyst_backtest.py     # News analyst evaluation
│   │   ├── system_backtest.py           # Full multi-agent system evaluation
│   │   ├── trader_deviation_test.py     # Synthesis Agent determinism tests
│   │   └── _proc_utils.py               # Parallel processing utilities
│   ├── dataflows/
│   │   ├── alpaca.py                    # Alpaca news and positions
│   │   ├── alpha_vantage*.py            # Alpha Vantage data sources
│   │   ├── config.py                    # Dataflow configuration
│   │   ├── interface.py                 # Unified dataflow interface
│   │   ├── sec_edgar.py                 # SEC EDGAR filing retrieval
│   │   ├── utils.py                     # Dataflow utilities
│   │   ├── y_finance.py                 # yFinance price and fundamental data
│   │   └── yfinance_news.py             # yFinance news feed
│   ├── graph/
│   │   ├── trading_graph.py             # Main orchestration (TradingGraph)
│   │   ├── setup.py                     # Graph construction
│   │   ├── propagation.py               # State propagation
│   │   ├── reflection.py                # Reflection and debate logic
│   │   ├── signal_processing.py         # Deterministic signal processing
│   │   ├── conditional_logic.py         # Conditional graph edges
│   │   ├── stock_screener.py            # Stock universe screening
│   │   └── russell_3000_cache.txt       # Cached Russell 3000 constituents
│   ├── llm_clients/
│   │   ├── anthropic_client.py          # Anthropic Claude client
│   │   ├── google_client.py             # Google Gemini client
│   │   ├── openai_client.py             # OpenAI GPT client
│   │   ├── base_client.py               # Abstract base client
│   │   ├── factory.py                   # Client factory
│   │   ├── rate_limiter.py              # Request rate limiting
│   │   └── validators.py                # Output validators
│   └── default_config.py              # Default configuration
├── tests/                        # Unit and integration tests
├── experiment_results/           # Final paper (LaTeX, PDF, tables)
├── assets/                       # Diagrams and screenshots
└── run_all_experiments.py        # Orchestrates all backtests
```

---

## Research Context: Three-Layer Architecture

This framework implements **Layer 2** of a proposed three-layer architecture for LLM-driven portfolio management.

### Layer 1: Global R&D Factory (Offline)

Layer 1 is entirely user-agnostic and operates offline. It produces a continuously growing library of validated alpha factors and a screened stock universe. All four analysis domains (Fundamental, Technical, Macro, and News/Sentiment) have offline discovery components: each discovers predictive patterns in its respective domain, validates them through the Falsification Critic, and registers them in the Alpha Factor Library.

**Track A: Continuous Alpha Discovery via MCTS**

All agent types mine alpha signals using Monte Carlo Tree Search combined with LLM-guided hypothesis generation. The LLM generates investment hypotheses in natural language; the MCTS engine explores the combinatorial space of formalizations; deterministic backtesting provides the reward signal. Track A runs continuously and the library only grows.

**The Falsification Critic**

Every factor must pass four statistical tests before entering the Alpha Factor Library:
1. Deflated Sharpe Ratio significance, adjusting for non-normality and multiple testing.
2. Bagged Combinatorially Purged Cross-Validation, ensuring out-of-sample robustness by embargoing data around test folds.
3. Regime Robustness: positive mean information coefficient across bull, bear, sideways, and high-volatility regimes.
4. Complexity-Adjusted Significance: the required significance threshold scales inversely with AST depth.

These standards separate genuine signals from backtest overfitting. The Falsification Critic is fully deterministic with no LLM involvement in the acceptance decision.

**Why Layer 1 Fixes Layer 2**

The empirical results show that the Fundamental analyst achieves only 32-54% directional accuracy. This is not because the LLM is bad at reading SEC filings; it is because the LLM is bad at knowing which patterns in those filings are predictive. A human fundamental analyst does not read a 10-K cold every quarter; she operates with a mental library of validated heuristics. Layer 1 provides the LLM with exactly this library: a collection of statistically validated factors, each with a known information coefficient, decay trajectory, and regime-conditional performance profile.

With Layer 1, the Layer 2 analyst is no longer pattern-matching in noise. It is reasoning over validated signals. The analyst's job shifts from discovery to interpretation---a task at which LLMs excel.

### Layer 2: Multi-Agent Analyst System (This Repository)

The system implemented here. Three specialized analysts with three-phase neuro-symbolic workflows, producing multi-horizon conviction signals aggregated by a dynamic Synthesis Agent.

### Layer 3: LLM Chief Investment Officer

Layer 3 is the only user-facing layer and the only layer receiving user-specific data. It selects optimal positions per user, where the number of positions is determined by the user's risk profile, portfolio size, and diversification requirements. The CIO is modeled after a Chief Investment Officer at a systematic firm, combining quantitative optimization with principled qualitative judgment and a long-term investment philosophy.

Critically, the CIO is an investor, not a trader. It thinks in terms of investment theses, position sizing within a macro framework, and multi-year compounding, not daily price action. Every decision is made with the question: "Does the underlying thesis still hold?"

**Investment Memory and Journal**

The LLM CIO maintains a persistent per-user Investment Journal stored in the database (separate from the ephemeral execution context). Before each session, this journal is loaded into the CIO's context. It contains:
- Active Investment Theses: For each current position, the original thesis articulated when the position was initiated.
- Decision History: A log of every past recommendation with the reasoning given at the time.
- Thesis Evolution: How the CIO's view has evolved over time, including thesis risk events and responses.
- Portfolio Strategy: The CIO's articulated strategy for this user, ensuring coherent investment philosophy across rebalancing events.

**Thesis Review and CIO Deliberation**

The core innovation is the deliberation process. The LLM CIO reads the quantitative risk assessment, the multi-horizon analyst reports, the Synthesis Agent's signal, and the Investment Journal, then deliberates as a Chief Investment Officer would:
- **Thesis Review:** For each existing position, the CIO asks whether the original investment thesis still holds. If the thesis is intact and the position is performing, the default is to hold.
- **Opportunity Assessment:** For stocks not currently held, the CIO reviews composite signals and identifies the most compelling new opportunities.
- **Portfolio Construction:** The CIO determines the target number of positions based on the user's profile.
- **Proposed Changes:** The CIO produces a proposed rebalancing plan with written reasoning for each action, cross-referenced against the Investment Journal for consistency.

**Constrained Optimization**

The CIO's proposed changes are validated through a deterministic constrained optimizer with:
- Maximum turnover cap: at most 20% of portfolio value per rebalancing event.
- Minimum holding period: positions held fewer than 30 trading days are locked unless a thesis-invalidating event has occurred.
- Transaction cost penalty: trades must exceed 3 times round-trip cost in expected improvement.
- Tax awareness: short-term gains carry higher implicit cost than long-term.
- Existing position bias: optimizer initialized from current weights with deviation penalty.
- Rebalancing threshold: if expected Sharpe improvement is below 0.1, the CIO recommends no action.

**Why Layer 3 Fixes Layer 2**

The empirical results show that the Layer 2 composite signal achieves only 54.8% directional accuracy. If a human portfolio manager made a new trading decision every time a noisy signal flipped from HOLD to BUY, they would churn their portfolio to death. Layer 3 prevents this by:
1. Enforcing thesis discipline: a position is not exited because the short-term signal weakened; it is exited only when the original thesis is invalidated.
2. Preventing over-trading: turnover caps, minimum holding periods, and rebalancing thresholds ensure the system ignores noise.
3. Maintaining consistency: the Investment Journal forces the LLM to confront its own past reasoning.

In short, Layer 3 does not make the analysts more accurate. It makes the portfolio more robust to analyst inaccuracy. Given that the empirical results show analysts are inaccurate, this is not a luxury---it is a necessity.

### Cross-Layer Provenance Chain

The provenance chain unifies all three layers, capturing both the computational trace (code) and the LLM reasoning trace (qualitative judgment) at every stage. When a user inspects any position, the system surfaces five levels of detail:

- **Level 1 (Summary):** "Buy AAPL at 4.2% weight. CIO thesis: AI Services expansion underpriced."
- **Level 2 (Drivers):** Primary factors with citations; alpha factors with historical returns and decay status; Synthesis Agent's weighting rationale.
- **Level 3 (Analysis):** Full Research Reports from each analyst, including both the LLM's qualitative thesis and the code's quantitative validation; sentiment citations with article URLs.
- **Level 4 (Computation):** Executable Python code for every numerical claim, with Source UUIDs linking to raw data.
- **Level 5 (Consistency):** Investment Journal entries showing how this recommendation is consistent with (or a departure from) the CIO's historical reasoning for this user.

This chain ensures that no recommendation is ever a black-box output. Every signal can be traced backward through the Synthesis Agent to the individual analyst reports, through the analyst's Phase 2 code to the underlying alpha factors, and through the alpha factors to their Layer 1 discovery history, validation statistics, and decay status.

---

## Schema Definitions

### Research Report

Each analyst produces a structured ResearchReport JSON object:

```
ResearchReport:
  company_of_interest: str
  trade_date: str
  report_type: str          # "fundamental", "market", "news"
  investment_thesis: str    # qualitative narrative
  mu: MuScores               # per-horizon expected returns
  sigma_contribution: MuScores
  conviction: MuScores       # per-horizon confidence [0, 1]
  key_risks: List[str]
  catalysts: List[str]
  metadata: Dict              # model, date, execution trace

MuScores:
  long_term: float
  medium_term: float
  short_term: float
```

### Composite Signal

The Synthesis Agent aggregates multiple ResearchReport objects into a CompositeSignal:

```
CompositeSignal:
  ticker: str
  date: str
  weights: Dict[str, float]       # per-analyst weights
  mu_composite: MuScores
  sigma_composite: MuScores
  conviction_composite: MuScores
  consistency_check: str          # cross-signal contradiction notes
  final_decision: str             # BUY / SELL / HOLD
```

---

## Contributing

Contributions are welcome. Whether fixing a bug, improving documentation, or suggesting a new feature, your input helps advance this research.
