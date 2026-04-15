"""
Market (Technical) Analyst — Phase 1/3 prompts, data gathering, and node factory.

Uses the base_analyst 3-phase subgraph to:
  1. Deterministically gather OHLCV data and technical indicators
  2. LLM plans which statistical/technical metrics to compute
  3. Code agent computes them
  4. LLM interprets results and forms a thesis on price direction
"""

from datetime import datetime

from src.agents.code_agent.code_agent import CodeValidationAgent
from src.agents.utils.schemas import AgentType, ResearchReport
from src.agents.analysts.base_analyst import (
    create_analyst_node,
    compute_date_range,
)
from src.llm_clients import create_llm_client, create_rate_limiter

# ── Horizon-specific focus instructions ─────────────────────────────────────

HORIZON_FOCUS = {
    "long_term": (
        "HORIZON: Long-term (1+ years).  Lookback = 365 days.\n"
        "\n"
        "PRIORITY INDICATORS (base set):\n"
        "  TREND   — 200-EMA slope & price position, linear-regression slope of\n"
        "            log-price (200d), ADX(14) for trend strength.\n"
        "  VOL     — 252-day realised vol (close-to-close), Yang-Zhang 252d,\n"
        "            vol-of-vol (rolling std of 21d vol).\n"
        "  REGIME  — HMM 2-state (bull/bear) posterior probability,\n"
        "            volatility regime (percentile rank of 60d vol vs 1yr).\n"
        "  MEAN-REV — Hurst exponent (R/S, 200d), z-score of price vs 200-SMA.\n"
        "  VOLUME  — OBV 200d slope, cumulative A/D line trend.\n"
        "  SUP/RES — Key support/resistance levels from pivot points (traditional\n"
        "            & Fibonacci), prior 252d swing highs/lows, volume-profile\n"
        "            price clusters.  Compute: nearest support S and resistance R,\n"
        "            dist_to_support = (price - S) / price, dist_to_resistance =\n"
        "            (R - price) / price, and S/R ratio = dist_to_resistance /\n"
        "            (dist_to_support + dist_to_resistance).  Ratio > 0.5 means\n"
        "            price is closer to support (bullish); < 0.5 closer to\n"
        "            resistance (bearish).  Also compute zone_width =\n"
        "            (R - S) / price to gauge range compression/expansion.\n"
        "\n"
        "FUNDAMENTAL CALIBRATION RULES (apply ALL that match the fundamentals snapshot):\n"
        "  BETA:\n"
        "    - Beta < 0.8 (low-vol): Multiply all lookback windows by 1.5;\n"
        "      set BB multiplier to 2.5; set base mu weights to\n"
        "      trend=0.65, momentum=0.05, mean_rev=0.05, volume=0.10,\n"
        "      regime=0.05, sup_res=0.10.\n"
        "    - 0.8 <= Beta <= 1.5 (normal): Keep base mu weights below.\n"
        "    - Beta > 1.5 (high-vol): Multiply all lookback windows by 0.75;\n"
        "      set base mu weights to\n"
        "      trend=0.45, momentum=0.20, mean_rev=0.10, volume=0.10,\n"
        "      regime=0.05, sup_res=0.10.\n"
        "  MARKET CAP (controls sigma blend ONLY):\n"
        "    - Market Cap > 200B (mega-cap): sigma_blend_r=0.60, sigma_blend_yz=0.40.\n"
        "    - 10B <= Market Cap <= 200B (mid-cap): sigma_blend_r=0.50, sigma_blend_yz=0.50.\n"
        "    - Market Cap < 10B (small-cap): sigma_blend_r=0.40, sigma_blend_yz=0.60;\n"
        "      add volume-weighted RSI.\n"
        "  DIVIDEND YIELD:\n"
        "    - Dividend Yield > 0.02 (2%): Use dividend-adjusted close for ALL\n"
        "      return-based metrics; set dividend_carry = dividend_yield.\n"
        "    - Dividend Yield <= 0.02: Use raw Close; set dividend_carry = 0.0.\n"
        "  PROFITABILITY (ROE, Operating Margin):\n"
        "    - ROE > 0.20 AND Operating Margin > 0.25: Transfer 0.10 from\n"
        "      mean_rev to trend in mu weights.\n"
        "    - ROE < 0.10 OR Operating Margin < 0.10: Transfer 0.10 from\n"
        "      trend to momentum in mu weights; add vol-of-vol if not present.\n"
        "  DEBT / FINANCIAL HEALTH:\n"
        "    - D/E > 2.0 OR Current Ratio < 1.0: Set credit_mult = 1.15;\n"
        "      add 5-day realised vol; transfer 0.05 from trend to vol in mu.\n"
        "    - D/E < 0.5 AND Current Ratio > 2.0: Set credit_mult = 0.90.\n"
        "    - Otherwise: Set credit_mult = 1.0.\n"
        "  SECTOR / INDUSTRY:\n"
        "    - Sector = 'Utilities' or 'Consumer Defensive': Transfer 0.10 from\n"
        "      momentum to mean_rev; multiply EMA windows by 1.5.\n"
        "    - Sector = 'Healthcare': Add 50d EMA; transfer 0.05 from momentum\n"
        "      to trend.\n"
        "\n"
        "BASE MU WEIGHTS (sum = 1.0, apply calibration transfers then renormalise):\n"
        "  trend = 0.55, momentum = 0.10, mean_rev = 0.10, volume = 0.10,\n"
        "  regime = 0.05, sup_res = 0.10\n"
        "\n"
        "SIGMA: sigma = max(credit_mult * regime_mult * zone_width_mult\n"
        "       * (sigma_blend_r * rv252 + sigma_blend_yz * yz252), 0.01)\n"
        "  regime_mult: if vol_pctile > 80 → 1.25; if vol_pctile < 20 → 0.90; else 1.0\n"
        "  zone_width_mult: if zone_width < 0.05 → 1.20 (range compression →\n"
        "    breakout risk widens sigma); if zone_width > 0.25 → 0.95 (wide\n"
        "    range reduces near-term vol); else 1.0"
    ),
    "medium_term": (
        "HORIZON: Medium-term (3-12 months).  Lookback = 90 days.\n"
        "\n"
        "PRIORITY INDICATORS (base set):\n"
        "  TREND   — 50/200-EMA crossover state, ADX(14) + DI spread,\n"
        "            Ichimoku cloud (tenkan/kijun cross, price vs cloud).\n"
        "  MOMENTUM — RSI(14), MACD(12,26,9) histogram, Stochastic(14,3,3).\n"
        "  VOL     — 63-day realised vol, ATR(14), Bollinger Band %B and width.\n"
        "  MEAN-REV — z-score of price vs 50-SMA, half-life of mean reversion\n"
        "             (OLS on lagged spread).\n"
        "  VOLUME  — CMF(21), OBV 50d slope, volume-weighted RSI.\n"
        "  SUP/RES — Key support/resistance from 90d pivot points (traditional\n"
        "            & Fibonacci), prior 90d swing highs/lows, Ichimoku\n"
        "            cloud boundaries as S/R zones.  Compute: nearest support S\n"
        "            and resistance R, dist_to_support = (price - S) / price,\n"
        "            dist_to_resistance = (R - price) / price, S/R ratio =\n"
        "            dist_to_resistance / (dist_to_support + dist_to_resistance).\n"
        "            Ratio > 0.5 = closer to support (bullish); < 0.5 =\n"
        "            closer to resistance (bearish).  Also compute zone_width =\n"
        "            (R - S) / price for range compression/expansion.\n"
        "\n"
        "FUNDAMENTAL CALIBRATION RULES (apply ALL that match):\n"
        "  BETA:\n"
        "    - Beta < 0.8: Set BB multiplier to 2.5; transfer 0.10 from momentum\n"
        "      to trend in mu weights.\n"
        "    - 0.8 <= Beta <= 1.5: Keep base weights.\n"
        "    - Beta > 1.5: Transfer 0.10 from trend to momentum.\n"
        "  MARKET CAP (controls sigma blend ONLY):\n"
        "    - Market Cap > 200B: sigma_blend_r=0.60, sigma_blend_yz=0.40.\n"
        "    - 10B <= Market Cap <= 200B: sigma_blend_r=0.50, sigma_blend_yz=0.50.\n"
        "    - Market Cap < 10B: sigma_blend_r=0.40, sigma_blend_yz=0.60;\n"
        "      add VWAP deviation and volume ratio (5d/20d).\n"
        "  DIVIDEND YIELD:\n"
        "    - Dividend Yield > 0.02: Use dividend-adjusted close;\n"
        "      set dividend_carry = dividend_yield.\n"
        "    - Dividend Yield <= 0.02: Use raw Close; set dividend_carry = 0.0.\n"
        "  PROFITABILITY:\n"
        "    - ROE > 0.20 AND Operating Margin > 0.25: Transfer 0.05 from\n"
        "      mean_rev to trend.\n"
        "    - ROE < 0.10 OR Operating Margin < 0.10: Transfer 0.05 from\n"
        "      mean_rev to momentum.\n"
        "  DEBT / FINANCIAL HEALTH:\n"
        "    - D/E > 2.0 OR Current Ratio < 1.0: Set credit_mult = 1.15;\n"
        "      add 10d realised vol.\n"
        "    - D/E < 0.5 AND Current Ratio > 2.0: Set credit_mult = 0.90.\n"
        "    - Otherwise: Set credit_mult = 1.0.\n"
        "  PE RATIO:\n"
        "    - PE Ratio (TTM) > 35 OR Forward PE > 30: Transfer 0.05 from trend\n"
        "      to mean_rev.\n"
        "    - PE Ratio (TTM) < 12: Transfer 0.05 from mean_rev to trend.\n"
        "\n"
        "BASE MU WEIGHTS (sum = 1.0, apply transfers then renormalise):\n"
        "  trend = 0.35, momentum = 0.35, mean_rev = 0.15, sup_res = 0.15\n"
        "\n"
        "SIGMA: sigma = max(credit_mult * regime_mult * zone_width_mult\n"
        "       * (sigma_blend_r * rv63 + sigma_blend_yz * yz63), 0.01)\n"
        "  yz63 = Yang-Zhang vol annualised over 63d\n"
        "  regime_mult: if vol_pctile > 80 → 1.25; if vol_pctile < 20 → 0.90; else 1.0\n"
        "  zone_width_mult: if zone_width < 0.05 → 1.20 (range compression →\n"
        "    breakout risk widens sigma); if zone_width > 0.25 → 0.95 (wide\n"
        "    range reduces near-term vol); else 1.0"
    ),
    "short_term": (
        "HORIZON: Short-term (next 12 trading days ≈ 2 weeks).  Lookback = 30 days.\n"
        "\n"
        "SHORT-TERM IS PRIMARILY MEAN-REVERSION.  Academic and practitioner\n"
        "research shows 1-2 week holding periods exhibit negative autocorrelation\n"
        "(reversal), NOT momentum.  Momentum only activates conditionally when\n"
        "the market is trending strongly (see REGIME GATE below).\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "MANDATORY INDICATORS (compute ALL every run with EXACT parameters)\n"
        "════════════════════════════════════════════════════════════════════\n"
        "These indicators MUST appear in every computation plan for short-term.\n"
        "Do NOT substitute, omit, or change parameters.  Your only freedom is\n"
        "the fundamental profile classification in Step 1.\n"
        "\n"
        "  MEAN-REV (primary in RANGING, secondary in TRENDING):\n"
        "    rsi_2:        RSI with period=2.\n"
        "                   RANGING mode normalisation: clip((50 - rsi) / 40, -1, +1)\n"
        "                     [RSI < 10 = oversold = +1 bullish]\n"
        "                   TRENDING mode normalisation: clip((rsi - 50) / 50, -1, +1)\n"
        "                     [RSI > 90 = overbought = +1 bullish]\n"
        "    zscore_10sma: z-score of price vs 10-SMA.\n"
        "                   Normalisation: clip(-z / 2, -1, +1)\n"
        "                     [negative z = price below mean = +1 bullish]\n"
        "    bb_pctb_10:   Bollinger %B with period=10, width=2.\n"
        "                   RANGING mode normalisation: clip(1 - 2*%B, -1, +1)\n"
        "                     [%B < 0 = below lower band = +1 bullish]\n"
        "                   TRENDING mode normalisation: clip(2*(%B - 0.5), -1, +1)\n"
        "                     [%B > 1 = above upper band = +1 bullish]\n"
        "\n"
        "  MOMENTUM (secondary in RANGING, primary in TRENDING):\n"
        "    stoch_kd:     Stochastic(5,3,3) %K.\n"
        "                   RANGING mode normalisation: clip((50 - %K) / 40, -1, +1)\n"
        "                     [%K < 10 = oversold = +1 bullish]\n"
        "                   TRENDING mode normalisation: clip((%K - 50) / 50, -1, +1)\n"
        "                     [%K > 90 = overbought = +1 bullish]\n"
        "    roc_5:        ROC with period=5.\n"
        "                   Normalisation: clip(roc / (3 * ATR_5 / close), -1, +1)\n"
        "    macd_hist:    MACD(5,13,4) histogram.\n"
        "                   Normalisation: clip(hist / (2 * ATR_5 / close), -1, +1)\n"
        "\n"
        "  REGIME (deterministic gate — NOT directional for mu):\n"
        "    kaufman_er:   ER(10) = abs(direction) / abs(volatility) over 10d.\n"
        "                   ER > 0.40 = trending; ER < 0.20 = ranging.\n"
        "    price_to_high: PTH = close / highest(high, 252).\n"
        "                   PTH >= 0.95 with volume_ratio > 1.2 = momentum mode.\n"
        "\n"
        "  TREND:\n"
        "    ema_cross:    EMA(5) vs EMA(10) crossover.\n"
        "                   Normalisation: +1 if fast > slow, -1 if fast < slow, 0 otherwise\n"
        "    price_ema10:  (close - EMA_10) / close.\n"
        "                   Normalisation: clip(value * 10, -1, +1)\n"
        "\n"
        "  VOLATILITY (sigma computation ONLY — NOT directional for mu):\n"
        "    ewma_vol:     EWMA vol with lambda=0.92 (seeded with 10d variance).\n"
        "                   ewma_vol_t = 0.92 * ewma_vol_{t-1} + 0.08 * ret^2\n"
        "                   ewma_vol_ann = sqrt(ewma_vol_t * 252)\n"
        "    yz_vol:       Yang-Zhang vol over 10d, annualised × sqrt(252).\n"
        "    atr_5:        ATR with period=5.\n"
        "\n"
        "  VOLUME:\n"
        "    obv_slope:    OBV 10d slope.\n"
        "                   Normalisation: clip(slope / std(OBV, 10), -1, +1)\n"
        "    cmf_10:       CMF with period=10.\n"
        "                   Normalisation: clip(CMF * 2, -1, +1)\n"
        "    volume_ratio: 5d avg volume / 20d avg volume.\n"
        "                   Normalisation: clip((vr - 1.0) / 0.5, -1, +1)\n"
        "\n"
        "  SUPPORT/RESISTANCE:\n"
        "    sr_ratio:     S/R ratio = (resistance - close) / (resistance - support).\n"
        "                   Normalisation: clip(2 * (sr_ratio - 0.5), -1, +1)\n"
        "                     [>0.5 = closer to support = bullish]\n"
        "    zone_width:   (resistance - support) / close.\n"
        "                   NOT directional — sigma input only.\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "REGIME GATE (mandatory — apply AFTER computing ER and PTH)\n"
        "════════════════════════════════════════════════════════════════════\n"
        "The regime gate switches BOTH normalisation mode AND mu weight allocation.\n"
        "\n"
        "  RANGING mode (DEFAULT): ER < 0.20, OR (ER 0.20-0.40 AND PTH < 0.95)\n"
        "    → Use RANGING normalisation for mean-reversion indicators (inverted)\n"
        "    → Use RANGING MODE weights (mean_rev dominant)\n"
        "  TRENDING mode: ER >= 0.40, OR (ER 0.20-0.40 AND PTH >= 0.95\n"
        "    AND volume_ratio > 1.2)\n"
        "    → Use TRENDING normalisation for mean-reversion indicators (standard)\n"
        "    → Use TRENDING MODE weights (momentum dominant)\n"
        "\n"
        "  REGIME CLARITY (for conviction computation):\n"
        "    ER < 0.20 OR ER > 0.40 → regime_clarity = 1.0 (clear regime)\n"
        "    ER 0.20-0.40 → regime_clarity = 0.5 (ambiguous regime)\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "FUNDAMENTAL CALIBRATION RULES (apply ALL that match, AFTER regime gate)\n"
        "════════════════════════════════════════════════════════════════════\n"
        "  BETA:\n"
        "    - Beta < 0.8: Set BB multiplier to 2.5; use ATR(10) instead of\n"
        "      ATR(5); transfer 0.05 from vol to trend in mu weights.\n"
        "    - 0.8 <= Beta <= 1.5: Keep base weights.\n"
        "    - Beta > 1.5: Set BB multiplier to 1.5; use ATR(3); transfer 0.05\n"
        "      from trend to momentum; set beta_sigma_mult = 1.10.\n"
        "    - Otherwise: Set beta_sigma_mult = 1.0.\n"
        "  MARKET CAP (controls sigma blend ONLY):\n"
        "    - Market Cap > 200B: sigma_blend_r=0.70, sigma_blend_yz=0.30.\n"
        "    - 10B <= Market Cap <= 200B: sigma_blend_r=0.50, sigma_blend_yz=0.50.\n"
        "    - Market Cap < 10B: sigma_blend_r=0.30, sigma_blend_yz=0.70;\n"
        "      add volume ratio as mandatory; transfer 0.05 from vol to volume\n"
        "      in mu weights.\n"
        "  DIVIDEND YIELD:\n"
        "    - Dividend Yield > 0.02: Use dividend-adjusted close;\n"
        "      set dividend_carry = dividend_yield * 12 / 252.\n"
        "    - Dividend Yield <= 0.02: Use raw Close; set dividend_carry = 0.0.\n"
        "  DEBT / FINANCIAL HEALTH:\n"
        "    - D/E > 2.0 OR Current Ratio < 1.0: Set credit_mult = 1.20;\n"
        "      add intraday range ratio; transfer 0.05 from trend to vol.\n"
        "    - D/E < 0.5 AND Current Ratio > 2.0: Set credit_mult = 0.90.\n"
        "    - Otherwise: Set credit_mult = 1.0.\n"
        "  SECTOR / INDUSTRY:\n"
        "    - Sector = 'Utilities' or 'Consumer Defensive': Transfer 0.10 from\n"
        "      momentum to mean_rev; add 5d SMA mean-reversion signal.\n"
        "  PROFITABILITY:\n"
        "    - ROE < 0.10 OR Operating Margin < 0.10: Transfer 0.05 from\n"
        "      trend to momentum in mu weights.\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "BASE MU WEIGHTS (sum = 1.0, apply regime gate then calibration\n"
        "  transfers then renormalise)\n"
        "════════════════════════════════════════════════════════════════════\n"
        "  RANGING MODE (mean-reversion dominant):\n"
        "    mean_rev = 0.35, momentum = 0.15, trend = 0.10, vol = 0.15,\n"
        "    volume = 0.10, sup_res = 0.15\n"
        "\n"
        "  TRENDING MODE (momentum dominant):\n"
        "    mean_rev = 0.15, momentum = 0.35, trend = 0.10, vol = 0.15,\n"
        "    volume = 0.10, sup_res = 0.15\n"
        "\n"
        "  NOTE: 'vol' weight is for the volatility CATEGORY in mu (not sigma).\n"
        "  Vol indicators that feed mu: ATR-normalised signals only.\n"
        "  Vol indicators that feed sigma: ewma_vol, yz_vol (NOT in mu weights).\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "MU FORMULA (DRIFT_SCALAR is horizon-specific)\n"
        "════════════════════════════════════════════════════════════════════\n"
        "  DRIFT_SCALAR for short-term = 0.05\n"
        "  mu = clip(composite_signal * 0.05 + dividend_carry, -0.08, +0.08)\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "SIGMA FORMULA\n"
        "════════════════════════════════════════════════════════════════════\n"
        "  sigma = max(adj_product * (sigma_blend_r * ewma_vol_ann\n"
        "           + sigma_blend_yz * yz_ann), 0.01)\n"
        "  adj_product = clip(credit_mult * beta_sigma_mult * regime_mult\n"
        "                     * zone_width_mult, 0.80, 1.50)\n"
        "    ↑ CAP prevents multiplicative adjustments from stacking beyond ±50%\n"
        "\n"
        "  ewma_vol_ann = EWMA-smoothed realised vol annualised.\n"
        "    lambda = 0.92 (optimised for 1-week horizon; half-life ≈ 8 days)\n"
        "    ewma_vol_t = 0.92 * ewma_vol_{t-1} + 0.08 * daily_return^2\n"
        "    ewma_vol_ann = sqrt(ewma_vol_t * 252)\n"
        "    Seed ewma_vol_0 with the 10d sample variance.\n"
        "\n"
        "  yz_ann = Yang-Zhang annualised volatility estimator.\n"
        "    yz = yang_zhang_vol * sqrt(252)\n"
        "    yang_zhang = sqrt(open_vol + close_vol - open_close_corr)\n"
        "    Uses N=10 for short-term.\n"
        "\n"
        "  regime_mult: if vol_pctile > 80 → 1.25; if vol_pctile < 20 → 0.90; else 1.0\n"
        "  zone_width_mult: if zone_width < 0.03 → 1.25; if zone_width > 0.15 → 0.90;\n"
        "    else 1.0\n"
        "\n"
        "════════════════════════════════════════════════════════════════════\n"
        "CONVICTION FORMULA (computed deterministically in phase4_output)\n"
        "════════════════════════════════════════════════════════════════════\n"
        "  conviction = clip(0.50 * signal_concordance\n"
        "                  + 0.20 * regime_clarity\n"
        "                  + 0.15 * signal_strength\n"
        "                  + 0.15 * data_completeness, 0, 1)\n"
        "\n"
        "  signal_concordance = |sum(w_i * norm_signal_i)| / sum(|w_i * norm_signal_i|)\n"
        "  regime_clarity     = 1.0 if ER < 0.20 or ER > 0.40; 0.5 otherwise\n"
        "  signal_strength    = min(|mu| / 0.04, 1.0)\n"
        "  data_completeness = n_computed / max(n_planned, 1)"
    ),
}

# ── Phase 1 system prompt (plan — decide WHAT to compute) ───────────────────

PHASE1_PROMPT = """\
You are a senior quantitative technical analyst executing a Fundamental \
Calibration workflow.  Your job is to plan the computation work — do NOT \
form a thesis or interpret anything yet.

The workflow has three mandatory steps.  You MUST complete all three and \
include the output of each in your plan.

════════════════════════════════════════════════════════════════════════
STEP 1: FUNDAMENTAL PROFILE CLASSIFICATION
════════════════════════════════════════════════════════════════════════

Extract the following from the fundamentals snapshot and classify each:

  1. beta_class:      "low" if Beta < 0.8, "normal" if 0.8 <= Beta <= 1.5, \
                       "high" if Beta > 1.5, "missing" if Beta is absent
  2. market_cap_class: "mega" if Market Cap > 200B, "mid" if 10B <= Market \
                       Cap <= 200B, "small" if Market Cap < 10B, "missing" \
                       if absent
  3. dividend_class:   "high" if Dividend Yield > 0.02, "low" if <= 0.02 \
                       or absent
  4. profitability_class: "high" if ROE > 0.20 AND Operating Margin > 0.25, \
                       "low" if ROE < 0.10 OR Operating Margin < 0.10, \
                       "moderate" otherwise
  5. debt_class:       "high_risk" if D/E > 2.0 OR Current Ratio < 1.0, \
                       "strong" if D/E < 0.5 AND Current Ratio > 2.0, \
                       "moderate" otherwise
  6. pe_class:         "expensive" if PE Ratio (TTM) > 35 OR Forward PE > 30, \
                       "cheap" if PE Ratio (TTM) < 12, "moderate" otherwise
  7. sector:           The Sector field from fundamentals (or "Unknown")

════════════════════════════════════════════════════════════════════════
STEP 2: MANDATORY COMPUTATION PLAN
════════════════════════════════════════════════════════════════════════

Using your fundamental profile classification AND the horizon-specific \
calibration rules provided separately, you MUST compute ALL mandatory \
indicators listed in the HORIZON_FOCUS section with EXACT parameters. \
Do NOT substitute, omit, or change parameters.  Your only additions \
should be: (a) any additional indicators triggered by calibration rules \
(e.g. ATR(10) instead of ATR(5) for low-beta stocks), and (b) the \
calibration-specific parameter adjustments documented in Step 1.

Every metric in the mandatory indicator set MUST appear in your plan with \
its exact parameter (e.g. RSI(2), not RSI(14); Stochastic(5,3,3), not \
Stochastic(14,3,3)).  The regime gate (ER and PTH) determines which \
normalisation mode to use for mean-reversion indicators:

  RANGING mode (ER < 0.20, or ER 0.20-0.40 with PTH < 0.95):
    Mean-reversion indicators (RSI, Stochastic, BB %B) use INVERTED \
    normalisation: oversold = bullish (+1), overbought = bearish (-1).
  TRENDING mode (ER >= 0.40, or ER 0.20-0.40 with PTH >= 0.95 and \
    volume_ratio > 1.2):
    Mean-reversion indicators use STANDARD normalisation: overbought = \
    bullish (+1), oversold = bearish (-1).
  Trend-following indicators (EMA crossover, price vs EMA, ROC, MACD \
    histogram) always use standard normalisation regardless of regime.

APPLYING CALIBRATION RULES:
  For EACH rule that matches your fundamental profile classification, you MUST:
  - Adjust window sizes, band multipliers, and lookback periods as specified.
  - Add or remove indicators as specified.
  - Shift mu weights as specified.
  - Adjust sigma blend weights and multipliers as specified.
  - Include dividend-adjusted close computation if dividend_class is "high".

  Document which rules fired in each metric's "metric_rationale" field.

For EVERY metric, write a clear computation_instruction that specifies:
  - Exact window sizes and parameters (e.g. RSI(14), not just "RSI")
  - The exact formula or library call
  - How to normalise the result to [-1, +1] for directional signals
  - Which calibration rule(s) influenced the parameter choice

Allowed libraries: math, statistics, numpy, pandas, talib, scipy, scikit-learn.

IMPORTANT — mu and sigma derivation (FIXED CONSTANTS — do NOT change):

After computing the individual metrics above, the code agent will derive
mu and sigma using the formulas below.  You MUST reference these exact
constants and formulas in your plan.

════════════════════════════════════════════════════════════════════════
NORMALIZATION REFERENCE TABLE
════════════════════════════════════════════════════════════════════════

Every directional signal MUST be normalised to [-1, +1] using EXACTLY the
formula listed for its indicator type.  Do NOT invent new formulas.

  +1 = bullish,  -1 = bearish  (for all categories)

MEAN-REVERSION INDICATORS — regime-dependent normalisation:
  These indicators MUST be inverted in RANGING mode so that oversold = +1
  (bullish, expecting upward reversion) and overbought = -1 (bearish,
  expecting downward reversion).

  RSI(N):
    RANGING:  norm = clip((50 - rsi) / 40, -1, +1)  [RSI<10 = +1 bullish]
    TRENDING: norm = clip((rsi - 50) / 50, -1, +1)  [RSI>90 = +1 bullish]
  Stochastic %K:
    RANGING:  norm = clip((50 - %K) / 40, -1, +1)  [%K<10 = +1 bullish]
    TRENDING: norm = clip((%K - 50) / 50, -1, +1)  [%K>90 = +1 bullish]
  BB %B:
    RANGING:  norm = clip(1 - 2*%B, -1, +1)  [%B<0 = +1 bullish]
    TRENDING: norm = clip(2*(%B - 0.5), -1, +1)  [%B>1 = +1 bullish]
  Williams %R:
    RANGING:  norm = clip((-50 - %R) / 40, -1, +1)  [%R<-80 = +1 bullish]
    TRENDING: norm = clip((%R + 50) / 50, -1, +1)  [%R>-20 = +1 bullish]

TREND-FOLLOWING INDICATORS — same normalisation in both modes:
  z-score vs SMA:      norm = clip(-z / 2, -1, +1)  [negative z = bullish]
  MACD histogram:      norm = clip(histogram / atr5, -1, +1)
  ROC(N):              norm = clip(roc / (3 * atr5 / close), -1, +1)
  EMA crossover:       +1 if fast > slow, -1 if fast < slow, 0 otherwise
  Price vs EMA:        norm = clip((price - ema) / (2 * atr5 / close), -1, +1)
  Lin-reg slope:       norm = clip(slope * 252 / (2 * rv), -1, +1)
  ADX:                 NOT directional — use as trend-strength filter only

VOLUME INDICATORS — same normalisation in both modes:
  OBV slope:           norm = clip(annualised_obv_change / 0.50, -1, +1)
  CMF:                 norm = clip(value * 2, -1, +1)
  Volume ratio:        norm = clip((vr - 1.0) / 0.5, -1, +1)

NON-DIRECTIONAL (use as sigma or regime inputs only, NOT in mu):
  S/R ratio:            norm = clip(2 * (sr_ratio - 0.5), -1, +1)  \
     [sr_ratio > 0.5 = closer to support = bullish; < 0.5 = bearish]
  dist_to_support:     NOT directional — use as zone_width input only
  dist_to_resistance:  NOT directional — use as zone_width input only
  zone_width:          NOT directional — use as zone_width_mult input only
  Kaufman ER:          NOT directional — use as REGIME GATE trigger only
  PTH (price/high):   NOT directional for mu — regime gate trigger only
  Yang-Zhang vol:     NOT directional — sigma input only
  EWMA vol:           NOT directional — sigma input only
  ATR:                NOT directional — normalisation scaling factor only

In each metric's computation_instruction, you MUST state which row of this
table applies and write the exact formula.

════════════════════════════════════════════════════════════════════════
MU FORMULA (DRIFT_SCALAR is horizon-specific)
════════════════════════════════════════════════════════════════════════

  DRIFT_SCALAR: default 0.15 for long/medium-term; the horizon focus \
above may specify a different value for short-term (currently 0.05).

  composite_signal = sum(weight_i * norm_signal_i)   for all i
  dividend_carry   = dividend_yield  if dividend_class = "high", else 0.0
  mu = clip(composite_signal * DRIFT_SCALAR + dividend_carry, -0.08, +0.08)

  Use the DRIFT_SCALAR value from the horizon focus section if it \
specifies one; otherwise use 0.15.

Your plan MUST include a metric called "signal_weighted_mu" whose
computation_instruction specifies:
  – which metrics to use as directional signals (including S/R ratio),
  – which NORMALIZATION REFERENCE TABLE row applies to each,
  – the weight for each signal (must sum to 1.0 after calibration transfers,
    including the sup_res weight),
  – the exact mu formula above (composite_signal * DRIFT_SCALAR + dividend_carry),
  – which calibration rules modified the weights from the base allocation.
  – FOR SHORT-TERM ONLY: the REGIME GATE logic that switches between
    RANGING MODE weights (mean_rev dominant) and TRENDING MODE weights
    (momentum dominant) based on Kaufman Efficiency Ratio and PTH.

If the horizon focus specifies a REGIME GATE, the plan MUST also include
a metric called "kaufman_efficiency_ratio" that computes the ER value
used by the gate, and a metric called "price_to_high_ratio" that computes
PTH = current_price / max(high, 252d lookback).

════════════════════════════════════════════════════════════════════════
SIGMA FORMULA (matches HORIZON_FOCUS definition)
════════════════════════════════════════════════════════════════════════

Long-term:  sigma = max(credit_mult * regime_mult * zone_width_mult
            * (sigma_blend_r * rv252 + sigma_blend_yz * yz252), 0.01)
Medium-term: sigma = max(credit_mult * regime_mult * zone_width_mult
            * (sigma_blend_r * rv63 + sigma_blend_yz * yz63), 0.01)
            yz63 = Yang-Zhang vol annualised over 63d
Short-term: sigma = max(credit_mult * beta_sigma_mult * regime_mult
            * zone_width_mult
            * (sigma_blend_r * ewma_vol_ann + sigma_blend_yz * yz_ann), 0.01)
            ewma_vol_ann = EWMA-smoothed realised vol annualised
              (lambda=0.94, seeded with 10d sample variance)
            yz_ann = Yang-Zhang vol annualised over 10d

  regime_mult: 1.25 if vol_pctile > 80; 0.90 if vol_pctile < 20; else 1.0
  zone_width_mult: from HORIZON_FOCUS (range compression/expansion
    based on zone_width = (R - S) / price).  Tight zones widen sigma
    (breakout risk); wide zones compress sigma.
  All blend weights and multipliers come from the calibration rules in
  HORIZON_FOCUS.

Your plan MUST include a metric called "regime_adjusted_sigma" whose
computation_instruction specifies:
  – which volatility estimators to compute,
  – the exact blend weights from the calibration rules,
  – the exact sigma formula for this horizon (including zone_width_mult),
  – how zone_width is derived from support/resistance metrics,
  – which calibration rules modified the blend from the base allocation.

════════════════════════════════════════════════════════════════════════
STEP 3: STRUCTURAL VALIDATION (do NOT skip)
════════════════════════════════════════════════════════════════════════

Before returning your plan, verify ALL of the following:

  A. DATA SUFFICIENCY: The available data has enough rows for the longest
     lookback window you request. If the data has N rows, every window size
     W must satisfy W < N. Remove or shorten any metric whose window exceeds N.

  B. PARAMETER SANITY: All EMA/SMA windows are positive integers. RSI periods
     are >= 2. Bollinger Band multipliers are > 0. No division by zero is
     possible in any formula.

  C. HORIZON ALIGNMENT: Short-term metrics use short windows (5-10 days),
     medium-term use medium windows (14-63 days), long-term use long windows
     (50-252 days). Do not mix e.g. a 200-day indicator into a short-term plan.

  D. CROSS-METRIC CONSISTENCY: Every metric name referenced in
     signal_weighted_mu's computation_instruction must exist as a separate
     metric in the plan. Every volatility estimator referenced in
     regime_adjusted_sigma's instruction must exist as a separate metric.
     signal_concordance must reference the same weights and signals as
     signal_weighted_mu. The sup_res weight in mu MUST reference a
     support/resistance metric.
     zone_width referenced in sigma MUST come from S/R metrics.
     Weight sums must equal 1.0.

  E. MU/SIGMA/CONCORDANCE PRESENCE: The plan MUST contain "signal_weighted_mu",
     "regime_adjusted_sigma", and "signal_concordance" as named metrics.
     The plan MUST also contain at least one support/resistance metric that
     produces S, R, S/R ratio, and zone_width.

════════════════════════════════════════════════════════════════════════
SIGNAL CONCORDANCE (required metric in the plan)
════════════════════════════════════════════════════════════════════════

  Your plan MUST include a metric called "signal_concordance" whose
  computation_instruction specifies:
    – concordance = |sum(weight_i * norm_signal_i)| / sum(|weight_i * norm_signal_i|)
    – This ranges from 0 (complete disagreement) to 1 (perfect agreement)
    – Exclude null signals from both numerator and denominator
    – Use the same weights and normalised signals as signal_weighted_mu

If any check fails, fix the plan before returning it.

Return ONLY the JSON computation plan."""

# ── Phase 2 system prompt (code agent domain context) ───────────────────────

PHASE2_PROMPT = """\
You are an expert quantitative technical analyst writing Python code.
Your outputs feed directly into a portfolio-management system — they must be
numerically precise, internally consistent, and fully traceable.

Available data: daily CSV with columns
  Date, Open, High, Low, Close, Volume, Dividends, Stock Splits.

════════════════════════════════════════════════════════════════════════
DATA HANDLING
════════════════════════════════════════════════════════════════════════

  • Dividends — if non-zero entries exist, compute dividend_adjusted_close
    as Close corrected for cumulative dividends (backward-adjust) before
    calculating ANY return-based or trend metric. Use raw Close only for
    intraday range ratios (High-Low)/Close.
    If the plan specifies dividend_class = "high", this adjustment is
    MANDATORY.
  • Stock Splits — if non-zero entries exist (values != 0.0), adjust
    Open/High/Low/Close and Volume for splits before any computation.
    Split factor on date t means price was divided by that factor.
  • If both columns are all zeros, skip adjustments and use raw prices.

════════════════════════════════════════════════════════════════════════
CORE COMPETENCIES
════════════════════════════════════════════════════════════════════════

Trend analysis, momentum oscillators, volatility estimation (close-to-close,
Parkinson, Yang-Zhang, ATR), volume-price analysis, regime classification,
mean-reversion statistics, and support/resistance level identification
(pivot points, swing highs/lows, volume-profile clusters).

════════════════════════════════════════════════════════════════════════
NORMALISATION REFERENCE TABLE (MANDATORY — same table as Phase 1)
════════════════════════════════════════════════════════════════════════

Every directional signal that feeds into mu MUST be normalised to [-1, +1]
using EXACTLY the formula listed for its indicator type.  Do NOT invent
your own formula.  +1 = bullish, -1 = bearish.

MEAN-REVERSION INDICATORS — regime-dependent normalisation:
  These indicators MUST be inverted in RANGING mode so that oversold = +1
  (bullish, expecting upward reversion) and overbought = -1 (bearish).

  RSI(N):
    RANGING:  norm = clip((50 - rsi) / 40, -1, +1)  [RSI<10 = +1 bullish]
    TRENDING: norm = clip((rsi - 50) / 50, -1, +1)  [RSI>90 = +1 bullish]
  Stochastic %K:
    RANGING:  norm = clip((50 - %K) / 40, -1, +1)  [%K<10 = +1 bullish]
    TRENDING: norm = clip((%K - 50) / 50, -1, +1)  [%K>90 = +1 bullish]
  BB %B:
    RANGING:  norm = clip(1 - 2*%B, -1, +1)  [%B<0 = +1 bullish]
    TRENDING: norm = clip(2*(%B - 0.5), -1, +1)  [%B>1 = +1 bullish]
  Williams %R:
    RANGING:  norm = clip((-50 - %R) / 40, -1, +1)  [%R<-80 = +1 bullish]
    TRENDING: norm = clip((%R + 50) / 50, -1, +1)  [%R>-20 = +1 bullish]

TREND-FOLLOWING INDICATORS — same normalisation in both modes:
  z-score vs SMA:      norm = clip(-z / 2, -1, +1)  [negative z = bullish]
  MACD histogram:      norm = clip(histogram / atr5, -1, +1)
  ROC(N):              norm = clip(roc / (3 * atr5 / close), -1, +1)
  EMA crossover:       +1 if fast > slow, -1 if fast < slow, 0 otherwise
  Price vs EMA:        norm = clip((price - ema) / (2 * atr5 / close), -1, +1)
  Lin-reg slope:       norm = clip(slope * 252 / (2 * rv), -1, +1)
  ADX:                 NOT directional — use as trend-strength filter only

VOLUME INDICATORS — same normalisation in both modes:
  OBV slope:           norm = clip(annualised_obv_change / 0.50, -1, +1)
  CMF:                 norm = clip(value * 2, -1, +1)
  Volume ratio:        norm = clip((vr - 1.0) / 0.5, -1, +1)

NON-DIRECTIONAL (use as sigma or regime inputs only, NOT in mu):
  S/R ratio:            norm = clip(2 * (sr_ratio - 0.5), -1, +1)
     [sr_ratio > 0.5 = closer to support = bullish; < 0.5 = bearish]
  dist_to_support:     NOT directional — use as zone_width input only
  dist_to_resistance:  NOT directional — use as zone_width input only
  zone_width:          NOT directional — use as zone_width_mult input only
  Kaufman ER:          NOT directional — use as REGIME GATE trigger only
  PTH (price/high):   NOT directional for mu — regime gate trigger only
  Yang-Zhang vol:     NOT directional — sigma input only
  EWMA vol:           NOT directional — sigma input only
  ATR:                NOT directional — normalisation scaling factor only

  clip(x, lo, hi) = max(lo, min(hi, x))

════════════════════════════════════════════════════════════════════════
MU COMPUTATION (DRIFT_SCALAR is horizon-specific)
════════════════════════════════════════════════════════════════════════

  1. Compute every directional metric in the plan (including S/R ratio).
  2. Normalise each signal to [-1, +1] using the table above.
  3. Apply the weights from the plan's "signal_weighted_mu" instruction
     (including the sup_res weight for S/R ratio).
  4. Compute the composite:
       composite_signal = sum(weight_i * norm_signal_i)
  5. Determine dividend_carry from the plan:
       If dividend_class = "high":
         long_term:   dividend_carry = dividend_yield
         medium_term:  dividend_carry = dividend_yield
         short_term:  dividend_carry = dividend_yield * 12 / 252
       Else: dividend_carry = 0.0
  6. Apply the mu formula:
       DRIFT_SCALAR: default 0.15 for long/medium-term; the horizon focus
       above may specify a different value for short-term (currently 0.05).
       Use the DRIFT_SCALAR from the horizon focus section if specified;
       otherwise use 0.15.
       mu = clip(composite_signal * DRIFT_SCALAR + dividend_carry, -0.08, +0.08)

  mu MUST be traceable back to the individual signal values and weights.

  FOR SHORT-TERM ONLY — REGIME GATE WEIGHT SWITCHING:
  If the horizon focus specifies a REGIME GATE, you MUST apply it before
  computing the composite signal:
    1. Compute Kaufman Efficiency Ratio ER(10).
    2. Compute Price-to-High Ratio PTH = price / max(high, 252d).
    3. Determine mode:
       RANGING (default): ER < 0.20, OR (ER 0.20-0.40 AND PTH < 0.95)
         → use RANGING MODE weights (mean_rev dominant)
       TRENDING: ER >= 0.40, OR (ER 0.20-0.40 AND PTH >= 0.95 AND
         volume_ratio > 1.2)
         → use TRENDING MODE weights (momentum dominant)
    4. Apply the selected weight set, then fundamental calibration transfers,
       then renormalise.
    5. Record which mode was selected in the mu trace inputs.

════════════════════════════════════════════════════════════════════════
SIGNAL CONCORDANCE COMPUTATION (required — add to computed_metrics)
════════════════════════════════════════════════════════════════════════

  After computing mu, you MUST also compute signal_concordance:

    concordance = |sum(weight_i * norm_signal_i)| / sum(|weight_i * norm_signal_i|)

  where the same weights and normalised signals from the mu computation
  are used.  Exclude null signals from both numerator and denominator.

  This ranges from 0 (complete disagreement — signals cancel) to 1
  (perfect agreement — all point the same way).

  Add "signal_concordance" to computed_metrics with its value, and add
  a corresponding entry to computation_traces with the (weight, norm_signal)
  pairs as trace inputs.

════════════════════════════════════════════════════════════════════════
SIGMA COMPUTATION (FIXED FORMULAS per horizon)
════════════════════════════════════════════════════════════════════════

  Use the EXACT formula for the current horizon.  Do NOT modify it.

  Long-term:
    sigma = max(credit_mult * regime_mult * zone_width_mult
                * (sigma_blend_r * rv252 + sigma_blend_yz * yz252), 0.01)

  Medium-term:
    yz63 = Yang-Zhang vol annualised over 63d
    sigma = max(credit_mult * regime_mult * zone_width_mult
                * (sigma_blend_r * rv63 + sigma_blend_yz * yz63), 0.01)

  Short-term:
    ewma_vol = EWMA-smoothed realised vol annualised.
      lambda = 0.92 (optimised for 1-week horizon; half-life ≈ 8 trading days)
      ewma_vol_t = lambda * ewma_vol_{t-1} + (1-lambda) * daily_return^2
      ewma_vol_ann = sqrt(ewma_vol_t * 252)
      Seed ewma_vol_0 with 10d sample variance.
    yz_ann = Yang-Zhang vol annualised over 10d.
      yz = sqrt(open_vol + close_vol - open_close_corr)
      (see HORIZON_FOCUS for full formula)
    adj_product = clip(credit_mult * beta_sigma_mult * regime_mult * zone_width_mult, 0.80, 1.50)
    sigma = max(adj_product * (sigma_blend_r * ewma_vol_ann + sigma_blend_yz * yz_ann), 0.01)
      ↑ CAP prevents multiplicative adjustments from stacking beyond ±50%

  Shared parameters (from the plan's calibration rules):
    regime_mult:    1.25 if vol_pctile > 80;  0.90 if vol_pctile < 20;  else 1.0
    credit_mult:    1.15 if D/E > 2.0 OR Current Ratio < 1.0;
                    0.90 if D/E < 0.5 AND Current Ratio > 2.0;  else 1.0
    beta_sigma_mult: 1.10 if Beta > 1.5 (short-term only); else 1.0
    zone_width_mult: from HORIZON_FOCUS — tight S/R zones (low zone_width)
                    widen sigma (breakout risk), wide zones compress sigma.
                    zone_width = (R - S) / price from support/resistance metrics.
    All blend weights (sigma_blend_r, sigma_blend_yz, etc.) come from the
    plan's calibration rules.

════════════════════════════════════════════════════════════════════════
STRUCTURAL VALIDATION (apply AFTER computing all metrics)
════════════════════════════════════════════════════════════════════════

After computing all metrics, validate the following invariants.  If any
check fails, adjust the value or set it to a safe default:

  1. RANGE CHECKS:
     - RSI-type values MUST be in [0, 100].  If outside, clip.
     - Stochastic %K/%D MUST be in [0, 100].  If outside, clip.
     - Williams %R MUST be in [-100, 0].  If outside, clip.
     - mu MUST be in [-0.08, +0.08].  If outside, clamp to nearest bound.
     - sigma MUST be >= 0.01 (floor).  If below, set to 0.01.
     - signal_concordance MUST be in [0, 1].  If outside, clip.
     - Normalised directional signals MUST be in [-1, +1].  If outside, clip.
     - Composite signal (weighted sum) MUST be in [-1, +1].  If outside, clip.

  2. CONSISTENCY CHECKS:
     - Every metric referenced in signal_weighted_mu MUST have been computed
       and appear in computed_metrics.
     - Every estimator referenced in regime_adjusted_sigma MUST have been
       computed and appear in computed_metrics.
     - Support S MUST be < current price; resistance R MUST be > current price.
       If S > price or R < price, re-identify the nearest correct levels.
     - zone_width MUST be > 0. If zero or negative, set zone_width_mult = 1.0.
     - Weight sums in mu computation MUST equal 1.0 (allow tolerance of 0.01).

  3. TRACE COMPLETENESS:
     - Every metric in computed_metrics MUST have a matching computation_trace
       with the same trace_id.
     - mu_trace_id and sigma_trace_id MUST each match a trace in
       computation_traces.
     - The trace "code" field MUST contain the full function source.

  4. NULL HANDLING:
     - If a metric cannot be computed from the available data, set its value
       to null (None in Python -> null in JSON) rather than 0 or NaN.
     - If a directional signal is null, exclude it from the mu composite and
       renormalise remaining weights to sum to 1.0.
     - If more than half the directional signals are null, set mu = 0.0.

  5. FORMULA ADHERENCE:
     - DRIFT_SCALAR MUST be the value specified in the horizon focus section
       (0.05 for short-term, 0.15 for medium/long-term) — no other values.
     - mu MUST be computed as clip(composite_signal * DRIFT_SCALAR + dividend_carry,
       -0.08, +0.08) — no alternative formulas.
     - sigma MUST use the exact horizon-specific formula from the SIGMA
       COMPUTATION section above (including adj_product cap) — no alternative
       formulas.
     - The product of all sigma multipliers (credit_mult * beta_sigma_mult *
       regime_mult * zone_width_mult) MUST be clipped to [0.80, 1.50] to
       prevent over-adjustment stacking.
     - zone_width MUST be computed as (R - S) / price where S < price < R.
     - FOR SHORT-TERM: Mean-reversion indicators (RSI, Stochastic, BB %B,
       Williams %R) MUST use RANGING normalisation when the regime gate
       determines RANGING mode, and TRENDING normalisation when TRENDING.
       This is critical — using trend-following normalisation for mean-reversion
       indicators in a ranging market produces inverted signals.
     - z-score MUST use clip(-z / 2, -1, +1) for short-term (not /3).
     - Each directional signal MUST use the normalisation formula from the
       NORMALISATION REFERENCE TABLE — no alternative normalisations.
     - signal_concordance MUST be computed as
       |sum(weight_i * norm_signal_i)| / sum(|weight_i * norm_signal_i|)
       — no alternative formulas.

════════════════════════════════════════════════════════════════════════
RULES
════════════════════════════════════════════════════════════════════════

  • Follow the computation plan exactly; do not skip any requested metric.
  • Every scalar result (including mu and sigma) must have its own
    computation trace so the reasoning chain is fully auditable.
  • Produce conservative, evidence-based estimates.  Do not extrapolate
    beyond what the data supports.
  • All rates are annualised decimals (0.12 = 12%, not 12 or "12%").
  Allowed libraries: math, statistics, numpy, pandas, talib, scipy, scikit-learn."""

# ── Phase 3 system prompt (thesis — interpret results) ──────────────────────

PHASE3_PROMPT = """\
You are a senior technical analyst.  You are given:
  1. A set of computed technical metrics — each with a numeric value and a
     computation_trace_id linking it to the exact code that produced it.
  2. The fundamental profile classification and calibration rules that were
     applied when selecting and parameterizing these metrics (documented in
     each metric's metric_rationale field).

Your job:
1. Interpret what each metric's VALUE means for this stock right now
   (value_interpretations).  Reference the fundamental context where relevant
   (e.g. "RSI(14)=72 is overbought, especially notable given this is a
   high-beta tech stock where momentum extremes tend to persist").

2. Synthesise the signals into a coherent investment thesis.  Ground every
   claim in a specific metric — cite it inline as
   [metric_name | trace:<computation_trace_id>].

   The thesis MUST address how the fundamental profile influenced the
   analysis.  For example:
   - If beta_class = "high", explain how the tighter/faster parameters
     affect the interpretation of momentum signals.
   - If dividend_class = "high", note that mu includes a dividend carry
     component and separate it from the pure price-signal component.
   - If debt_class = "high_risk", discuss the wider sigma and what it
     implies for position sizing.
   - If price is near support (S/R ratio > 0.5), discuss the bullish
     bias from support proximity and reduced downside; if near resistance
     (S/R ratio < 0.5), discuss the bearish headwind and potential
     breakout/breakdown scenarios.
   - If zone_width is narrow (range compression), flag the elevated
     breakout risk and explain how zone_width_mult widens sigma to
     account for this.

3. Explain how the individual signals were aggregated into mu and sigma
   (reference the signal_weighted_mu and regime_adjusted_sigma metrics).
   State the explicit weight allocation and which calibration rules modified
   it from the base allocation.  For sigma, explicitly explain the
   zone_width_mult and how the S/R zone width influenced the final sigma.

4. Identify catalysts (bullish signals) and risks (bearish signals),
   each tied to a specific metric and trace.  Frame risks in terms of the
   fundamental profile where applicable (e.g. "high D/E ratio amplifies
   volatility risk during market stress").  Include proximity to support
   as a bullish catalyst and proximity to resistance as a bearish risk.

5. Do NOT set conviction — it is computed deterministically from
   signal_concordance, regime_clarity, signal_strength, and data_completeness
   by the system.  Leave the "conviction" field as 0.0; it will be
   overridden automatically.

6. REGIME AND NORMALISATION (REQUIRED for short-term analysis):
   Your thesis MUST explicitly state:
   - Which regime mode was active (RANGING or TRENDING) based on Kaufman ER
     and PTH, including the exact ER and PTH values.
   - Whether mean-reversion indicators (RSI, Stochastic, BB %B) were inverted
     (they are inverted in RANGING mode — oversold = bullish).
   - The regime_clarity assessment: ER < 0.20 or ER > 0.40 = clear regime
     (regime_clarity = 1.0); ER between 0.20-0.40 = ambiguous regime
     (regime_clarity = 0.5).  Explain how this affects conviction.

Do NOT introduce any claim that is not backed by a computed metric or the
stated fundamental profile."""


# ── Data gathering (deterministic, no LLM) ──────────────────────────────────


def gather_technical_data(ticker: str, trade_date: str, lookback_days: int) -> dict:
    """Fetch OHLCV data and a comprehensive set of technical indicators."""
    from src.agents.utils.agent_utils import get_stock_data

    start_date, end_date = compute_date_range(trade_date, lookback_days)

    data = {}
    data["stock_data"] = get_stock_data.invoke(
        {"symbol": ticker, "start_date": start_date, "end_date": end_date}
    )

    return data


# ── Node factory ─────────────────────────────────────────────────────────────


def create_market_analyst(reasoning_llm, code_agent, research_depth="medium"):
    """Create a market/technical analyst node for the outer AgentState graph.

    Parameters
    ----------
    reasoning_llm  : LangChain chat model for Phase 1 (plan) and Phase 3 (thesis)
    code_agent     : CodeValidationAgent instance for Phase 2 (compute)
    research_depth : "shallow" | "medium" | "deep"
    """
    analyst_config = {
        "agent_type": AgentType.TECHNICAL,
        "state_key": "market_report",
        "gather_fn": gather_technical_data,
        "phase1_system_prompt": PHASE1_PROMPT,
        "phase2_system_prompt": PHASE2_PROMPT,
        "phase3_system_prompt": PHASE3_PROMPT,
        "horizon_focus": HORIZON_FOCUS,
        "active_horizons": (
            "short_term",
        ),  # Only run short-term; LT/MT get zero placeholders
    }

    return create_analyst_node(
        reasoning_llm,
        code_agent,
        analyst_config,
        research_depth,
        verbose=True,
    )


if __name__ == "__main__":
    import pathlib
    import statistics

    # Simple local run settings (plain text, no CLI args).
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    # market_analyst.py lives at src/agents/analysts/ — 3 levels up = Multi-Agent-AI-PM/
    _project_root = str(pathlib.Path(__file__).resolve().parents[3])

    ticker = "AAPL"
    trade_date = today_str
    research_depth = "shallow"
    num_runs = 1  # number of repeated runs

    llm_provider = "ollama"
    llm_model = "minimax-m2.7:cloud"
    llm_base_url = "http://localhost:11434/v1"
    llm_max_retries = 10

    code_model = "minimax-m2.7:cloud"
    code_timeout = 60
    code_max_iterations = 5

    llm_client = create_llm_client(
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        max_retries=llm_max_retries,
        reasoning_effort="high",
        temperature=0,
        seed=42,
    )
    reasoning_llm = llm_client.get_llm()

    code_agent = CodeValidationAgent(
        model=code_model,
        timeout=code_timeout,
        max_iterations=code_max_iterations,
        analyst_type="technical",
        project_root=_project_root,
        verbose=False,
    )

    market_node = create_market_analyst(
        reasoning_llm,
        code_agent,
        research_depth,
    )

    init_state = {
        "company_of_interest": ticker,
        "trade_date": trade_date,
    }

    # ── Collect mu, sigma, and conviction from each run ──────────────────────
    horizons = ["long_term", "medium_term", "short_term"]
    mu_runs: dict[str, list[float]] = {h: [] for h in horizons}
    sigma_runs: dict[str, list[float]] = {h: [] for h in horizons}
    conviction_runs: dict[str, list[float]] = {h: [] for h in horizons}

    for run_idx in range(1, num_runs + 1):
        print(
            f"\n{'=' * 60}\n"
            f"  Run {run_idx}/{num_runs} — {ticker} on {trade_date} "
            f"(depth={research_depth})\n"
            f"{'=' * 60}",
            flush=True,
        )

        result = market_node(init_state)
        report_json = result.get("market_report", "{}")

        try:
            report = ResearchReport.model_validate_json(report_json)
            print(f"\n--- Run {run_idx} Report ---")
            print(report.model_dump_json(indent=2))

            for h in horizons:
                mu_runs[h].append(getattr(report.mu, h))
                sigma_runs[h].append(getattr(report.sigma_contribution, h))
                conviction_runs[h].append(getattr(report.conviction, h))

        except Exception as exc:  # noqa: BLE001
            print(f"\nRun {run_idx}: Failed to parse ResearchReport. Reason:", exc)
            print(report_json)

    # ── Deviation summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DEVIATION SUMMARY  ({num_runs} runs, {ticker} @ {trade_date})")
    print(f"{'=' * 60}\n")

    for label, runs in [
        ("mu", mu_runs),
        ("sigma_contribution", sigma_runs),
        ("conviction", conviction_runs),
    ]:
        print(f"  {label}:")
        for h in horizons:
            values = runs[h]
            if len(values) < 2:
                print(f"    {h}: only {len(values)} successful run(s) — skipping")
                continue
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            spread = max(values) - min(values)
            print(
                f"    {h}:  values={[round(v, 6) for v in values]}  "
                f"mean={mean:.6f}  stdev={stdev:.6f}  spread={spread:.6f}"
            )
        print()


# if __name__ == "__main__":
#     import pathlib
#     import statistics
#     import math
#     import time
#     import yfinance as yf
#     import pandas as pd

#     # Simple local run settings (plain text, no CLI args).
#     today_str = datetime.now().strftime("%H:%M:%S")
#     # market_analyst.py lives at src/agents/analysts/ — 3 levels up = Multi-Agent-AI-PM/
#     _project_root = str(pathlib.Path(__file__).resolve().parents[3])

#     # ── Configuration ────────────────────────────────────────────────────────
#     ticker = "AAPL"
#     research_depth = "shallow"
#     start_date = "2026-01-01"
#     end_date = "2026-03-31"
#     initial_capital = 100_000.0

#     llm_provider = "ollama"
#     llm_model = "minimax-m2.7:cloud"
#     llm_base_url = "http://localhost:11434/v1"
#     llm_max_retries = 10

#     code_model = "minimax-m2.7:cloud"
#     code_timeout = 60
#     code_max_iterations = 5

#     # ── Setup LLM and code agent ───────────────────────────────────────────
#     llm_client = create_llm_client(
#         provider=llm_provider,
#         model=llm_model,
#         base_url=llm_base_url,
#         max_retries=llm_max_retries,
#         reasoning_effort="high",
#         temperature=0,
#         seed=42,
#     )
#     reasoning_llm = llm_client.get_llm()

#     code_agent = CodeValidationAgent(
#         model=code_model,
#         timeout=code_timeout,
#         max_iterations=code_max_iterations,
#         analyst_type="technical",
#         project_root=_project_root,
#         verbose=True,
#     )

#     market_node = create_market_analyst(
#         reasoning_llm,
#         code_agent,
#         research_depth,
#     )

#     # ── Get trading days from yfinance ──────────────────────────────────────
#     print(f"Fetching {ticker} trading days from {start_date} to {end_date}...")
#     price_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
#     if price_df.empty:
#         raise SystemExit(f"No price data for {ticker} in [{start_date}, {end_date}]")
#     trading_days = sorted(price_df.index.strftime("%Y-%m-%d").tolist())
#     print(
#         f"Found {len(trading_days)} trading days: {trading_days[0]} to {trading_days[-1]}"
#     )

#     # ── Backtest: run analyst each day, build position tracker ───────────────
#     # Short-term only; short selling enabled
#     # NO LOOK-AHEAD: decide after close, execute at next day's Open.
#     # Transaction costs: commission $0.005/share + slippage 5bps.
#     horizons = ["short_term"]

#     cash = initial_capital
#     shares = 0.0  # positive=long, negative=short
#     portfolio_value = initial_capital  # initialised before first iteration
#     peak = initial_capital  # for drawdown tracking
#     portfolio_values: list[float] = []
#     daily_returns: list[float] = []

#     # Transaction cost parameters
#     COMMISSION_PER_SHARE = 0.005  # $0.005/share (IBKR-like)
#     SLIPPAGE_BPS = 5  # 0.05% of execution price

#     backtest_log: list[dict] = []

#     print(f"\n{'=' * 80}")
#     print(
#         f"  BACKTEST: {ticker}  |  {start_date} → {end_date}  |  Capital: ${initial_capital:,.0f}  |  Short-Term Only + Short Selling"
#     )
#     print(f"  Execution: next-day Open  |  Costs: ${COMMISSION_PER_SHARE}/share + {SLIPPAGE_BPS}bps slippage")
#     print(f"{'=' * 80}\n")

#     # pending_signal stores (target_frac, mu, sigma, conv, sigma_ann, sigma_horizon)
#     # from the analyst run on the previous day.
#     # On day i+1 we execute at Open using day i's signal.
#     pending_signal = None

#     try:
#         for day_idx, trade_date in enumerate(trading_days):
#             open_price = float(price_df.loc[trade_date, "Open"])
#             close_price = float(price_df.loc[trade_date, "Close"])
#             t_start = time.time()

#             # ── Execute pending signal from yesterday at today's Open ────────
#             if pending_signal is not None:
#                 (
#                     target_frac,
#                     mu_composite,
#                     sigma_composite,
#                     conv_composite,
#                     sigma_ann,
#                     sigma_horizon,
#                 ) = pending_signal
#                 pending_signal = None

#                 # Execute trades at open_price with transaction costs
#                 portfolio_value_before = cash + shares * open_price

#                 max_short_shares = int(
#                     (initial_capital * 0.50) / open_price
#                 )  # max 50% short
#                 if target_frac > 0:
#                     target_shares = int(
#                         (target_frac * portfolio_value_before) / open_price
#                     )
#                 elif target_frac < 0:
#                     target_shares = -min(
#                         int(
#                             (abs(target_frac) * portfolio_value_before) / open_price
#                         ),
#                         max_short_shares,
#                     )
#                 else:
#                     target_shares = 0

#                 delta = target_shares - shares

#                 # Apply transaction costs to execution price
#                 if delta > 0:
#                     # Buying: pay open_price * (1 + slippage) + commission
#                     exec_price = open_price * (1 + SLIPPAGE_BPS / 10_000)
#                     total_cost = delta * exec_price + delta * COMMISSION_PER_SHARE
#                     if total_cost <= cash:
#                         shares += delta
#                         cash -= total_cost
#                         action = "BUY"
#                     else:
#                         affordable = int(
#                             cash / (exec_price + COMMISSION_PER_SHARE)
#                         )
#                         if affordable > 0:
#                             shares += affordable
#                             cash -= affordable * (
#                                 exec_price + COMMISSION_PER_SHARE
#                             )
#                         action = "BUY_PARTIAL"
#                 elif delta < 0:
#                     shares_to_trade = abs(delta)
#                     if shares >= 0:
#                         # Selling: receive open_price * (1 - slippage) - commission
#                         exec_price = open_price * (1 - SLIPPAGE_BPS / 10_000)
#                         if shares_to_trade <= shares:
#                             proceeds = (
#                                 shares_to_trade * exec_price
#                                 - shares_to_trade * COMMISSION_PER_SHARE
#                             )
#                             shares -= shares_to_trade
#                             cash += proceeds
#                             action = (
#                                 "SELL" if target_shares <= 0 else "SELL_PARTIAL"
#                             )
#                         else:
#                             # Sell all long + open short
#                             exec_price_sell = open_price * (
#                                 1 - SLIPPAGE_BPS / 10_000
#                             )
#                             close_long = shares
#                             open_short = shares_to_trade - close_long
#                             proceeds = (
#                                 shares_to_trade * exec_price_sell
#                                 - shares_to_trade * COMMISSION_PER_SHARE
#                             )
#                             shares = -open_short
#                             cash += proceeds
#                             action = "SHORT"
#                     else:
#                         # Already short
#                         if target_shares < shares:
#                             # Increase short
#                             exec_price = open_price * (
#                                 1 - SLIPPAGE_BPS / 10_000
#                             )
#                             short_proceeds = (
#                                 shares_to_trade * exec_price
#                                 - shares_to_trade * COMMISSION_PER_SHARE
#                             )
#                             cash += short_proceeds
#                             shares -= shares_to_trade
#                             action = "SHORT_MORE"
#                         else:
#                             # Cover short: buy at open + slippage + commission
#                             exec_price = open_price * (
#                                 1 + SLIPPAGE_BPS / 10_000
#                             )
#                             cover_cost = (
#                                 shares_to_trade * exec_price
#                                 + shares_to_trade * COMMISSION_PER_SHARE
#                             )
#                             shares += shares_to_trade
#                             cash -= cover_cost
#                             action = (
#                                 "COVER" if shares >= 0 else "COVER_PARTIAL"
#                             )
#                 else:
#                     action = "HOLD"
#             else:
#                 # No pending signal (first day) — no trade
#                 action = "NO_SIGNAL"
#                 mu_composite = 0.0
#                 sigma_composite = 0.01
#                 conv_composite = 0.0
#                 sigma_ann = 0.01
#                 sigma_horizon = 0.01

#             # Portfolio value at today's Close (marks-to-market after execution)
#             portfolio_value = cash + shares * close_price
#             portfolio_values.append(portfolio_value)

#             # Daily return (Close-to-Close)
#             prev_value = (
#                 portfolio_values[-2] if len(portfolio_values) >= 2 else initial_capital
#             )
#             daily_ret = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0.0
#             daily_returns.append(daily_ret)

#             # Override action label for all-cash positions
#             if shares == 0 and action not in ("BUY", "BUY_PARTIAL"):
#                 action = "CASH"

#             # ── Run the analyst for this day (signal executes TOMORROW) ───────
#             print(
#                 f"start={time.strftime('%H:%M:%S', time.localtime(t_start))}  "
#                 f"[{day_idx + 1}/{len(trading_days)}] {trade_date} — running analyst...",
#                 flush=True,
#             )

#             init_state = {
#                 "company_of_interest": ticker,
#                 "trade_date": trade_date,
#             }

#             try:
#                 result = market_node(init_state)
#                 report_json = result.get("market_report", "{}")
#                 report = ResearchReport.model_validate_json(report_json)
#             except Exception as exc:
#                 print(f"  ⚠ Analyst failed: {exc}")
#                 report = None
#             t_end = time.time()
#             t_elapsed = t_end - t_start

#             # ── Extract signals (short_term only) ──────────────────────────────
#             if report is not None:
#                 mu_st = getattr(report.mu, "short_term", 0.0)
#                 sigma_st = getattr(report.sigma_contribution, "short_term", 0.01)
#                 conv_st = getattr(report.conviction, "short_term", 0.0)
#                 # LT/MT are placeholders (not run)
#                 mu_lt = mu_mt = 0.0
#                 sigma_lt = sigma_mt = 0.01
#                 conv_lt = conv_mt = 0.0

#                 mu_composite_sig = mu_st
#                 sigma_composite_sig = sigma_st
#                 conv_composite_sig = conv_st
#             else:
#                 mu_lt = mu_mt = mu_st = 0.0
#                 sigma_lt = sigma_mt = sigma_st = 0.01
#                 conv_lt = conv_mt = conv_st = 0.0
#                 mu_composite_sig = 0.0
#                 sigma_composite_sig = 0.01
#                 conv_composite_sig = 0.0

#             # ── Position sizing: Sharpe-scaled vol targeting ────────────────
#             # mu is a horizon return (12-day for short-term), sigma is annualised.
#             # Vol targeting sets base position size (industry standard).
#             # Horizon Sharpe ratio (mu/sigma_H) scales position down for weak
#             # signals: Sharpe >= 1.0 → full vol-target, < 1.0 → proportional.
#             # Conviction is a confidence gate (not a position multiplier):
#             #   if conviction < 0.10 → don't trade.

#             TARGET_VOL = 0.15  # 15% annualised target volatility
#             HORIZON_DAYS = {"short_term": 12, "medium_term": 63, "long_term": 252}
#             horizon = "short_term"  # currently short-term only

#             sigma_ann_sig = sigma_composite_sig  # annualised (from EWMA/Yang-Zhang)
#             sigma_horizon_sig = sigma_ann_sig * math.sqrt(
#                 HORIZON_DAYS[horizon] / 252
#             )

#             if (
#                 conv_composite_sig < 0.10
#                 or sigma_horizon_sig <= 0.005
#                 or abs(mu_composite_sig) < 0.003
#             ):
#                 # Insufficient confidence or signal too weak — cash
#                 target_frac_sig = 0.0
#             else:
#                 # Vol-targeted base position (annualised units)
#                 base_frac = TARGET_VOL / max(sigma_ann_sig, 0.05)
#                 # Horizon Sharpe ratio: strong signals → full size
#                 sharpe_h = abs(mu_composite_sig) / max(sigma_horizon_sig, 0.005)
#                 signal_scale = min(sharpe_h, 1.0)
#                 target_frac_sig = base_frac * signal_scale
#                 target_frac_sig = math.copysign(
#                     target_frac_sig, mu_composite_sig
#                 )
#                 # Drawdown throttle: if current drawdown > 10%, scale down
#                 peak = max(peak, portfolio_value)
#                 if peak > 0:
#                     current_dd = (peak - portfolio_value) / peak
#                     if current_dd > 0.10:
#                         dd_scale = math.exp(-3.0 * (current_dd - 0.10))
#                         target_frac_sig *= dd_scale
#                 # Cap at ±100% of portfolio
#                 target_frac_sig = max(-1.0, min(1.0, target_frac_sig))

#             # Store signal for execution at tomorrow's Open
#             pending_signal = (
#                 target_frac_sig,
#                 mu_composite_sig,
#                 sigma_composite_sig,
#                 conv_composite_sig,
#                 sigma_ann_sig,
#                 sigma_horizon_sig,
#             )

#             # ── Logging (uses execution-day metrics from pending_signal) ──────
#             backtest_log.append(
#                 {
#                     "date": trade_date,
#                     "action": action,
#                     "exec_price": open_price,
#                     "close_price": close_price,
#                     "mu_st": mu_st if report else 0.0,
#                     "sigma_st": sigma_st if report else 0.01,
#                     "sigma_horizon": sigma_horizon_sig,
#                     "conv_st": conv_st if report else 0.0,
#                     "mu_composite": mu_composite_sig,
#                     "sigma_composite": sigma_composite_sig,
#                     "conv_composite": conv_composite_sig,
#                     "target_frac": target_frac_sig,
#                     "shares": shares,
#                     "cash": cash,
#                     "portfolio_value": portfolio_value,
#                 }
#             )

#             print(
#                 f"  open=${open_price:.2f}  close=${close_price:.2f}  "
#                 f"mu={mu_composite_sig:+.4f}  "
#                 f"sigma_ann={sigma_ann_sig:.4f}  sigma_H={sigma_horizon_sig:.4f}  "
#                 f"conv={conv_composite_sig:.2f}  "
#                 f"target={target_frac_sig:+.0%}  shares={shares:+.0f}  "
#                 f"portfolio=${portfolio_value:,.2f}  "
#                 f"start={time.strftime('%H:%M:%S', time.localtime(t_start))}  "
#                 f"end={time.strftime('%H:%M:%S', time.localtime(t_end))}  "
#                 f"took={t_elapsed:.1f}s"
#             )

#             # Sleep between runs to avoid rate limits
#             if day_idx < len(trading_days) - 1:
#                 print("  sleeping 60s before next day...", flush=True)
#                 time.sleep(60)
#     except KeyboardInterrupt:
#         print(f"\n\n  *** Interrupted after {day_idx + 1}/{len(trading_days)} days ***")
#         trading_days = trading_days[: day_idx + 1]
#         price_df = price_df.loc[trading_days]

#     # ── Liquidate all open positions at the last trading day's Close ──────
#     last_price = float(price_df.iloc[-1]["Close"])
#     if shares != 0:
#         # Apply transaction costs to liquidation
#         if shares > 0:
#             exec_price = last_price * (1 - SLIPPAGE_BPS / 10_000)
#             proceeds = shares * exec_price - shares * COMMISSION_PER_SHARE
#         else:
#             exec_price = last_price * (1 + SLIPPAGE_BPS / 10_000)
#             proceeds = shares * exec_price - abs(shares) * COMMISSION_PER_SHARE
#         cash += proceeds
#         shares = 0
#         portfolio_values[-1] = cash
#         print(f"\n  Liquidated open positions at ${last_price:.2f} → final cash: ${cash:,.2f}")

#     # ── Compute performance metrics ────────────────────────────────────────
#     if len(portfolio_values) < 2:
#         print("\nNot enough data points to compute performance metrics.")
#         raise SystemExit(1)

#     final_value = cash
#     initial = portfolio_values[0]
#     total_return = (final_value - initial_capital) / initial_capital

#     # Annualised return (simple, based on calendar days)
#     first_date = pd.Timestamp(trading_days[0])
#     last_date = pd.Timestamp(trading_days[-1])
#     n_calendar_days = (last_date - first_date).days
#     n_trading_days = len(trading_days)
#     arr = (
#         (1 + total_return) ** (365 / max(n_calendar_days, 1)) - 1
#         if total_return > -1
#         else -1.0
#     )

#     # Sharpe ratio (annualised, assuming risk-free = 0)
#     if len(daily_returns) > 1 and statistics.stdev(daily_returns) > 0:
#         sharpe = (
#             statistics.mean(daily_returns) / statistics.stdev(daily_returns)
#         ) * math.sqrt(252)
#     else:
#         sharpe = 0.0

#     # Maximum drawdown
#     peak = portfolio_values[0]
#     max_dd = 0.0
#     for pv in portfolio_values:
#         if pv > peak:
#             peak = pv
#         dd = (peak - pv) / peak if peak > 0 else 0.0
#         if dd > max_dd:
#             max_dd = dd

#     # ── Buy-and-hold benchmark ────────────────────────────────────────────
#     bh_start_price = float(price_df.iloc[0]["Close"])
#     bh_end_price = float(price_df.iloc[-1]["Close"])
#     bh_shares = int(initial_capital / bh_start_price)
#     bh_remaining_cash = initial_capital - bh_shares * bh_start_price
#     bh_final_value = bh_shares * bh_end_price + bh_remaining_cash
#     bh_total_return = (bh_final_value - initial_capital) / initial_capital
#     bh_arr = (
#         (1 + bh_total_return) ** (365 / max(n_calendar_days, 1)) - 1
#         if bh_total_return > -1
#         else -1.0
#     )

#     # Buy-and-hold daily returns for Sharpe and MDD
#     bh_daily_returns = price_df["Close"].pct_change().dropna().tolist()
#     if len(bh_daily_returns) > 1 and statistics.stdev(bh_daily_returns) > 0:
#         bh_sharpe = (
#             statistics.mean(bh_daily_returns) / statistics.stdev(bh_daily_returns)
#         ) * math.sqrt(252)
#     else:
#         bh_sharpe = 0.0

#     bh_peak = bh_start_price
#     bh_max_dd = 0.0
#     for price in price_df["Close"]:
#         p = float(price)
#         if p > bh_peak:
#             bh_peak = p
#         dd = (bh_peak - p) / bh_peak if bh_peak > 0 else 0.0
#         if dd > bh_max_dd:
#             bh_max_dd = dd

#     # ── Print results ──────────────────────────────────────────────────────
#     print(f"\n{'=' * 70}")
#     print(f"  BACKTEST RESULTS  |  {ticker}  |  {start_date} → {end_date}")
#     print(f"{'=' * 70}")
#     print(f"  {'Metric':<25} {'Strategy':>12} {'Buy&Hold':>12}")
#     print(f"  {'-' * 49}")
#     print(
#         f"  {'Initial Capital':<25} ${initial_capital:>11,.2f} ${initial_capital:>11,.2f}"
#     )
#     print(f"  {'Final Value':<25} ${final_value:>11,.2f} ${bh_final_value:>11,.2f}")
#     print(
#         f"  {'Total Return (CR%)':<25} {total_return * 100:>11.2f}% {bh_total_return * 100:>11.2f}%"
#     )
#     print(f"  {'Annualised Return':<25} {arr * 100:>11.2f}% {bh_arr * 100:>11.2f}%")
#     print(f"  {'Sharpe Ratio':<25} {sharpe:>12.4f} {bh_sharpe:>12.4f}")
#     print(
#         f"  {'Max Drawdown (MDD%)':<25} {max_dd * 100:>11.2f}% {bh_max_dd * 100:>11.2f}%"
#     )
#     print(f"  {'Trading Days':<25} {n_trading_days:>12d} {n_trading_days:>12d}")
#     print(f"{'=' * 70}")
#     print(
#         f"  Outperformance vs B&H:  CR% {(total_return - bh_total_return) * 100:>+.2f}%  "
#         f"Sharpe {sharpe - bh_sharpe:>+.4f}  MDD {(max_dd - bh_max_dd) * 100:>+.2f}%"
#     )
#     print(f"{'=' * 70}\n")

#     # ── Daily log CSV ───────────────────────────────────────────────────────
#     log_df = pd.DataFrame(backtest_log)
#     log_path = (
#         pathlib.Path(_project_root)
#         / "results"
#         / ticker
#         / f"backtest_{start_date}_{end_date}.csv"
#     )
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     log_df.to_csv(log_path, index=False)
#     print(f"  Daily log saved to: {log_path}")
