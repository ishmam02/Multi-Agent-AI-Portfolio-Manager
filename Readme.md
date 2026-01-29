# Multi-Agent AI Portfolio Manager

> **Note:** This project is under active development. Documentation and features will be updated as progress is made.

## Project Overview

Portfolio management—the problem of allocating capital across multiple assets to maximize returns while managing risk—has traditionally relied on expert knowledge and handcrafted rules. Classical approaches require strong assumptions about return distributions and correlations that rarely hold in practice. These limitations make portfolio optimization an attractive domain for artificial intelligence.

This capstone project focuses on building a **multi-agent AI system** for end-of-day market analysis and personalized portfolio recommendations.

## System Architecture

The system employs multiple specialized agents that independently analyze different dimensions of the market:

| Agent                 | Responsibility                                                   |
| --------------------- | ---------------------------------------------------------------- |
| **Technical Agent**   | Analyzes technical price patterns using transformer-based models |
| **Fundamental Agent** | Evaluates company fundamentals and financial metrics             |
| **Macro Agent**       | Monitors macroeconomic indicators and trends                     |
| **Sentiment Agent**   | Processes news and social sentiment data                         |

An **aggregation layer** synthesizes these independent analyses into a unified market view. A **personalization layer** then adapts recommendations to individual user profiles based on:

- Risk tolerance
- Investment time horizon
- Financial goals

## Why This Matters

Different users with the same market view should receive different recommendations. For example, a conservative retiree and an aggressive young investor should not hold the same portfolio. Most existing research focuses on optimizing a single objective function—this project studies how to map market analysis to user-specific actions.

---

## Project Structure

```
Multi-Agent-AI-Portfolio-Manager/
├── README.md                 # This file
├── using_colab.md            # Guide for running on Google Colab
├── technical_ai/             # Technical analysis agent
│   ├── README.md             # Setup and usage instructions
│   ├── main.py               # Main training script
│   ├── run_main.sh           # Shell script for training
│   └── database/             # Data storage
└── (more agents coming soon)
```

---

## Getting Started

### Running Locally

Each agent folder contains its own `README.md` with specific setup and usage instructions:

- [Technical Agent](technical_ai/README.md) - Transformer-based price pattern analysis

### Running on Google Colab (Recommended for GPU Access)

If you don't have a local GPU, we recommend using Google Colab for training. See our detailed setup guide:

- [Google Colab Setup Guide](using_colab.md)

---

## Requirements

See individual agent folders for specific requirements.

---

## Current Progress

- [x] Technical Agent - Initial implementation
- [ ] Fundamental Agent - Coming soon
- [ ] Macro Agent - Coming soon
- [ ] Sentiment Agent - Coming soon
- [ ] Aggregation Layer - Coming soon
- [ ] Personalization Layer - Coming soon
- [ ] User Interface - Coming soon

---
