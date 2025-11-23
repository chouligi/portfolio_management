# Portfolio & Angel Investing Lab

Experiments and utilities for managing my personal investments:
- Public markets portfolio analysis (ETFs, cash, scenario planning)
- Angel investing support tools, including **AngelCopilot** evaluation

---

## Contents

### 1. Public Markets / Portfolio Management

This part of the repo contains small utilities and notebooks I use for:

- Building and stress-testing ETF / stock allocations
- Running simple scenario analysis (e.g. contributions over time, drawdowns)
- Checking how different savings / investing strategies impact long-term wealth

Typical things you’ll find here:

- Python scripts and/or notebooks for:
  - Loading historical price data
  - Computing returns, volatility, and drawdowns
  - Running simple Monte Carlo or scenario simulations
- Helper functions for visualizing:
  - Allocation breakdowns
  - Portfolio growth over time
  - Risk/return trade-offs

> Adjust this section to match whatever you actually keep in the repo  
> (e.g. name specific notebooks or modules if you’d like).

---

### 2. AngelCopilot Evaluation

This section contains code and data to **evaluate the AngelCopilot rubric** I use for
angel investing decisions.

The goal is to sanity-check that the rubric:

1. **Separates deal quality**  
   Higher-quality / harder-to-access deals (Tier A, e.g. syndicate / curated)
   should receive higher overall scores than easier-to-access deals (Tier B).

2. **Is reasonably robust**  
   Re-scoring the same company multiple times should give similar results
   (low variance compared to the signal between tiers).

