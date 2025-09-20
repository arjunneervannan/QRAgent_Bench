# QRAgent - Custom RL Environment for Factor Strategy Development

A reinforcement learning environment for AI agents to develop and improve quantitative factor strategies through iterative observation, analysis, and enhancement.

## Overview

This environment allows RL agents to:
- **Observe** market data and factor performance
- **Improve** factor strategies through programmatic modifications
- **Learn** from performance feedback and reward signals

## Key Features

- **Customizable Reward System**: JSON-configurable reward functions
- **Factor DSL**: JSON-based domain-specific language for factor definition
- **Real-time Backtesting**: In-sample performance evaluation with equal-weight baseline comparison
- **Data Analysis Tools**: Built-in tools for market data exploration
- **Leakage Prevention**: Built-in safeguards against data leakage

## Quick Start

```python
from envs.factor_env import FactorImproveEnv

# Initialize environment
env = FactorImproveEnv("data/ff25_daily.csv", test_train_split=0.8, timesteps=100, plot_path="results/plots")

# Reset and start episode
obs, info = env.reset()

# Take actions
action = {"type": "OBSERVE", "tool": "describe_data"}
obs, reward, terminated, truncated, info = env.step(action)

action = {"type": "FACTOR_IMPROVE", "new_program": factor_program}
obs, reward, terminated, truncated, info = env.step(action)

action = {"type": "STOP"}
obs, reward, terminated, truncated, info = env.step(action)
```

## Installation

```bash
pip install -r requirements.txt
```

Download Fama-French 25 Portfolios data and save as `data/ff25_daily.csv`.

## Project Structure

```
├── envs/
│   ├── factor_env.py              # Main RL environment
│   ├── reward_calculator.py       # Functional reward calculation
│   └── default_reward_config.json # Reward configuration
├── engine/
│   ├── backtester.py             # Backtesting engine
│   ├── data_loader.py            # Data loading
│   └── metrics.py                # Performance metrics
├── factors/
│   ├── program.py                # Factor DSL operations
│   └── validate.py               # Validation logic
└── training/
    └── sample_training_loop.py   # Example training script
```

## Action Types

- **OBSERVE**: Analyze data using built-in tools
- **FACTOR_IMPROVE**: Propose new factor strategies
- **REFLECT**: Internal reasoning step
- **STOP**: End episode and trigger final evaluation

## Reward Configuration

Customize rewards by editing `envs/default_reward_config.json` or passing a custom config:

```python
env = FactorImproveEnv(data_path, split, timesteps, "custom_reward_config.json")
```

## Factor DSL

Define factors using JSON:

```json
{
  "nodes": [
    {"id": "x0", "op": "rolling_return", "n": 126},
    {"id": "x1", "op": "zscore_xs", "src": "x0"}
  ],
  "output": "x1"
}
```

Available operations: `rolling_return`, `ema`, `zscore_xs`, `demean_xs`, `winsor_quantile`, `add`, `sub`, `mul`, `clip`, `delay`, `combine`.

## License

MIT License