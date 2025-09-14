# QRAgent_Bench - Enhanced Factor Improvement Agent

A comprehensive quantitative finance research framework for AI agents to iteratively improve cross-sectional factor strategies through observation, analysis, and strategic enhancement.

## 🚀 Features

### Enhanced Action Types
- **OBSERVE**: Analyze dataset using various tools (describe_data, plot_returns, analyze_factor_performance)
- **FACTOR_IMPROVE**: Add new factors to existing models with weighted combination
- **EVALUATE**: Run out-of-sample backtests for final performance assessment

### Advanced Analytics
- **Dataset Analysis**: Comprehensive statistical analysis and visualization
- **Factor Performance**: Detailed factor analysis with IC statistics
- **In-Sample Backtesting**: Real-time performance evaluation with plot generation
- **Reward Tracking**: Incremental reward system for step-by-step improvements

### Robust Infrastructure
- **JSON-based Factor DSL**: Safe, validated factor definition language
- **Cross-Sectional Backtesting**: Market-neutral long/short strategies with turnover constraints
- **Data Leakage Prevention**: Built-in safeguards and execution delays
- **Comprehensive Validation**: Action and program validation with detailed error reporting

## 📁 Project Structure

```
QRAgent_Bench/
├── agent/                    # Agent interface and prompts
│   └── prompt.py            # Action schemas and prompt building
├── data/                    # Data directory (FF25 data goes here)
│   └── ff25_daily.csv      # Fama-French 25 portfolios (download required)
├── engine/                  # Backtesting and data processing
│   ├── backtester.py       # Cross-sectional backtesting engine
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── data_analysis.py    # Data analysis and visualization tools
│   └── metrics.py          # Performance metrics calculation
├── envs/                    # Environment implementations
│   └── factor_env.py       # Enhanced factor improvement environment
├── factors/                 # Factor definition and validation
│   ├── program.py          # Factor DSL implementation (core operations only)
│   ├── validate.py         # Action and program validation
│   ├── baseline_program.json    # Baseline factor definition
│   └── candidate_program.json   # Current candidate factor
├── training/                # Training and evaluation scripts
│   ├── run_episodes.py     # Simple episode runner
│   ├── grpo_trainer.py     # GRPO training implementation
│   └── build_prefs_for_dpo.py  # DPO preference data builder
├── tests/                   # Test suites
│   └── test_leakage.py     # Data leakage tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd QRAgent_Bench
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**:
   - Download Fama-French 25 Portfolios daily returns
   - Save as `data/ff25_daily.csv`
   - Expected format: Date column + 25 portfolio return columns

## 🎯 Quick Start

### Basic Usage
```python
from envs.factor_env import FactorImproveEnv

# Initialize environment
env = FactorImproveEnv()
obs, _ = env.reset()

# Observe the dataset
action = {"type": "OBSERVE", "tool": "describe_data"}
obs, reward, done, _, info = env.step(action)

# Improve the factor
new_factor = {
    "nodes": [
        {"id": "x0", "op": "rolling_return", "n": 63},
        {"id": "score", "op": "zscore_xs", "src": "x0"}
    ],
    "output": "score"
}
action = {"type": "FACTOR_IMPROVE", "new_factor": new_factor, "weight": 0.3}
obs, reward, done, _, info = env.step(action)

# Evaluate out-of-sample
action = {"type": "EVALUATE"}
obs, reward, done, _, info = env.step(action)
```

### Run Training Episodes
```bash
python training/run_episodes.py
```

### Run Tests
```bash
pytest tests/
```

## 🔧 Factor DSL

The project uses a JSON-based Domain Specific Language for defining factors:

```json
{
  "nodes": [
    {"id": "x0", "op": "rolling_return", "n": 126},
    {"id": "x1", "op": "rolling_return", "n": 21},
    {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
    {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
    {"id": "score", "op": "zscore_xs", "src": "x3"}
  ],
  "output": "score"
}
```

### Available Operations
- **Time Series**: `rolling_return`, `ema`, `delay`
- **Cross-Sectional**: `zscore_xs`, `demean_xs`, `winsor_quantile`
- **Mathematical**: `add`, `sub`, `mul`, `clip`
- **Combination**: `combine` (weighted combination of multiple factors)

## 📊 Data Analysis Tools

The `engine/data_analysis.py` module provides comprehensive data analysis capabilities:

### Core Functions
- **`describe_data()`** - Basic dataset statistics and information
- **`plot_returns()`** - Portfolio return visualizations
- **`analyze_factor_performance()`** - Factor performance analysis with IC statistics
- **`plot_factor_analysis()`** - Comprehensive factor analysis plots
- **`get_data_summary()`** - Detailed dataset summary with portfolio statistics

### Analysis Features
- **Statistical Analysis**: Mean, std, skewness, kurtosis, correlations
- **Performance Metrics**: Sharpe ratios, Information Coefficient (IC), IC Information Ratio
- **Visualization**: Cumulative returns, distributions, correlation heatmaps, rolling metrics
- **Portfolio Analysis**: Individual portfolio statistics and cross-portfolio correlations

## 📊 Action Types

### OBSERVE Actions
- `describe_data`: Get dataset statistics and information
- `plot_returns`: Generate portfolio return visualizations
- `analyze_factor_performance`: Analyze factor performance characteristics

### FACTOR_IMPROVE Actions
- Propose complete new factor programs (DAGs)
- Automatic in-sample backtesting
- Performance improvement tracking

### Automatic Evaluation
- **STOP action** triggers automatic out-of-sample evaluation
- No manual EVALUATE action needed
- Final performance metrics calculated automatically

## 🎮 Environment Features

### Reward System
- **Base Reward**: OOS Sharpe ratio performance
- **Incremental Rewards**: Step-by-step improvement bonuses
- **Penalties**: Turnover costs, step count, data leakage
- **Guardrails**: Validation and safety checks

### State Management
- **Budget System**: Limited action steps per episode
- **Performance Tracking**: Baseline vs current performance
- **Reward Logging**: Complete episode and incremental reward history
- **Program Evolution**: Factor model development over time

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_enhanced_functionality.py
```

Or run specific tests:
```bash
python simple_test.py
pytest tests/test_leakage.py
```

## 📈 Performance Metrics

The backtesting engine provides comprehensive performance analysis:
- **Sharpe Ratio**: Risk-adjusted returns (gross and net)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Turnover**: Portfolio turnover and transaction costs
- **Information Coefficient**: Factor predictive power
- **Leakage Detection**: Data leakage prevention

## 🔬 Research Applications

- **Quantitative Research**: Factor strategy development and testing
- **AI/ML Research**: Reinforcement learning in financial domains
- **Educational**: Learning quantitative finance and backtesting
- **Production**: Systematic trading strategy development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Fama-French data from Kenneth French's data library
- Built on top of Gymnasium for RL environments
- Uses pandas, numpy, and matplotlib for data analysis