import numpy as np
import pandas as pd

def ann_factor(ret_freq: str = "daily") -> float:
    return {"daily": np.sqrt(252.0), "weekly": np.sqrt(52.0), "monthly": np.sqrt(12.0)}[ret_freq]

def sharpe(returns: pd.Series, ret_freq: str = "daily", eps: float = 1e-9) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=0) + eps
    return float(mu / sd) * ann_factor(ret_freq)

def sortino(returns: pd.Series, ret_freq: str = "daily", eps: float = 1e-9) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    dd = downside.std(ddof=0) + eps
    mu = returns.mean()
    return float(mu / dd) * ann_factor(ret_freq)

def max_drawdown(cum: pd.Series) -> float:
    peaks = cum.cummax()
    dd = (cum / peaks) - 1.0
    return float(dd.min())

def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, ret_freq: str = "daily") -> float:
    """Calculate information ratio: (strategy_return - benchmark_return) / vol_of_excess_returns"""
    strategy_returns = pd.Series(strategy_returns).dropna()
    benchmark_returns = pd.Series(benchmark_returns).dropna()
    
    if strategy_returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align the series
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_index) == 0:
        return 0.0
    
    strategy_aligned = strategy_returns.loc[common_index]
    benchmark_aligned = benchmark_returns.loc[common_index]
    
    # Calculate excess returns
    excess_returns = strategy_aligned - benchmark_aligned
    
    # Calculate information ratio
    excess_mean = excess_returns.mean()
    excess_vol = excess_returns.std(ddof=0)
    
    if excess_vol == 0:
        return 0.0
    
    return float(excess_mean / excess_vol) * ann_factor(ret_freq)

def pure_sharpe(returns: pd.Series, risk_free_rate: float = 0.02, ret_freq: str = "daily") -> float:
    """Calculate pure Sharpe ratio assuming a static risk-free rate"""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / ann_factor(ret_freq) ** 2
    
    # Calculate excess returns
    excess_returns = returns - daily_rf
    
    # Calculate Sharpe ratio
    mu = excess_returns.mean()
    sd = excess_returns.std(ddof=0)
    
    if sd == 0:
        return 0.0
    
    return float(mu / sd) * ann_factor(ret_freq)
