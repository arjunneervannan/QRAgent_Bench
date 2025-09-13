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
