from __future__ import annotations
import pandas as pd
import numpy as np

def rolling_return(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return (1.0 + df).rolling(n, min_periods=n).apply(lambda x: x.prod() - 1.0, raw=True)

def ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()

def zscore_xs(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sig = df.std(axis=1) + eps
    return (df.sub(mu, axis=0)).div(sig, axis=0)

def demean_xs(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)

def winsor_quantile(df: pd.DataFrame, q: float = 0.02) -> pd.DataFrame:
    lo = df.quantile(q, axis=1)
    hi = df.quantile(1 - q, axis=1)
    return df.clip(lo, hi, axis=0)

def describe_data(returns: pd.DataFrame) -> dict:
    """Generate basic descriptive statistics for the dataset."""
    return {
        "shape": returns.shape,
        "date_range": {
            "start": str(returns.index.min()),
            "end": str(returns.index.max()),
            "total_days": len(returns)
        },
        "columns": list(returns.columns),
        "missing_values": returns.isnull().sum().to_dict(),
        "basic_stats": {
            "mean": returns.mean().mean(),
            "std": returns.std().mean(),
            "min": returns.min().min(),
            "max": returns.max().max()
        },
        "correlation_matrix": returns.corr().to_dict()
    }

def plot_returns(returns: pd.DataFrame, title: str = "Portfolio Returns") -> str:
    """Generate a plot of portfolio returns and save as image."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    plt.subplot(2, 2, 1)
    cumulative_returns.plot(legend=False, alpha=0.7)
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    
    # Plot return distribution
    plt.subplot(2, 2, 2)
    returns.stack().hist(bins=50, alpha=0.7)
    plt.title("Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    
    # Plot correlation heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(returns.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    
    # Plot volatility over time
    plt.subplot(2, 2, 4)
    rolling_vol = returns.rolling(21).std()
    rolling_vol.mean(axis=1).plot()
    plt.title("Rolling Volatility (21-day)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "temp_returns_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def analyze_factor_performance(scores: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Analyze the performance characteristics of a factor."""
    # Calculate factor returns (simple long-short)
    factor_returns = (scores * returns).sum(axis=1)
    
    return {
        "factor_stats": {
            "mean": float(factor_returns.mean()),
            "std": float(factor_returns.std()),
            "sharpe": float(factor_returns.mean() / factor_returns.std() * np.sqrt(252)),
            "skewness": float(factor_returns.skew()),
            "kurtosis": float(factor_returns.kurtosis())
        },
        "score_stats": {
            "mean": float(scores.mean().mean()),
            "std": float(scores.std().mean()),
            "min": float(scores.min().min()),
            "max": float(scores.max().max())
        },
        "ic_stats": {
            "mean_ic": float((scores * returns).mean().mean()),
            "ic_std": float((scores * returns).std().mean()),
            "ic_ir": float((scores * returns).mean().mean() / (scores * returns).std().mean())
        }
    }

def evaluate_program(program: dict, returns: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a small JSON-defined DAG of primitives to produce factor scores.
    Program example:
    {
      "nodes": [
        {"id":"x0","op":"rolling_return","n":126},
        {"id":"x1","op":"rolling_return","n":21},
        {"id":"x2","op":"sub","a":"x0","b":"x1"},
        {"id":"x3","op":"winsor_quantile","src":"x2","q":0.02},
        {"id":"score","op":"zscore_xs","src":"x3"}
      ],
      "output":"score"
    }
    """
    nodes = {n["id"]: n for n in program["nodes"]}
    cache: dict[str, pd.DataFrame] = {}

    def get(x):
        if isinstance(x, pd.DataFrame):
            return x
        if x in cache:
            return cache[x]
        n = nodes[x]
        op = n["op"]
        if op == "rolling_return":
            out = rolling_return(returns, int(n["n"]))
        elif op == "ema":
            src = get(n.get("src")) if "src" in n else returns
            out = ema(src, int(n["n"]))
        elif op == "sub":
            out = get(n["a"]) - get(n["b"])
        elif op == "add":
            out = get(n["a"]) + get(n["b"])
        elif op == "mul":
            out = get(n["a"]) * get(n["b"])
        elif op == "winsor_quantile":
            out = winsor_quantile(get(n["src"]), float(n.get("q", 0.02)))
        elif op == "clip":
            out = get(n["src"]).clip(float(n["lo"]), float(n["hi"]))
        elif op == "zscore_xs":
            out = zscore_xs(get(n["src"]))
        elif op == "demean_xs":
            out = demean_xs(get(n["src"]))
        elif op == "delay":
            out = get(n["src"]).shift(int(n["d"]))
        elif op == "combine":
            parts = [get(i) for i in n["inputs"]]
            w = n.get("weights", [1.0 / len(parts)] * len(parts))
            out = sum(p * w_i for p, w_i in zip(parts, w))
        else:
            raise ValueError(f"Unknown op: {op}")
        cache[x] = out
        return out

    scores = get(program["output"])
    scores = scores.reindex_like(returns).fillna(0.0)
    return scores
