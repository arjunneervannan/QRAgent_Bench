from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Tuple

def validate_dataframe(df: pd.DataFrame, operation_name: str) -> None:
    """Validate DataFrame for common issues that could cause NaN results."""
    if df.empty:
        raise ValueError(f"{operation_name}: Input DataFrame is empty")
    
    if df.isna().all().all():
        raise ValueError(f"{operation_name}: Input DataFrame contains only NaN values")
    
    # Check for infinite values
    if np.isinf(df).any().any():
        warnings.warn(f"{operation_name}: Input DataFrame contains infinite values")
    
    # Check for extremely large values that might cause overflow
    if (df.abs() > 1e6).any().any():
        warnings.warn(f"{operation_name}: Input DataFrame contains very large values (>1e6)")

def rolling_return(df: pd.DataFrame, n: int, validate: bool = True) -> pd.DataFrame:
    """
    Calculate rolling returns over n periods.
    
    Args:
        df: DataFrame of returns (should be in decimal form, e.g., 0.01 for 1%)
        n: Number of periods for rolling calculation
        validate: Whether to perform input validation
    
    Returns:
        DataFrame of rolling returns. First n-1 rows will be NaN due to min_periods=n.
        This is expected behavior, not a bug.
    """
    if validate:
        validate_dataframe(df, "rolling_return")
        
        if n <= 0:
            raise ValueError(f"rolling_return: n must be positive, got {n}")
        
        if n > len(df):
            warnings.warn(f"rolling_return: n ({n}) is greater than data length ({len(df)}). "
                         f"Result will be all NaN.")
    
    # Ensure we're working with numeric data
    if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
        raise ValueError("rolling_return: DataFrame must contain only numeric data")
    
    # Calculate rolling return: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    result = (1.0 + df).rolling(n, min_periods=n).apply(lambda x: x.prod() - 1.0, raw=True)
    
    # Validate result
    if validate:
        expected_nan_count = (n - 1) * df.shape[1]
        actual_nan_count = result.isna().sum().sum()
        
        if actual_nan_count != expected_nan_count and actual_nan_count > expected_nan_count:
            warnings.warn(f"rolling_return: Unexpected NaN count. Expected {expected_nan_count} "
                         f"(first {n-1} rows), got {actual_nan_count}")
    
    return result

def ema(df: pd.DataFrame, n: int, validate: bool = True) -> pd.DataFrame:
    """Calculate exponential moving average."""
    if validate:
        validate_dataframe(df, "ema")
        if n <= 0:
            raise ValueError(f"ema: n must be positive, got {n}")
    
    result = df.ewm(span=n, adjust=False, min_periods=n).mean()
    
    if validate and result.isna().all().all():
        warnings.warn("ema: Result contains only NaN values")
    
    return result

def zscore_xs(df: pd.DataFrame, eps: float = 1e-9, validate: bool = True) -> pd.DataFrame:
    """Calculate cross-sectional z-scores."""
    if validate:
        validate_dataframe(df, "zscore_xs")
        if eps <= 0:
            raise ValueError(f"zscore_xs: eps must be positive, got {eps}")
    
    mu = df.mean(axis=1)
    sig = df.std(axis=1) + eps
    
    # Check for zero standard deviation
    if validate and (sig == eps).any():
        warnings.warn("zscore_xs: Some rows have zero standard deviation")
    
    result = (df.sub(mu, axis=0)).div(sig, axis=0)
    
    if validate and result.isna().any().any():
        warnings.warn("zscore_xs: Result contains NaN values")
    
    return result

def demean_xs(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Demean cross-sectionally."""
    if validate:
        validate_dataframe(df, "demean_xs")
    
    result = df.sub(df.mean(axis=1), axis=0)
    
    if validate and result.isna().any().any():
        warnings.warn("demean_xs: Result contains NaN values")
    
    return result

def winsor_quantile(df: pd.DataFrame, q: float = 0.02, validate: bool = True) -> pd.DataFrame:
    """Winsorize data at given quantiles."""
    if validate:
        validate_dataframe(df, "winsor_quantile")
        if not 0 <= q <= 0.5:
            raise ValueError(f"winsor_quantile: q must be between 0 and 0.5, got {q}")
    
    lo = df.quantile(q, axis=1)
    hi = df.quantile(1 - q, axis=1)
    result = df.clip(lo, hi, axis=0)
    
    if validate and result.isna().any().any():
        warnings.warn("winsor_quantile: Result contains NaN values")
    
    return result


def evaluate_program(program: dict, returns: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
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
    if validate:
        validate_dataframe(returns, "evaluate_program")
        
        if not isinstance(program, dict):
            raise ValueError("evaluate_program: program must be a dictionary")
        
        if "nodes" not in program or "output" not in program:
            raise ValueError("evaluate_program: program must contain 'nodes' and 'output' keys")
        
        if not isinstance(program["nodes"], list) or len(program["nodes"]) == 0:
            raise ValueError("evaluate_program: program.nodes must be a non-empty list")
    
    nodes = {n["id"]: n for n in program["nodes"]}
    cache: dict[str, pd.DataFrame] = {}

    def get(x):
        if isinstance(x, pd.DataFrame):
            return x
        if x in cache:
            return cache[x]
        
        if x not in nodes:
            raise ValueError(f"evaluate_program: Node '{x}' not found in program")
        
        n = nodes[x]
        op = n["op"]
        
        try:
            if op == "rolling_return":
                out = rolling_return(returns, int(n["n"]), validate=validate)
            elif op == "ema":
                src = get(n.get("src")) if "src" in n else returns
                out = ema(src, int(n["n"]), validate=validate)
            elif op == "sub":
                a, b = get(n["a"]), get(n["b"])
                if validate and a.shape != b.shape:
                    raise ValueError(f"evaluate_program: Shape mismatch in sub operation: {a.shape} vs {b.shape}")
                out = a - b
            elif op == "add":
                a, b = get(n["a"]), get(n["b"])
                if validate and a.shape != b.shape:
                    raise ValueError(f"evaluate_program: Shape mismatch in add operation: {a.shape} vs {b.shape}")
                out = a + b
            elif op == "mul":
                a, b = get(n["a"]), get(n["b"])
                if validate and a.shape != b.shape:
                    raise ValueError(f"evaluate_program: Shape mismatch in mul operation: {a.shape} vs {b.shape}")
                out = a * b
            elif op == "winsor_quantile":
                out = winsor_quantile(get(n["src"]), float(n.get("q", 0.02)), validate=validate)
            elif op == "clip":
                src = get(n["src"])
                out = src.clip(float(n["lo"]), float(n["hi"]))
            elif op == "zscore_xs":
                out = zscore_xs(get(n["src"]), validate=validate)
            elif op == "demean_xs":
                out = demean_xs(get(n["src"]), validate=validate)
            elif op == "delay":
                out = get(n["src"]).shift(int(n["d"]))
            elif op == "combine":
                parts = [get(i) for i in n["inputs"]]
                w = n.get("weights", [1.0 / len(parts)] * len(parts))
                if validate and len(parts) != len(w):
                    raise ValueError(f"evaluate_program: Mismatch between inputs ({len(parts)}) and weights ({len(w)})")
                out = sum(p * w_i for p, w_i in zip(parts, w))
            else:
                raise ValueError(f"evaluate_program: Unknown op: {op}")
            
            # Validate result
            if validate and out.isna().all().all():
                warnings.warn(f"evaluate_program: Node '{x}' (op: {op}) produced only NaN values")
            
            cache[x] = out
            return out
            
        except Exception as e:
            raise RuntimeError(f"evaluate_program: Error in node '{x}' (op: {op}): {str(e)}") from e

    try:
        scores = get(program["output"])
        scores = scores.reindex_like(returns).fillna(0.0)
        
        if validate:
            if scores.isna().any().any():
                warnings.warn("evaluate_program: Final scores contain NaN values after fillna(0.0)")
            
            # Check for reasonable score ranges
            if (scores.abs() > 10).any().any():
                warnings.warn("evaluate_program: Final scores contain extreme values (>10)")
        
        return scores
        
    except Exception as e:
        raise RuntimeError(f"evaluate_program: Failed to evaluate program: {str(e)}") from e
