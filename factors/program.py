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
