from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Generate rebalance dates based on frequency."""
    return index.to_series().resample(freq).last().index

def cross_sectional_ls(
    returns: pd.DataFrame,
    scores: pd.DataFrame,
    rebalance: str = "ME",
    top_q: float = 0.2,
    tc_bps: float = 10.0,          # cost per 100% turnover
    turnover_cap: float = 1.5,     # L1 cap per rebalance
    delay_days: int = 1,
) -> dict:
    """
    Vectorized cross-sectional long/short with monthly rebalance and 1-day execution delay.
    - Market-neutral (equal-weight long top quantile, short bottom quantile)
    - Turnover cap applied per rebalance step
    - Linear transaction costs applied from daily weight changes

    Returns dict with weights, strategy gross returns, strategy net returns, and t-cost vector.
    """
    returns = returns.dropna(how="all").copy()
    scores = scores.reindex_like(returns).copy()
    returns, scores = returns.dropna().copy(), scores.dropna().copy()

    rebal_idx = _rebalance_dates(returns.index)
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    prev_w = pd.Series(0.0, index=returns.columns)

    # Build weights piecewise
    for t0, t1 in zip(rebal_idx[:-1], rebal_idx[1:]):
        # last available score up to t0
        st = scores.loc[:t0]
        if st.empty: 
            continue
        s = st.iloc[-1].copy()
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cross-sectional ranking on the day
        ranks = s.rank(pct=True)
        long_mask  = (ranks >= 1.0 - top_q)
        short_mask = (ranks <= top_q)
        
        # Check if we have valid positions
        if long_mask.sum() == 0 or short_mask.sum() == 0:
            # If no valid positions due to identical scores, use equal weight
            if s.std() == 0 or s.nunique() == 1:
                # All scores are identical - use equal weight strategy
                w = pd.Series(1.0 / len(s), index=s.index)
            else:
                # No positions if not enough names for other reasons
                w = prev_w * 0.0
        else:
            w_long  = long_mask.astype(float)
            w_short = short_mask.astype(float)
            w_long  = w_long / w_long.sum()
            w_short = w_short / w_short.sum()
            w = w_long - w_short  # market-neutral

        # Turnover cap
        raw_turn = (w - prev_w).abs().sum()
        if raw_turn > turnover_cap and raw_turn > 0:
            w = prev_w + (w - prev_w) * (turnover_cap / raw_turn)

        # Hold weights until next rebalance
        # Ensure w is properly aligned with weights columns
        w_aligned = w.reindex(weights.columns, fill_value=0.0)
        # Use values to avoid broadcasting issues
        weights.loc[t0:t1, :] = w_aligned.values
        prev_w = w_aligned

    # Execution delay to avoid look-ahead
    weights = weights.shift(delay_days).fillna(0.0)

    # Strategy returns
    strat_gross = (weights * returns).sum(axis=1)
    daily_turn = weights.diff().abs().sum(axis=1).fillna(0.0)
    tc = (tc_bps / 1e4) * daily_turn
    strat_net = strat_gross - tc

    out = {
        "weights": weights,
        "strategy_gross_returns": strat_gross,
        "strategy_net_returns": strat_net,
        "t_cost_vector": tc,
    }
    return out


def equal_weight_baseline(returns: pd.DataFrame, rebalance: str = "ME") -> dict:
    """
    Calculate equal-weight baseline weights with monthly rebalancing and intra-month drift.
    
    Args:
        returns: DataFrame of returns
        rebalance: Rebalancing frequency ("ME" for month-end, "MS" for month-start, etc.)
    
    Returns:
        Dictionary with weights and strategy returns
    """
    n_portfolios = returns.shape[1]
    initial_weight = 1.0 / n_portfolios
    
    rebal_dates = _rebalance_dates(returns.index)
    
    # Initialize weights DataFrame
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    # Vectorized approach: process each rebalancing period
    for i, rebal_date in enumerate(rebal_dates):
        # Find the next rebalancing date (or end of data)
        next_rebal_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        
        # Get the period data
        period_mask = (returns.index >= rebal_date) & (returns.index <= next_rebal_date)
        period_returns = returns.loc[period_mask]
        
        if period_returns.empty:
            continue
            
        # Start with equal weights at rebalancing date
        period_weights = pd.DataFrame(initial_weight, index=period_returns.index, columns=period_returns.columns)
        
        # Vectorized drift calculation within the period
        if len(period_returns) > 1:
            # Calculate cumulative returns for drift
            cum_returns = (1 + period_returns).cumprod()
            
            # Apply drift to weights (vectorized)
            for j in range(1, len(period_returns)):
                # Previous day's weights adjusted by returns
                prev_weights = period_weights.iloc[j-1]
                new_weights = prev_weights * (1 + period_returns.iloc[j-1])
                # Renormalize to maintain equal weight
                period_weights.iloc[j] = new_weights / new_weights.sum()
        
        # Assign to main weights DataFrame
        weights.loc[period_mask] = period_weights
    
    # Calculate strategy returns
    strategy_returns = (weights * returns).sum(axis=1)
    
    return {
        "weights": weights,
        "strategy_returns": strategy_returns,
    }

