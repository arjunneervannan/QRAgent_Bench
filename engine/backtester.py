from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import *

def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Generate rebalance dates based on frequency."""
    return index.to_series().resample(freq).last().index

def _get_rebalance_frequency(rebalance: str) -> str:
    """Convert rebalance parameter to pandas frequency string."""
    rebalance_map = {
        "ME": "ME",  # Month End
        "MS": "MS",  # Month Start
        "QE": "QE",  # Quarter End
        "QS": "QS",  # Quarter Start
        "YE": "YE",  # Year End
        "YS": "YS",  # Year Start
    }
    return rebalance_map.get(rebalance, "ME")  # Default to month end

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

    Returns dict with series, weights, Sharpe/Sortino, max drawdown, avg turnover, and a leakage flag.
    """
    returns = returns.dropna(how="all").copy()
    scores = scores.reindex_like(returns).copy()
    returns, scores = returns.dropna().copy(), scores.dropna().copy()

    # Rebalance schedule
    rebalance_freq = _get_rebalance_frequency(rebalance)
    rebal_idx = _rebalance_dates(returns.index, rebalance_freq)
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
        "series_gross": strat_gross,
        "series_net": strat_net,
        "sharpe_gross": sharpe(strat_gross, "daily"),
        "sharpe_net": sharpe(strat_net, "daily"),
        "sortino_net": sortino(strat_net, "daily"),
        "max_dd": max_drawdown((1.0 + strat_net).cumprod()),
        "avg_turnover": float(daily_turn.resample(rebalance).sum().mean() if len(daily_turn) else 0.0),
        "leakage_flag": bool(delay_days < 1),
    }
    return out

def plot_strategy_results(strategy_weights: pd.DataFrame, 
                         strategy_net_returns: pd.Series, 
                         strategy_gross_returns: pd.Series,
                         equal_weight_weights: pd.DataFrame,
                         returns: pd.DataFrame,
                         title: str = "Strategy Results", 
                         plot_path: str = None) -> str:
    """Generate comprehensive 2x2 grid plots for strategy evaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    # Calculate equal weight returns
    equal_weight_returns = (equal_weight_weights * returns).sum(axis=1)
    
    # Calculate pure Sharpe ratios
    strategy_pure_sharpe_net = pure_sharpe(strategy_net_returns, "daily")
    strategy_pure_sharpe_gross = pure_sharpe(strategy_gross_returns, "daily")
    equal_weight_pure_sharpe = pure_sharpe(equal_weight_returns, "daily")
    
    # Calculate information ratio
    info_ratio = information_ratio(strategy_net_returns, equal_weight_returns, "daily")
    
    # 1. Net performance, gross performance, and equal weight baseline
    ax1 = axes[0, 0]
    cumulative_net = (1 + strategy_net_returns).cumprod()
    cumulative_gross = (1 + strategy_gross_returns).cumprod()
    cumulative_equal_weight = (1 + equal_weight_returns).cumprod()
    
    cumulative_net.plot(ax=ax1, label="Strategy Net", alpha=0.8, linewidth=2)
    cumulative_gross.plot(ax=ax1, label="Strategy Gross", alpha=0.8, linewidth=2)
    cumulative_equal_weight.plot(ax=ax1, label="Equal Weight", alpha=0.8, linestyle='--', linewidth=2)
    
    ax1.set_title("Cumulative Performance")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 252-day rolling information ratio
    ax2 = axes[0, 1]
    rolling_window = 252
    if len(strategy_net_returns) >= rolling_window:
        rolling_ir = []
        for i in range(rolling_window, len(strategy_net_returns)):
            strategy_window = strategy_net_returns.iloc[i-rolling_window:i]
            equal_weight_window = equal_weight_returns.iloc[i-rolling_window:i]
            ir = information_ratio(strategy_window, equal_weight_window, "daily")
            rolling_ir.append(ir)
        
        rolling_ir_series = pd.Series(rolling_ir, index=strategy_net_returns.index[rolling_window:])
        rolling_ir_series.plot(ax=ax2, color='green', alpha=0.8, linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_title("252-Day Rolling Information Ratio")
    ax2.set_ylabel("Information Ratio")
    ax2.grid(True, alpha=0.3)
    
    # 3. 252-day rolling Sharpe for strategy and equal weight baseline
    ax3 = axes[1, 0]
    if len(strategy_net_returns) >= rolling_window:
        strategy_rolling_sharpe = strategy_net_returns.rolling(rolling_window).mean() / strategy_net_returns.rolling(rolling_window).std() * np.sqrt(252)
        equal_weight_rolling_sharpe = equal_weight_returns.rolling(rolling_window).mean() / equal_weight_returns.rolling(rolling_window).std() * np.sqrt(252)
        
        strategy_rolling_sharpe.plot(ax=ax3, color='blue', alpha=0.8, linewidth=2, label="Strategy")
        equal_weight_rolling_sharpe.plot(ax=ax3, color='orange', alpha=0.8, linewidth=2, linestyle='--', label="Equal Weight")
    
    ax3.set_title("252-Day Rolling Sharpe Ratio")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown graph
    ax4 = axes[1, 1]
    # Strategy drawdown
    strategy_cumulative = (1 + strategy_net_returns).cumprod()
    strategy_peaks = strategy_cumulative.cummax()
    strategy_drawdown = (strategy_cumulative / strategy_peaks) - 1
    strategy_drawdown.plot(ax=ax4, color='red', alpha=0.8, linewidth=2, label="Strategy")
    ax4.fill_between(strategy_drawdown.index, strategy_drawdown, 0, alpha=0.3, color='red')
    
    # Equal weight drawdown
    equal_weight_cumulative = (1 + equal_weight_returns).cumprod()
    equal_weight_peaks = equal_weight_cumulative.cummax()
    equal_weight_drawdown = (equal_weight_cumulative / equal_weight_peaks) - 1
    equal_weight_drawdown.plot(ax=ax4, color='orange', alpha=0.8, linewidth=2, linestyle='--', label="Equal Weight")
    ax4.fill_between(equal_weight_drawdown.index, equal_weight_drawdown, 0, alpha=0.2, color='orange')
    
    ax4.set_title("Drawdown")
    ax4.set_ylabel("Drawdown")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add metrics summary as text overlay
    fig.text(0.02, 0.02, 
             f"Strategy Pure Sharpe (Net): {strategy_pure_sharpe_net:.3f}\n"
             f"Strategy Pure Sharpe (Gross): {strategy_pure_sharpe_gross:.3f}\n"
             f"Equal Weight Pure Sharpe: {equal_weight_pure_sharpe:.3f}\n"
             f"Information Ratio: {info_ratio:.3f}",
             fontsize=10, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if plot_path is None:
        plot_path = f"strategy_results_{title.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def equal_weight_baseline(returns: pd.DataFrame, rebalance: str = "ME") -> pd.DataFrame:
    """
    Calculate equal-weight baseline weights with monthly rebalancing and intra-month drift.
    
    Args:
        returns: DataFrame of returns
        rebalance: Rebalancing frequency ("ME" for month-end, "MS" for month-start, etc.)
    
    Returns:
        DataFrame of equal-weight baseline weights
    """
    n_portfolios = returns.shape[1]
    initial_weight = 1.0 / n_portfolios
    
    # Get rebalance dates
    rebalance_freq = _get_rebalance_frequency(rebalance)
    rebal_dates = _rebalance_dates(returns.index, rebalance_freq)
    
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
    
    return weights

