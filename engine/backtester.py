from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import sharpe, sortino, max_drawdown

def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    # month-end boundaries using resample
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

    Returns dict with series, weights, Sharpe/Sortino, max drawdown, avg turnover, and a leakage flag.
    """
    returns = returns.dropna(how="all").copy()
    scores = scores.reindex_like(returns).copy()
    returns, scores = returns.dropna().copy(), scores.dropna().copy()

    # Rebalance schedule
    rebal_idx = _rebalance_dates(returns.index, rebalance)
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

def plot_backtest_results(backtest_results: dict, title: str = "Backtest Results") -> str:
    """Generate comprehensive plots of backtest results and save as image."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(title, fontsize=16)
    
    # 1. Cumulative returns
    ax1 = axes[0, 0]
    cumulative_gross = (1 + backtest_results["series_gross"]).cumprod()
    cumulative_net = (1 + backtest_results["series_net"]).cumprod()
    cumulative_gross.plot(ax=ax1, label="Gross", alpha=0.8)
    cumulative_net.plot(ax=ax1, label="Net", alpha=0.8)
    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe ratio
    ax2 = axes[0, 1]
    rolling_sharpe = backtest_results["series_net"].rolling(252).mean() / backtest_results["series_net"].rolling(252).std() * np.sqrt(252)
    rolling_sharpe.plot(ax=ax2, color='green', alpha=0.8)
    ax2.set_title("Rolling Sharpe Ratio (252-day)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[0, 2]
    cumulative = (1 + backtest_results["series_net"]).cumprod()
    peaks = cumulative.cummax()
    drawdown = (cumulative / peaks) - 1
    drawdown.plot(ax=ax3, color='red', alpha=0.8)
    ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax3.set_title("Drawdown")
    ax3.set_ylabel("Drawdown")
    ax3.grid(True, alpha=0.3)
    
    # 4. Return distribution
    ax4 = axes[1, 0]
    backtest_results["series_net"].hist(bins=50, ax=ax4, alpha=0.7, color='blue')
    ax4.axvline(backtest_results["series_net"].mean(), color='red', linestyle='--', label=f'Mean: {backtest_results["series_net"].mean():.4f}')
    ax4.set_title("Return Distribution")
    ax4.set_xlabel("Daily Return")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Turnover over time
    ax5 = axes[1, 1]
    daily_turnover = backtest_results["weights"].diff().abs().sum(axis=1)
    daily_turnover.plot(ax=ax5, alpha=0.7, color='orange')
    ax5.set_title("Daily Turnover")
    ax5.set_ylabel("Turnover")
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance metrics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    metrics_text = f"""
    Performance Metrics:
    
    Gross Sharpe: {backtest_results['sharpe_gross']:.3f}
    Net Sharpe: {backtest_results['sharpe_net']:.3f}
    Sortino: {backtest_results['sortino_net']:.3f}
    Max Drawdown: {backtest_results['max_dd']:.3f}
    Avg Turnover: {backtest_results['avg_turnover']:.3f}
    Leakage Flag: {backtest_results['leakage_flag']}
    """
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"backtest_results_{title.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def run_in_sample_backtest(
    returns: pd.DataFrame,
    scores: pd.DataFrame,
    rebalance: str = "ME",
    top_q: float = 0.2,
    tc_bps: float = 10.0,
    turnover_cap: float = 1.5,
    delay_days: int = 1,
    generate_plot: bool = False
) -> dict:
    """
    Run an in-sample backtest with optional plot generation.
    Returns the same format as cross_sectional_ls but with additional plot path if requested.
    """
    # Run the standard backtest
    backtest_results = cross_sectional_ls(
        returns=returns,
        scores=scores,
        rebalance=rebalance,
        top_q=top_q,
        tc_bps=tc_bps,
        turnover_cap=turnover_cap,
        delay_days=delay_days
    )
    
    # Add plot path if requested
    if generate_plot:
        plot_path = plot_backtest_results(backtest_results, "In-Sample Backtest Results")
        backtest_results["plot_path"] = plot_path
    
    return backtest_results
