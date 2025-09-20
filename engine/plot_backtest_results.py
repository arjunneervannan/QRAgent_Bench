from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    
    # Calculate basic metrics manually (no metrics import)
    strategy_sharpe_net = strategy_net_returns.mean() / strategy_net_returns.std() * np.sqrt(252) if strategy_net_returns.std() > 0 else 0
    strategy_sharpe_gross = strategy_gross_returns.mean() / strategy_gross_returns.std() * np.sqrt(252) if strategy_gross_returns.std() > 0 else 0
    equal_weight_sharpe = equal_weight_returns.mean() / equal_weight_returns.std() * np.sqrt(252) if equal_weight_returns.std() > 0 else 0
    
    # Calculate information ratio manually
    excess_returns = strategy_net_returns - equal_weight_returns
    info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    
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
            excess_window = strategy_window - equal_weight_window
            ir = excess_window.mean() / excess_window.std() * np.sqrt(252) if excess_window.std() > 0 else 0
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
             f"Strategy Sharpe (Net): {strategy_sharpe_net:.3f}\n"
             f"Strategy Sharpe (Gross): {strategy_sharpe_gross:.3f}\n"
             f"Equal Weight Sharpe: {equal_weight_sharpe:.3f}\n"
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
