from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_factor_analysis(scores: pd.DataFrame, returns: pd.DataFrame, title: str = "Factor Analysis") -> str:
    """Generate comprehensive factor analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Calculate factor returns
    factor_returns = (scores * returns).sum(axis=1)
    
    # 1. Factor scores distribution
    ax1 = axes[0, 0]
    scores.stack().hist(bins=50, ax=ax1, alpha=0.7)
    ax1.set_title("Factor Scores Distribution")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Frequency")
    
    # 2. Factor returns over time
    ax2 = axes[0, 1]
    factor_returns.plot(ax=ax2, alpha=0.7)
    ax2.set_title("Factor Returns Over Time")
    ax2.set_ylabel("Daily Return")
    
    # 3. Rolling Sharpe ratio
    ax3 = axes[0, 2]
    rolling_sharpe = factor_returns.rolling(252).mean() / factor_returns.rolling(252).std() * np.sqrt(252)
    rolling_sharpe.plot(ax=ax3, color='green', alpha=0.8)
    ax3.set_title("Rolling Sharpe Ratio (252-day)")
    ax3.set_ylabel("Sharpe Ratio")
    
    # 4. Information Coefficient over time
    ax4 = axes[1, 0]
    ic_series = (scores * returns).mean(axis=1)
    ic_series.rolling(63).mean().plot(ax=ax4, alpha=0.7)
    ax4.set_title("Rolling Information Coefficient (63-day)")
    ax4.set_ylabel("IC")
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 5. Factor returns distribution
    ax5 = axes[1, 1]
    factor_returns.hist(bins=50, ax=ax5, alpha=0.7, color='blue')
    ax5.axvline(factor_returns.mean(), color='red', linestyle='--', 
                label=f'Mean: {factor_returns.mean():.4f}')
    ax5.set_title("Factor Returns Distribution")
    ax5.set_xlabel("Daily Return")
    ax5.set_ylabel("Frequency")
    ax5.legend()
    
    # 6. Performance metrics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    sharpe = factor_returns.mean() / factor_returns.std() * np.sqrt(252)
    ic_mean = (scores * returns).mean().mean()
    ic_ir = ic_mean / (scores * returns).std().mean()
    
    metrics_text = f"""
    Factor Performance Metrics:
    
    Sharpe Ratio: {sharpe:.3f}
    Mean IC: {ic_mean:.3f}
    IC IR: {ic_ir:.3f}
    Mean Return: {factor_returns.mean():.4f}
    Volatility: {factor_returns.std():.4f}
    Skewness: {factor_returns.skew():.3f}
    Kurtosis: {factor_returns.kurtosis():.3f}
    """
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"factor_analysis_{title.replace(' ', '_').lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def get_data_summary(returns: pd.DataFrame) -> dict:
    """Get a comprehensive summary of the dataset."""
    return {
        "dataset_info": {
            "name": "Fama-French 25 Portfolios",
            "frequency": "Daily",
            "shape": returns.shape,
            "date_range": {
                "start": str(returns.index.min()),
                "end": str(returns.index.max()),
                "total_days": len(returns),
                "years": len(returns) / 252
            }
        },
        "portfolio_info": {
            "num_portfolios": len(returns.columns),
            "portfolio_names": list(returns.columns),
            "missing_data": {
                "total_missing": returns.isnull().sum().sum(),
                "missing_by_portfolio": returns.isnull().sum().to_dict()
            }
        },
        "return_statistics": {
            "overall": {
                "mean": float(returns.mean().mean()),
                "std": float(returns.std().mean()),
                "min": float(returns.min().min()),
                "max": float(returns.max().max()),
                "skewness": float(returns.skew().mean()),
                "kurtosis": float(returns.kurtosis().mean())
            },
            "by_portfolio": {
                col: {
                    "mean": float(returns[col].mean()),
                    "std": float(returns[col].std()),
                    "sharpe": float(returns[col].mean() / returns[col].std() * np.sqrt(252))
                }
                for col in returns.columns
            }
        },
        "correlation_analysis": {
            "mean_correlation": float(returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()),
            "max_correlation": float(returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].max()),
            "min_correlation": float(returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].min())
        }
    }
