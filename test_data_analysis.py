#!/usr/bin/env python3
"""
Test script for the data analysis module.
"""

import pandas as pd
import numpy as np
from engine.data_analysis import describe_data, plot_returns, analyze_factor_performance, get_data_summary
from engine.data_loader import load_ff25_daily

def test_data_analysis_tools():
    """Test the data analysis tools independently."""
    print("=== Testing Data Analysis Tools ===\n")
    
    # Load data
    print("Loading data...")
    returns = load_ff25_daily()
    print(f"✓ Data loaded: {returns.shape}")
    
    # Test describe_data
    print("\n1. Testing describe_data...")
    description = describe_data(returns)
    print(f"✓ Shape: {description['shape']}")
    print(f"✓ Date range: {description['date_range']['start']} to {description['date_range']['end']}")
    print(f"✓ Columns: {len(description['columns'])}")
    print(f"✓ Mean return: {description['basic_stats']['mean']:.6f}")
    
    # Test plot_returns
    print("\n2. Testing plot_returns...")
    plot_path = plot_returns(returns, "Test Portfolio Returns")
    print(f"✓ Plot saved to: {plot_path}")
    
    # Test get_data_summary
    print("\n3. Testing get_data_summary...")
    summary = get_data_summary(returns)
    print(f"✓ Dataset: {summary['dataset_info']['name']}")
    print(f"✓ Years of data: {summary['dataset_info']['date_range']['years']:.1f}")
    print(f"✓ Mean correlation: {summary['correlation_analysis']['mean_correlation']:.3f}")
    
    # Test analyze_factor_performance
    print("\n4. Testing analyze_factor_performance...")
    # Create a simple factor
    factor_scores = returns.rolling(21).mean()
    # Cross-sectional z-score normalization
    factor_scores = factor_scores.sub(factor_scores.mean(axis=1), axis=0).div(factor_scores.std(axis=1), axis=0)
    factor_scores = factor_scores.fillna(0)
    
    performance = analyze_factor_performance(factor_scores, returns)
    print(f"✓ Factor Sharpe: {performance['factor_stats']['sharpe']:.3f}")
    print(f"✓ Mean IC: {performance['ic_stats']['mean_ic']:.3f}")
    print(f"✓ IC IR: {performance['ic_stats']['ic_ir']:.3f}")
    
    print("\n" + "=" * 40)
    print("✓ All data analysis tests passed!")

if __name__ == "__main__":
    test_data_analysis_tools()
