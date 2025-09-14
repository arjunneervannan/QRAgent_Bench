#!/usr/bin/env python3
"""
Test script for the enhanced factor environment functionality.
Tests OBSERVE and FACTOR_IMPROVE actions.
"""

import json
import numpy as np
from envs.factor_env import FactorImproveEnv
from engine.data_analysis import describe_data, plot_returns, analyze_factor_performance
from factors.validate import validate_action

def test_observe_actions():
    """Test OBSERVE action functionality."""
    print("=== Testing OBSERVE Actions ===\n")
    
    env = FactorImproveEnv()
    obs, _ = env.reset()
    
    # Test 1: Describe data
    print("1. Testing describe_data tool:")
    action = {"type": "OBSERVE", "tool": "describe_data"}
    is_valid, errors = validate_action(action)
    print(f"   Action valid: {is_valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    if "observation_result" in obs["last_eval"]:
        result = obs["last_eval"]["observation_result"]
        print(f"   Data shape: {result['shape']}")
        print(f"   Date range: {result['date_range']['start']} to {result['date_range']['end']}")
        print(f"   Columns: {len(result['columns'])}")
    print()
    
    # Test 2: Plot returns
    print("2. Testing plot_returns tool:")
    action = {"type": "OBSERVE", "tool": "plot_returns"}
    is_valid, errors = validate_action(action)
    print(f"   Action valid: {is_valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    if "observation_result" in obs["last_eval"]:
        plot_path = obs["last_eval"]["observation_result"]
        print(f"   Plot saved to: {plot_path}")
    print()
    
    # Test 3: Analyze factor performance
    print("3. Testing analyze_factor_performance tool:")
    factor_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 63},
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    action = {"type": "OBSERVE", "tool": "analyze_factor_performance", "factor_program": factor_program}
    is_valid, errors = validate_action(action)
    print(f"   Action valid: {is_valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    if "observation_result" in obs["last_eval"]:
        result = obs["last_eval"]["observation_result"]
        print(f"   Factor Sharpe: {result['factor_stats']['sharpe']:.3f}")
        print(f"   Mean IC: {result['ic_stats']['mean_ic']:.3f}")
    print()

def test_factor_improve_action():
    """Test FACTOR_IMPROVE action functionality."""
    print("=== Testing FACTOR_IMPROVE Action ===\n")
    
    env = FactorImproveEnv()
    obs, _ = env.reset()
    
    # Create a new complete factor program
    new_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 63},
            {"id": "x1", "op": "rolling_return", "n": 5},
            {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
            {"id": "score", "op": "zscore_xs", "src": "x2"}
        ],
        "output": "score"
    }
    
    action = {"type": "FACTOR_IMPROVE", "new_program": new_program}
    is_valid, errors = validate_action(action)
    print(f"1. Action valid: {is_valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    
    if "in_sample_results" in obs["last_eval"]:
        results = obs["last_eval"]["in_sample_results"]
        print(f"   In-sample Sharpe: {results['sharpe_net']:.3f}")
        print(f"   Improvement: {obs['last_eval']['improvement']:.3f}")
        print(f"   Incremental reward: {obs['last_eval']['incremental_reward']:.3f}")
        if "plot_path" in results:
            print(f"   Plot saved to: {results['plot_path']}")
    
    print(f"   Current program nodes: {len(obs['current_program']['nodes'])}")
    print(f"   Episode rewards: {obs['episode_rewards']}")
    print(f"   Incremental rewards: {obs['incremental_rewards']}")
    print()

def test_full_episode():
    """Test a complete episode with the new actions."""
    print("=== Testing Full Episode ===\n")
    
    env = FactorImproveEnv()
    obs, _ = env.reset()
    
    print(f"Initial budget: {obs['budget_left']}")
    print(f"Baseline performance: {obs['baseline_performance']}")
    print()
    
    # Step 1: Observe data
    print("Step 1: Observing data...")
    action = {"type": "OBSERVE", "tool": "describe_data"}
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}, Budget left: {obs['budget_left']}")
    print()
    
    # Step 2: Improve factor
    print("Step 2: Improving factor...")
    new_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 42},
            {"id": "x1", "op": "ema", "n": 10, "src": "x0"},
            {"id": "score", "op": "zscore_xs", "src": "x1"}
        ],
        "output": "score"
    }
    action = {"type": "FACTOR_IMPROVE", "new_program": new_program}
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}, Budget left: {obs['budget_left']}")
    if "improvement" in obs["last_eval"]:
        print(f"   Improvement: {obs['last_eval']['improvement']:.3f}")
    print()
    
    # Step 3: Evaluate OOS
    print("Step 3: Evaluating out-of-sample...")
    action = {"type": "EVALUATE"}
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}, Budget left: {obs['budget_left']}")
    print(f"   OOS Sharpe: {obs['last_eval']['oos_sharpe']:.3f}")
    print(f"   Turnover: {obs['last_eval']['turnover']:.3f}")
    print(f"   Tests pass: {obs['last_eval']['tests_pass']}")
    print()
    
    # Step 4: Stop
    print("Step 4: Stopping...")
    action = {"type": "STOP"}
    obs, reward, done, _, info = env.step(action)
    print(f"   Reward: {reward:.3f}, Done: {done}")
    print()
    
    print("Final episode summary:")
    print(f"   Total episode rewards: {obs['episode_rewards']}")
    print(f"   Incremental rewards: {obs['incremental_rewards']}")
    print(f"   Final OOS Sharpe: {obs['last_eval']['oos_sharpe']:.3f}")

def test_invalid_actions():
    """Test invalid actions to ensure proper error handling."""
    print("=== Testing Invalid Actions ===\n")
    
    # Test invalid OBSERVE action
    print("1. Invalid OBSERVE action (missing tool):")
    action = {"type": "OBSERVE"}
    is_valid, errors = validate_action(action)
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    print()
    
    # Test invalid FACTOR_IMPROVE action
    print("2. Invalid FACTOR_IMPROVE action (missing new_program):")
    action = {"type": "FACTOR_IMPROVE"}
    is_valid, errors = validate_action(action)
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    print()
    
    # Test invalid tool
    print("3. Invalid observation tool:")
    action = {"type": "OBSERVE", "tool": "invalid_tool"}
    is_valid, errors = validate_action(action)
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    print()

if __name__ == "__main__":
    print("Testing Enhanced Factor Environment Functionality\n")
    print("=" * 60)
    
    test_observe_actions()
    test_factor_improve_action()
    test_full_episode()
    test_invalid_actions()
    
    print("=" * 60)
    print("All tests completed!")
