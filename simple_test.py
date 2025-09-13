#!/usr/bin/env python3
"""
Simple test for the enhanced factor environment.
"""

import json
from envs.factor_env import FactorImproveEnv

def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing Enhanced Factor Environment")
    print("=" * 40)
    
    # Initialize environment
    env = FactorImproveEnv()
    obs, _ = env.reset()
    
    print(f"✓ Environment initialized")
    print(f"✓ Budget: {obs['budget_left']}")
    print(f"✓ Data shape: {env.returns.shape}")
    print(f"✓ Baseline program loaded: {len(obs['current_program']['nodes'])} nodes")
    
    # Test OBSERVE action
    print("\nTesting OBSERVE action...")
    action = {"type": "OBSERVE", "tool": "describe_data"}
    obs, reward, done, _, info = env.step(action)
    
    print(f"✓ OBSERVE action executed")
    print(f"✓ Reward: {reward:.3f}")
    print(f"✓ Budget remaining: {obs['budget_left']}")
    print(f"✓ Done: {done}")
    
    if "observation_result" in obs["last_eval"]:
        result = obs["last_eval"]["observation_result"]
        print(f"✓ Data analysis completed: {result['shape'][0]} rows, {result['shape'][1]} columns")
    
    # Test FACTOR_IMPROVE action
    print("\nTesting FACTOR_IMPROVE action...")
    new_factor = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 63},
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    
    action = {"type": "FACTOR_IMPROVE", "new_factor": new_factor, "weight": 0.3}
    obs, reward, done, _, info = env.step(action)
    
    print(f"✓ FACTOR_IMPROVE action executed")
    print(f"✓ Reward: {reward:.3f}")
    print(f"✓ Budget remaining: {obs['budget_left']}")
    print(f"✓ Done: {done}")
    
    if "in_sample_results" in obs["last_eval"]:
        results = obs["last_eval"]["in_sample_results"]
        print(f"✓ In-sample backtest completed")
        print(f"✓ Sharpe ratio: {results['sharpe_net']:.3f}")
        print(f"✓ Improvement: {obs['last_eval'].get('improvement', 0):.3f}")
    
    print(f"✓ Current program has {len(obs['current_program']['nodes'])} nodes")
    print(f"✓ Episode rewards: {obs['episode_rewards']}")
    
    print("\n" + "=" * 40)
    print("✓ All basic tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
