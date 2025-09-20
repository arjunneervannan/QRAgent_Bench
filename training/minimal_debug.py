#!/usr/bin/env python3
"""
Minimal debug script - just the essentials.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔧 Minimal Debug")
    
    # Test imports
    try:
        from envs.factor_env import FactorImproveEnv
        print("✅ Imports OK")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test environment
    try:
        env = FactorImproveEnv("data/ff25_value_weighted.csv", 0.8, 3)
        obs, _ = env.reset()
        print(f"✅ Env OK - Budget: {obs['budget_left']}")
    except Exception as e:
        print(f"❌ Env error: {e}")
        return
    
    # Test actions
    actions = [
        {"type": "OBSERVE", "tool": "describe_data"},
        {"type": "FACTOR_IMPROVE", "new_program": {
            "nodes": [{"id": "x0", "op": "rolling_return", "n": 126}, {"id": "score", "op": "zscore_xs", "src": "x0"}],
            "output": "score"
        }},
        {"type": "INVALID"},
        {"type": "STOP"}
    ]
    
    for i, action in enumerate(actions, 1):
        try:
            obs, reward, done = env.step(action)
            print(f"Step {i}: {action['type']} -> Reward: {reward:.2f}, Done: {done}")
            if 'validation_errors' in obs:
                print(f"  Errors: {obs['validation_errors']}")
        except Exception as e:
            print(f"Step {i}: {action['type']} -> ERROR: {e}")
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
