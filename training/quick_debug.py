#!/usr/bin/env python3
"""
Quick debug script - minimal and focused.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔧 Quick Debug Test")
    print("=" * 30)
    
    # Test 1: Import everything
    try:
        from envs.factor_env import FactorImproveEnv
        from factors.validate import validate_program, validate_action
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Create environment
    try:
        env = FactorImproveEnv(
            data_path="data/ff25_value_weighted.csv",
            test_train_split=0.8,
            timesteps=3
        )
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        return
    
    # Test 3: Reset environment
    try:
        obs, _ = env.reset()
        print(f"✅ Environment reset - Budget: {obs['budget_left']}")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return
    
    # Test 4: Valid action
    try:
        action = {"type": "OBSERVE", "tool": "describe_data"}
        obs, reward, done = env.step(action)
        print(f"✅ Valid action - Reward: {reward:.3f}, Done: {done}")
    except Exception as e:
        print(f"❌ Valid action failed: {e}")
    
    # Test 5: Invalid action
    try:
        action = {"type": "INVALID"}
        obs, reward, done = env.step(action)
        print(f"✅ Invalid action handled - Reward: {reward:.3f}")
    except Exception as e:
        print(f"❌ Invalid action failed: {e}")
    
    # Test 6: Factor program validation
    try:
        program = {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        }
        is_valid, errors = validate_program(program)
        print(f"✅ Program validation - Valid: {is_valid}")
        if errors:
            print(f"   Errors: {errors}")
    except Exception as e:
        print(f"❌ Program validation failed: {e}")
    
    print("\n🎉 Debug complete!")

if __name__ == "__main__":
    main()
