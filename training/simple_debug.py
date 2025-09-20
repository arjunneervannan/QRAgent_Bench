#!/usr/bin/env python3
"""
Simple debug script for QRAgent_Bench.
Tests all components with clear error reporting.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.factor_env import FactorImproveEnv
from factors.validate import validate_program, validate_action


def test_validation():
    """Test factor program validation."""
    print("🔍 TESTING VALIDATION")
    print("=" * 40)
    
    # Valid program
    valid_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "rolling_return", "n": 21},
            {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
            {"id": "score", "op": "zscore_xs", "src": "x2"}
        ],
        "output": "score"
    }
    
    # Invalid program (circular dependency)
    invalid_program = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "sub", "a": "x0", "b": "x2"},
            {"id": "x2", "op": "sub", "a": "x1", "b": "x0"}
        ],
        "output": "x2"
    }
    
    print("✅ Valid program:")
    is_valid, errors = validate_program(valid_program)
    print(f"   Result: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"   - {error}")
    
    print("\n❌ Invalid program (circular dependency):")
    is_valid, errors = validate_program(invalid_program)
    print(f"   Result: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"   - {error}")


def test_actions():
    """Test different action types."""
    print("\n🎯 TESTING ACTIONS")
    print("=" * 40)
    
    actions = [
        {"type": "OBSERVE", "tool": "describe_data"},
        {"type": "OBSERVE", "tool": "invalid_tool"},
        {"type": "FACTOR_IMPROVE", "new_program": {"nodes": [], "output": "x"}},
        {"type": "INVALID_ACTION"},
    ]
    
    for i, action in enumerate(actions, 1):
        print(f"\nAction {i}: {action['type']}")
        is_valid, errors = validate_action(action)
        print(f"   Valid: {'YES' if is_valid else 'NO'}")
        if errors:
            for error in errors:
                print(f"   - {error}")


def test_environment():
    """Test environment initialization and basic functionality."""
    print("\n🌍 TESTING ENVIRONMENT")
    print("=" * 40)
    
    try:
        env = FactorImproveEnv(
            data_path="data/ff25_value_weighted.csv",
            test_train_split=0.8,
            timesteps=5
        )
        print("✅ Environment created successfully")
        print(f"   Data shape: {env.returns.shape}")
        print(f"   Budget: {env.timesteps}")
        
        # Test reset
        obs, _ = env.reset()
        print(f"✅ Environment reset - Budget: {obs['budget_left']}")
        
        # Test valid OBSERVE action
        print("\n📊 Testing OBSERVE action:")
        action = {"type": "OBSERVE", "tool": "describe_data"}
        try:
            obs, reward, done = env.step(action)
            print(f"   ✅ OBSERVE successful - Reward: {reward:.3f}, Done: {done}")
            print(f"   Budget left: {obs['budget_left']}")
        except Exception as e:
            print(f"   ❌ OBSERVE failed: {e}")
        
        # Test valid FACTOR_IMPROVE action
        print("\n🔧 Testing FACTOR_IMPROVE action:")
        valid_program = {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "score", "op": "zscore_xs", "src": "x2"}
            ],
            "output": "score"
        }
        action = {"type": "FACTOR_IMPROVE", "new_program": valid_program}
        try:
            obs, reward, done = env.step(action)
            print(f"   ✅ FACTOR_IMPROVE successful - Reward: {reward:.3f}, Done: {done}")
            print(f"   Budget left: {obs['budget_left']}")
            if 'investment_performance' in obs:
                perf = obs['investment_performance']
                print(f"   Strategy Sharpe: {perf.get('strategy_sharpe_net', 'N/A'):.3f}")
        except Exception as e:
            print(f"   ❌ FACTOR_IMPROVE failed: {e}")
        
        # Test invalid action
        print("\n❌ Testing invalid action:")
        action = {"type": "INVALID_ACTION"}
        try:
            obs, reward, done = env.step(action)
            print(f"   Invalid action handled - Reward: {reward:.3f}, Done: {done}")
            if 'validation_errors' in obs:
                print(f"   Validation errors: {obs['validation_errors']}")
        except Exception as e:
            print(f"   ❌ Invalid action failed: {e}")
        
        # Test STOP action
        print("\n🛑 Testing STOP action:")
        action = {"type": "STOP"}
        try:
            obs, reward, done = env.step(action)
            print(f"   ✅ STOP successful - Reward: {reward:.3f}, Done: {done}")
            if 'investment_performance' in obs:
                perf = obs['investment_performance']
                print(f"   Final Sharpe: {perf.get('strategy_sharpe_net', 'N/A'):.3f}")
        except Exception as e:
            print(f"   ❌ STOP failed: {e}")
            
    except Exception as e:
        print(f"❌ Environment test failed: {e}")


def main():
    """Run all debug tests."""
    print("🔧 QRAgent_Bench - Simple Debug Script")
    print("=" * 50)
    
    test_validation()
    test_actions()
    test_environment()
    
    print("\n✅ Debug tests completed!")


if __name__ == "__main__":
    main()
