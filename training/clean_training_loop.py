#!/usr/bin/env python3
"""
Clean, simple training loop for debugging.
"""

import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from envs.factor_env import FactorImproveEnv
from factors.validate import validate_program, validate_action


class SimpleAgent:
    """Simple deterministic agent for testing."""
    
    def __init__(self):
        self.step = 0
        self.actions = [
            # Test valid OBSERVE
            {"type": "OBSERVE", "tool": "describe_data"},
            
            # Test valid FACTOR_IMPROVE
            {"type": "FACTOR_IMPROVE", "new_program": {
                "nodes": [
                    {"id": "x0", "op": "rolling_return", "n": 126},
                    {"id": "x1", "op": "rolling_return", "n": 21},
                    {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                    {"id": "score", "op": "zscore_xs", "src": "x2"}
                ],
                "output": "score"
            }},
            
            # Test invalid action
            {"type": "INVALID_ACTION"},
            
            # Test bad factor program
            {"type": "FACTOR_IMPROVE", "new_program": {
                "nodes": [
                    {"id": "x0", "op": "rolling_return", "n": 126},
                    {"id": "x1", "op": "sub", "a": "x0", "b": "x2"},  # Circular dependency
                    {"id": "x2", "op": "sub", "a": "x1", "b": "x0"}
                ],
                "output": "x2"
            }},
            
            # Test another valid FACTOR_IMPROVE
            {"type": "FACTOR_IMPROVE", "new_program": {
                "nodes": [
                    {"id": "x0", "op": "rolling_return", "n": 252},
                    {"id": "x1", "op": "ema", "n": 21, "src": "x0"},
                    {"id": "score", "op": "zscore_xs", "src": "x1"}
                ],
                "output": "score"
            }},
            
            # Test STOP
            {"type": "STOP"}
        ]
    
    def get_action(self, obs):
        if self.step >= len(self.actions):
            return {"type": "STOP"}
        
        action = self.actions[self.step]
        self.step += 1
        return action


def print_step(step, action, obs, reward, done):
    """Print step information clearly."""
    print(f"\n--- STEP {step} ---")
    print(f"Action: {action['type']}")
    
    if action['type'] == 'FACTOR_IMPROVE':
        print(f"Program nodes: {len(action['new_program']['nodes'])}")
    
    print(f"Reward: {reward:.3f}")
    print(f"Done: {done}")
    print(f"Budget left: {obs.get('budget_left', 'N/A')}")
    
    if 'validation_errors' in obs:
        print(f"❌ Errors: {obs['validation_errors']}")
    
    if 'investment_performance' in obs:
        perf = obs['investment_performance']
        print(f"📈 Performance:")
        for key, value in perf.items():
            if key != 'plot_path' and isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")


def main():
    print("🚀 Clean Training Loop")
    print("=" * 40)
    
    # Create environment
    try:
        env = FactorImproveEnv(
            data_path="data/ff25_value_weighted.csv",
            test_train_split=0.8,
            timesteps=10
        )
        print("✅ Environment ready")
    except Exception as e:
        print(f"❌ Environment failed: {e}")
        return
    
    # Create agent
    agent = SimpleAgent()
    print(f"✅ Agent ready with {len(agent.actions)} test actions")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"✅ Environment reset - Budget: {obs['budget_left']}")
    
    # Run episode
    total_reward = 0.0
    step = 0
    
    while True:
        step += 1
        
        # Get action
        action = agent.get_action(obs)
        
        # Execute action
        try:
            obs, reward, done = env.step(action)
            total_reward += reward
            
            # Print step info
            print_step(step, action, obs, reward, done)
            
            if done:
                break
                
        except Exception as e:
            print(f"\n❌ STEP {step} FAILED: {e}")
            print(f"Action: {action}")
            break
    
    print(f"\n🏁 EPISODE COMPLETE")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Total steps: {step}")


if __name__ == "__main__":
    main()
