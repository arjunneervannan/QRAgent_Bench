#!/usr/bin/env python3
"""
Debug-optimized training loop for QRAgent_Bench.
Deterministic agent with comprehensive error testing and clear debugging output.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from envs.factor_env import FactorImproveEnv
from factors.validate import validate_program, validate_action


class DebugAgent:
    """Deterministic agent for comprehensive testing and debugging."""
    
    def __init__(self):
        self.step_count = 0
        # Predefined test scenarios covering all action types and error cases
        self.test_scenarios = [
            # Test 1: Valid OBSERVE actions
            {"type": "OBSERVE", "tool": "describe_data"},
            {"type": "OBSERVE", "tool": "plot_returns"},
            
            # Test 2: Valid FACTOR_IMPROVE with good program
            {"type": "FACTOR_IMPROVE", "new_program": self._get_valid_momentum_program()},
            
            # Test 3: Invalid OBSERVE (wrong tool)
            {"type": "OBSERVE", "tool": "invalid_tool"},
            
            # Test 4: Invalid OBSERVE (missing tool)
            {"type": "OBSERVE"},
            
            # Test 5: Invalid FACTOR_IMPROVE (circular dependency)
            {"type": "FACTOR_IMPROVE", "new_program": self._get_circular_dag_program()},
            
            # Test 6: Invalid FACTOR_IMPROVE (missing required params)
            {"type": "FACTOR_IMPROVE", "new_program": self._get_invalid_params_program()},
            
            # Test 7: Invalid FACTOR_IMPROVE (invalid operation)
            {"type": "FACTOR_IMPROVE", "new_program": self._get_invalid_op_program()},
            
            # Test 8: Invalid action type
            {"type": "INVALID_ACTION"},
            
            # Test 9: REFLECT action
            {"type": "REFLECT", "note": "Debug reflection"},
            
            # Test 10: Another valid FACTOR_IMPROVE
            {"type": "FACTOR_IMPROVE", "new_program": self._get_ema_program()},
            
            # Test 11: STOP action
            {"type": "STOP"}
        ]
    
    def get_action(self, obs):
        """Get next action from predefined test scenarios."""
        if self.step_count >= len(self.test_scenarios):
            return {"type": "STOP"}
        
        action = self.test_scenarios[self.step_count]
        self.step_count += 1
        return action
    
    def _get_valid_momentum_program(self):
        """Valid momentum factor program."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x3"}
            ],
            "output": "score"
        }
    
    def _get_circular_dag_program(self):
        """Program with circular dependency for testing error handling."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "sub", "a": "x0", "b": "x2"},  # References x2
                {"id": "x2", "op": "sub", "a": "x1", "b": "x0"}   # References x1 (circular!)
            ],
            "output": "x2"
        }
    
    def _get_invalid_params_program(self):
        """Program with missing required parameters."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return"},  # Missing 'n' parameter
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        }
    
    def _get_invalid_op_program(self):
        """Program with invalid operation."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "score", "op": "invalid_operation", "src": "x0"}
            ],
            "output": "score"
        }
    
    def _get_ema_program(self):
        """Another valid program using EMA."""
        return {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 252},
                {"id": "x1", "op": "ema", "n": 21, "src": "x0"},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "score", "op": "zscore_xs", "src": "x2"}
            ],
            "output": "score"
        }


def print_step_header(step, action):
    """Print clear step header with action details."""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {action['type']}")
    print(f"{'='*60}")
    
    # Print action details
    print(f"Action: {json.dumps(action, indent=2)}")
    
    # Validate action before execution
    is_valid, errors = validate_action(action)
    if is_valid:
        print("✅ Action validation: PASSED")
    else:
        print("❌ Action validation: FAILED")
        for error in errors:
            print(f"   - {error}")


def print_observation(obs, reward, done):
    """Print observation details with clear formatting."""
    print(f"\n📊 OBSERVATION:")
    print(f"   Budget left: {obs.get('budget_left', 'N/A')}")
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    
    # Print specific observation results
    if 'observation_result' in obs:
        result_type = type(obs['observation_result']).__name__
        print(f"   Observation result: {result_type}")
    
    if 'investment_performance' in obs:
        perf = obs['investment_performance']
        print(f"   📈 Investment Performance:")
        for key, value in perf.items():
            if key != 'plot_path' and isinstance(value, (int, float)):
                print(f"      {key}: {value:.4f}")
        if 'plot_path' in perf:
            print(f"      plot_path: {perf['plot_path']}")
    
    if 'validation_errors' in obs:
        print(f"   ❌ Validation Errors:")
        for error in obs['validation_errors']:
            print(f"      - {error}")


def test_factor_program_validation():
    """Test factor program validation separately for clear debugging."""
    print(f"\n{'='*60}")
    print("TESTING FACTOR PROGRAM VALIDATION")
    print(f"{'='*60}")
    
    test_programs = [
        ("Valid Momentum Program", DebugAgent()._get_valid_momentum_program()),
        ("Circular DAG Program", DebugAgent()._get_circular_dag_program()),
        ("Missing Parameters", DebugAgent()._get_invalid_params_program()),
        ("Invalid Operation", DebugAgent()._get_invalid_op_program()),
    ]
    
    for name, program in test_programs:
        print(f"\n--- {name} ---")
        print(f"Program: {json.dumps(program, indent=2)}")
        
        is_valid, errors = validate_program(program)
        if is_valid:
            print("✅ Validation: PASSED")
        else:
            print("❌ Validation: FAILED")
            for error in errors:
                print(f"   - {error}")


def main():
    """Main debug training loop."""
    print("🔧 QRAgent_Bench - Debug Training Loop")
    print("="*60)
    
    # Test factor program validation first
    test_factor_program_validation()
    
    # Initialize environment
    print(f"\n{'='*60}")
    print("INITIALIZING ENVIRONMENT")
    print(f"{'='*60}")
    
    try:
        env = FactorImproveEnv(
            data_path="data/ff25_value_weighted.csv",
            test_train_split=0.8,
            timesteps=15  # Small number for quick testing
        )
        print("✅ Environment initialized successfully")
        print(f"   Data shape: {env.returns.shape}")
        print(f"   Split point: {env.split}")
        print(f"   Budget: {env.timesteps}")
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return
    
    # Initialize debug agent
    agent = DebugAgent()
    print(f"✅ Debug agent initialized with {len(agent.test_scenarios)} test scenarios")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"✅ Environment reset - Initial budget: {obs['budget_left']}")
    
    # Run debug episode
    print(f"\n{'='*60}")
    print("RUNNING DEBUG EPISODE")
    print(f"{'='*60}")
    
    total_reward = 0.0
    step = 0
    
    while True:
        step += 1
        
        # Get action from agent
        action = agent.get_action(obs)
        
        # Print step information
        print_step_header(step, action)
        
        # Execute action
        try:
            obs, reward, done = env.step(action)
            total_reward += reward
            
            print_observation(obs, reward, done)
            
            if done:
                print(f"\n🏁 EPISODE COMPLETED")
                print(f"   Total reward: {total_reward:.3f}")
                print(f"   Total steps: {step}")
                break
                
        except Exception as e:
            print(f"❌ STEP EXECUTION FAILED: {e}")
            print(f"   Action that failed: {action}")
            break
    
    print(f"\n{'='*60}")
    print("DEBUG SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Total steps: {step}")


if __name__ == "__main__":
    main()