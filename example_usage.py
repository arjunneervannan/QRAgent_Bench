#!/usr/bin/env python3
"""
Example usage of the enhanced QRAgent_Bench system.
Demonstrates the new prompt system and full DAG factor improvement.
"""

import json
from envs.factor_env import FactorImproveEnv
from agent.prompt import PromptBuilder
from engine.data_analysis import describe_data, plot_returns, analyze_factor_performance

def demonstrate_prompt_system():
    """Demonstrate the different prompt types."""
    print("=== Prompt System Demonstration ===\n")
    
    builder = PromptBuilder()
    
    # Example observation
    obs = {
        "budget_left": 3,
        "current_program": {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        },
        "last_eval": {"oos_sharpe": 0.0, "turnover": 0.0, "tests_pass": True, "leak": False},
        "baseline_performance": None,
        "current_performance": None,
        "episode_rewards": [],
        "incremental_rewards": []
    }
    
    # 1. Basic prompt
    print("1. Basic Prompt:")
    basic_prompt = builder.build_basic_prompt("Improve OOS Sharpe >= 0.2", obs)
    print(basic_prompt[:200] + "...\n")
    
    # 2. Response prompt
    print("2. Response Prompt:")
    response = "Data analysis shows 25 portfolios with daily returns from 1926-2025. Mean return is 0.0008, std is 0.015."
    response_prompt = builder.build_response_prompt(response, obs)
    print(response_prompt[:200] + "...\n")
    
    # 3. Observation prompt
    print("3. Observation Prompt:")
    observation_result = {
        "shape": [104092, 25],
        "date_range": {"start": "1926-07-01", "end": "2025-06-30"},
        "basic_stats": {"mean": 0.0008, "std": 0.015}
    }
    obs_prompt = builder.build_observation_prompt("describe_data", observation_result, obs)
    print(obs_prompt[:200] + "...\n")

def demonstrate_full_episode():
    """Demonstrate a complete episode with the new system."""
    print("=== Full Episode Demonstration ===\n")
    
    env = FactorImproveEnv()
    builder = PromptBuilder()
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Initial state: Budget={obs['budget_left']}, Program nodes={len(obs['current_program']['nodes'])}")
    
    # Step 1: Observe data
    print("\n--- Step 1: Observing Data ---")
    action = {"type": "OBSERVE", "tool": "describe_data"}
    obs, reward, done, _, info = env.step(action)
    print(f"Action: {action}")
    print(f"Reward: {reward:.3f}, Budget: {obs['budget_left']}")
    if "observation_result" in obs["last_eval"]:
        result = obs["last_eval"]["observation_result"]
        print(f"Data shape: {result['shape']}")
    
    # Step 2: Improve factor with full DAG
    print("\n--- Step 2: Improving Factor ---")
    new_program = {
        "nodes": [
            # Original momentum signal
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "rolling_return", "n": 21},
            {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
            
            # Additional mean reversion signal
            {"id": "x3", "op": "rolling_return", "n": 5},
            {"id": "x4", "op": "ema", "n": 10, "src": "x3"},
            {"id": "x5", "op": "sub", "a": "x3", "b": "x4"},
            
            # Combine signals
            {"id": "x6", "op": "add", "a": "x2", "b": "x5"},
            
            # Final processing
            {"id": "x7", "op": "winsor_quantile", "src": "x6", "q": 0.02},
            {"id": "score", "op": "zscore_xs", "src": "x7"}
        ],
        "output": "score"
    }
    
    action = {"type": "FACTOR_IMPROVE", "new_program": new_program}
    obs, reward, done, _, info = env.step(action)
    print(f"Action: {action}")
    print(f"Reward: {reward:.3f}, Budget: {obs['budget_left']}")
    if "in_sample_results" in obs["last_eval"]:
        results = obs["last_eval"]["in_sample_results"]
        print(f"In-sample Sharpe: {results['sharpe_net']:.3f}")
        print(f"Improvement: {obs['last_eval']['improvement']:.3f}")
    
    # Step 3: Evaluate OOS
    print("\n--- Step 3: Evaluating Out-of-Sample ---")
    action = {"type": "EVALUATE"}
    obs, reward, done, _, info = env.step(action)
    print(f"Action: {action}")
    print(f"Reward: {reward:.3f}, Budget: {obs['budget_left']}")
    print(f"OOS Sharpe: {obs['last_eval']['oos_sharpe']:.3f}")
    print(f"Turnover: {obs['last_eval']['turnover']:.3f}")
    
    # Step 4: Stop
    print("\n--- Step 4: Stopping ---")
    action = {"type": "STOP"}
    obs, reward, done, _, info = env.step(action)
    print(f"Action: {action}")
    print(f"Reward: {reward:.3f}, Done: {done}")
    
    print(f"\nFinal Summary:")
    print(f"  Episode rewards: {obs['episode_rewards']}")
    print(f"  Incremental rewards: {obs['incremental_rewards']}")
    print(f"  Final OOS Sharpe: {obs['last_eval']['oos_sharpe']:.3f}")

def demonstrate_prompt_builder_usage():
    """Show how to use the PromptBuilder class effectively."""
    print("=== PromptBuilder Usage Examples ===\n")
    
    builder = PromptBuilder()
    
    # Example context
    context = {
        "budget_left": 2,
        "current_program": {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 63},
                {"id": "score", "op": "zscore_xs", "src": "x0"}
            ],
            "output": "score"
        },
        "episode_rewards": [0.1, 0.5],
        "incremental_rewards": [0.5]
    }
    
    # 1. After observation
    print("1. After observation tool:")
    obs_prompt = builder.build_observation_prompt(
        "describe_data", 
        {"shape": [1000, 25], "mean": 0.001}, 
        context
    )
    print(obs_prompt[:150] + "...\n")
    
    # 2. After factor improvement
    print("2. After factor improvement:")
    improvement_result = {
        "sharpe_net": 0.8,
        "improvement": 0.3,
        "incremental_reward": 0.6,
        "avg_turnover": 1.2,
        "tests_pass": True
    }
    imp_prompt = builder.build_improvement_prompt(improvement_result, context)
    print(imp_prompt[:150] + "...\n")
    
    # 3. After evaluation
    print("3. After evaluation:")
    eval_result = {
        "oos_sharpe": 0.75,
        "turnover": 1.1,
        "tests_pass": True,
        "leak": False
    }
    eval_prompt = builder.build_evaluation_prompt(eval_result, context)
    print(eval_prompt[:150] + "...\n")

if __name__ == "__main__":
    print("QRAgent_Bench - Enhanced Factor Improvement System")
    print("=" * 60)
    
    demonstrate_prompt_system()
    print("\n" + "=" * 60)
    
    demonstrate_prompt_builder_usage()
    print("\n" + "=" * 60)
    
    demonstrate_full_episode()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
