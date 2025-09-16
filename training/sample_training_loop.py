#!/usr/bin/env python3
"""
Simple training loop for QRAgent_Bench.
Demonstrates how to train an AI agent to improve factor strategies.
"""

import random
import json
import numpy as np
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from envs.factor_env import FactorImproveEnv
from agent.prompt import PromptBuilder


class RandomAgent:
    """Simple random agent for demonstration purposes."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.action_history = []
        
    def select_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Select a random valid action based on current observation."""
        budget_left = obs.get('budget_left', 0)
        
        if budget_left <= 0:
            return {"type": "STOP"}
        
        # Define action probabilities
        action_weights = {
            "OBSERVE": 0.3,
            "FACTOR_IMPROVE": 0.6,
            "REFLECT": 0.05,
            "STOP": 0.05
        }
        
        # Adjust weights based on budget
        if budget_left <= 1:
            action_weights["STOP"] = 0.8
            action_weights["FACTOR_IMPROVE"] = 0.2
        elif budget_left <= 2:
            action_weights["FACTOR_IMPROVE"] = 0.7
            action_weights["OBSERVE"] = 0.2
            action_weights["STOP"] = 0.1
        
        # Select action type
        action_type = self.rng.choices(
            list(action_weights.keys()),
            weights=list(action_weights.values())
        )[0]
        
        if action_type == "OBSERVE":
            tools = ["describe_data", "plot_returns", "analyze_factor_performance"]
            tool = self.rng.choice(tools)
            action = {"type": "OBSERVE", "tool": tool}
            
            # Add factor_program for analyze_factor_performance
            if tool == "analyze_factor_performance":
                action["factor_program"] = obs.get('current_program', {})
        
        elif action_type == "FACTOR_IMPROVE":
            # Generate a random factor program
            action = {
                "type": "FACTOR_IMPROVE",
                "new_program": self._generate_random_program()
            }
        
        elif action_type == "REFLECT":
            action = {
                "type": "REFLECT",
                "note": f"Random reflection at step {len(self.action_history) + 1}"
            }
        
        else:  # STOP
            action = {"type": "STOP"}
        
        self.action_history.append(action)
        return action
    
    def _generate_random_program(self) -> Dict[str, Any]:
        """Generate a random factor program."""
        # Simple momentum-based factor
        n1 = self.rng.randint(20, 252)  # Long-term momentum
        n2 = self.rng.randint(5, 63)    # Short-term momentum
        
        # Sometimes add mean reversion
        if self.rng.random() < 0.3:
            # Mean reversion factor
            program = {
                "nodes": [
                    {"id": "x0", "op": "rolling_return", "n": n2},
                    {"id": "x1", "op": "ema", "n": n1, "src": "x0"},
                    {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                    {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
                    {"id": "score", "op": "zscore_xs", "src": "x3"}
                ],
                "output": "score"
            }
        else:
            # Momentum factor
            program = {
                "nodes": [
                    {"id": "x0", "op": "rolling_return", "n": n1},
                    {"id": "x1", "op": "rolling_return", "n": n2},
                    {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                    {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
                    {"id": "score", "op": "zscore_xs", "src": "x3"}
                ],
                "output": "score"
            }
        
        return program


class TrainingLogger:
    """Simple logger for training progress."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.final_sharpes = []
        self.improvements = []
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   final_sharpe: float, improvement: float):
        """Log episode results."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.final_sharpes.append(final_sharpe)
        self.improvements.append(improvement)
        
        print(f"Episode {episode:3d}: "
              f"Reward={reward:6.3f}, "
              f"Length={length:2d}, "
              f"Final Sharpe={final_sharpe:6.3f}, "
              f"Improvement={improvement:6.3f}")
    
    def log_summary(self):
        """Log training summary."""
        if not self.episode_rewards:
            return
            
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Average Reward: {np.mean(self.episode_rewards):.3f} ± {np.std(self.episode_rewards):.3f}")
        print(f"Average Length: {np.mean(self.episode_lengths):.3f} ± {np.std(self.episode_lengths):.3f}")
        print(f"Average Final Sharpe: {np.mean(self.final_sharpes):.3f} ± {np.std(self.final_sharpes):.3f}")
        print(f"Average Improvement: {np.mean(self.improvements):.3f} ± {np.std(self.improvements):.3f}")
        print(f"Best Episode Reward: {max(self.episode_rewards):.3f}")
        print(f"Best Final Sharpe: {max(self.final_sharpes):.3f}")
        print("="*60)


def run_training_episode(env: FactorImproveEnv, agent: RandomAgent, 
                        episode: int, verbose: bool = True) -> Tuple[float, int, float, float]:
    """Run a single training episode."""
    obs, _ = env.reset()
    total_reward = 0.0
    step_count = 0
    
    if verbose:
        print(f"\n--- Episode {episode} ---")
        print(f"Initial budget: {obs['budget_left']}")
    
    while True:
        # Agent selects action
        action = agent.select_action(obs)
        
        if verbose:
            print(f"Step {step_count + 1}: {action['type']}")
        
        # Environment step
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if verbose:
            print(f"  Reward: {reward:.3f}, Budget left: {obs['budget_left']}")
            if 'improvement' in obs.get('last_eval', {}):
                print(f"  Improvement: {obs['last_eval']['improvement']:.3f}")
            if 'factor_improve_error' in obs.get('last_eval', {}):
                print(f"  ❌ FACTOR_IMPROVE ERROR: {obs['last_eval']['factor_improve_error']}")
            if 'validation_errors' in obs.get('last_eval', {}):
                print(f"  ❌ VALIDATION ERROR: {obs['last_eval']['validation_errors']}")
        
        if done:
            break
    
    # Extract final metrics
    final_sharpe = obs.get('last_eval', {}).get('oos_sharpe', 0.0)
    improvement = obs.get('last_eval', {}).get('improvement', 0.0)
    
    if verbose:
        print(f"Episode {episode} completed:")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Steps: {step_count}")
        print(f"  Final OOS Sharpe: {final_sharpe:.3f}")
        print(f"  Improvement: {improvement:.3f}")
    
    return total_reward, step_count, final_sharpe, improvement


def main():
    """Main training loop."""
    print("QRAgent_Bench - Simple Training Loop")
    print("="*50)
    print("Starting main function...")
    
    # Initialize environment
    try:
        env = FactorImproveEnv()
        print("✓ Environment initialized successfully")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure data/ff25_daily.csv exists")
        return
    
    # Initialize agent and logger
    agent = RandomAgent(seed=42)
    logger = TrainingLogger()
    
    # Training parameters
    num_episodes = 3
    verbose = True
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Agent: Random Agent (seed=42)")
    print(f"Environment: FactorImproveEnv")
    print()
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        try:
            reward, length, final_sharpe, improvement = run_training_episode(
                env, agent, episode, verbose=verbose
            )
            logger.log_episode(episode, reward, length, final_sharpe, improvement)
            
        except Exception as e:
            print(f"✗ Episode {episode} failed: {e}")
            continue
    
    # Log summary
    logger.log_summary()
    
    # Save results
    results = {
        "episode_rewards": logger.episode_rewards,
        "episode_lengths": logger.episode_lengths,
        "final_sharpes": logger.final_sharpes,
        "improvements": logger.improvements,
        "agent_type": "RandomAgent",
        "num_episodes": num_episodes
    }
    
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to training_results.json")


if __name__ == "__main__":
    main()
