import json
import numpy as np
from pathlib import Path

def load_reward_config(config_path: str = None) -> dict:
    """Load reward configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "default_reward_config.json"
    
    with open(config_path, 'r') as f:
        return json.load(f)

def calculate_reward(action_type: str, config: dict, **kwargs) -> float:
    """Calculate reward based on action type, config, and performance data."""
    if action_type == "FACTOR_IMPROVE":
        improvement = kwargs["current_sharpe"] - kwargs["equal_weight_sharpe"]
        return improvement * config["factor_improve"]["base_reward_multiplier"]
    
    elif action_type == "OBSERVE":
        return (config["observe"]["success_reward"] if kwargs.get("success", True) 
                else config["observe"]["failure_reward"])
    
    elif action_type == "REFLECT":
        return config["reflect"]["reward"]
    
    elif action_type == "STOP":
        cfg = config["stop"]
        base = cfg["base_multiplier"] * kwargs["oos_sharpe"]
        costs = cfg["cost_per_turnover"] * kwargs["turnover"] + cfg["cost_per_step"] * kwargs["steps_used"]
        guard = cfg["pass_guard"] if kwargs["tests_pass"] else cfg["fail_guard"]
        pen = -1.0 if kwargs.get("leak", False) else 0.0
        return float(np.tanh(base - costs) + guard + pen)
    
    elif action_type == "VALIDATION_ERROR":
        return config["validation_error"]["reward"]
    
    return 0.0
