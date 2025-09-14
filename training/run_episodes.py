from __future__ import annotations
import json, random
from pathlib import Path
from envs.factor_env import FactorImproveEnv
from agent.prompt import build_prompt

# --- Heuristic policy stub (replace with an LLM) ---
def simple_policy(obs: dict) -> dict:
    # First step: observe the data
    if obs["budget_left"] == 4:
        return {"type": "OBSERVE", "tool": "describe_data"}
    # Second step: try to improve the factor
    if obs["budget_left"] == 3:
        # Create a complete new factor program
        new_program = {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "rolling_return", "n": 63},
                {"id": "x4", "op": "ema", "n": 10, "src": "x3"},
                {"id": "x5", "op": "add", "a": "x2", "b": "x4"},
                {"id": "x6", "op": "winsor_quantile", "src": "x5", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x6"}
            ],
            "output": "score"
        }
        return {"type": "FACTOR_IMPROVE", "new_program": new_program}
    # Final step: stop (triggers automatic evaluation)
    return {"type": "STOP"}

def llm_call(prompt: str) -> str:
    # Placeholder LLM entry point — swap with your model API.
    # For now, just call the heuristic policy based on the embedded context.
    # The 'prompt' contains last_obs; parse budget_left quickly:
    budget_left = 0
    for line in prompt.splitlines():
        if "budget_left" in line:
            try:
                budget_left = int(''.join(ch for ch in line if ch.isdigit()))
            except Exception:
                pass
    # We cannot reconstruct obs easily here, so just map budget_left to actions:
    if budget_left == 4:
        return json.dumps({"type": "OBSERVE", "tool": "describe_data"})
    elif budget_left == 3:
        # Create a complete new factor program
        new_program = {
            "nodes": [
                {"id": "x0", "op": "rolling_return", "n": 126},
                {"id": "x1", "op": "rolling_return", "n": 21},
                {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
                {"id": "x3", "op": "rolling_return", "n": 63},
                {"id": "x4", "op": "ema", "n": 10, "src": "x3"},
                {"id": "x5", "op": "add", "a": "x2", "b": "x4"},
                {"id": "x6", "op": "winsor_quantile", "src": "x5", "q": 0.02},
                {"id": "score", "op": "zscore_xs", "src": "x6"}
            ],
            "output": "score"
        }
        return json.dumps({"type": "FACTOR_IMPROVE", "new_program": new_program})
    else:
        return json.dumps({"type": "STOP"})

def main():
    env = FactorImproveEnv()
    total_eps = 5
    for ep in range(total_eps):
        obs, _ = env.reset()
        done = False
        R = 0.0
        while not done:
            prompt = build_prompt("Improve OOS Sharpe >= 0.2 with turnover <= 1.5", obs)
            action = json.loads(llm_call(prompt))
            obs, r, done, _, _ = env.step(action)
            R += r
        print(f"Episode {ep+1}/{total_eps} total_reward={R:.3f} last_eval={obs['last_eval']}")

if __name__ == "__main__":
    main()
