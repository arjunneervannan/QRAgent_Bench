from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import gymnasium as gym
from engine.data_loader import load_ff25_daily
from engine.backtester import cross_sectional_ls, run_in_sample_backtest
from factors.program import evaluate_program, describe_data, plot_returns, analyze_factor_performance
from factors.validate import validate_action, validate_program

class FactorImproveEnv(gym.Env):
    """Enhanced environment for factor improvement with OBSERVE and FACTOR_IMPROVE actions."""
    metadata = {"render_modes": []}

    def __init__(self, data_path: str = "data/ff25_daily.csv", is_frac: float = 0.8):
        super().__init__()
        self.data_path = data_path
        self.returns = load_ff25_daily(self.data_path)
        self.split = int(is_frac * len(self.returns))
        self.params = {"top_q": 0.2, "turnover_cap": 1.5, "delay_days": 1}
        self.budget = 4
        self.steps_used = 0
        self.last_eval = {"oos_sharpe": 0.0, "turnover": 0.0, "tests_pass": True, "leak": False}
        
        # Load baseline factor
        self.baseline_program = json.loads(Path("factors/baseline_program.json").read_text())
        self.current_program = self.baseline_program.copy()
        
        # Reward tracking
        self.episode_rewards = []
        self.incremental_rewards = []
        self.baseline_performance = None
        self.current_performance = None
        
        # Observation tools available
        self.observation_tools = {
            "describe_data": self._describe_data,
            "plot_returns": self._plot_returns,
            "analyze_factor_performance": self._analyze_factor_performance
        }

    def _obs(self):
        obs = {
            "budget_left": self.budget,
            "last_eval": self.last_eval,
            "params": self.params,
            "current_program": self.current_program,
            "baseline_performance": self.baseline_performance,
            "current_performance": self.current_performance,
            "episode_rewards": self.episode_rewards,
            "incremental_rewards": self.incremental_rewards
        }
        
        # Include validation errors if they exist
        if "validation_errors" in self.last_eval:
            obs["validation_errors"] = self.last_eval["validation_errors"]
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.budget = 4
        self.steps_used = 0
        self.params = {"top_q": 0.2, "turnover_cap": 1.5, "delay_days": 1}
        self.last_eval = {"oos_sharpe": 0.0, "turnover": 0.0, "tests_pass": True, "leak": False}
        self.current_program = self.baseline_program.copy()
        
        # Reset reward tracking
        self.episode_rewards = []
        self.incremental_rewards = []
        self.baseline_performance = None
        self.current_performance = None
        
        return self._obs(), {}

    def _describe_data(self, **kwargs):
        """Execute describe_data tool."""
        return describe_data(self.returns)

    def _plot_returns(self, **kwargs):
        """Execute plot_returns tool."""
        return plot_returns(self.returns, "Portfolio Returns Analysis")

    def _analyze_factor_performance(self, factor_program, **kwargs):
        """Execute analyze_factor_performance tool."""
        scores = evaluate_program(factor_program, self.returns)
        return analyze_factor_performance(scores, self.returns)

    def _combine_factors(self, base_program, new_factor, weight):
        """Combine the base program with a new factor using the specified weight."""
        # Create a new program that combines the base with the new factor
        combined_nodes = base_program["nodes"].copy()
        new_nodes = new_factor["nodes"].copy()
        
        # Add the new factor nodes with modified IDs to avoid conflicts
        for node in new_nodes:
            node["id"] = f"new_{node['id']}"
            # Update references in the new factor
            if "src" in node:
                node["src"] = f"new_{node['src']}"
            if "a" in node:
                node["a"] = f"new_{node['a']}"
            if "b" in node:
                node["b"] = f"new_{node['b']}"
            if "inputs" in node:
                node["inputs"] = [f"new_{inp}" for inp in node["inputs"]]
        
        # Add a combine node
        combine_node = {
            "id": "combined_score",
            "op": "combine",
            "inputs": [base_program["output"], f"new_{new_factor['output']}"],
            "weights": [1.0 - weight, weight]
        }
        
        combined_nodes.extend(new_nodes)
        combined_nodes.append(combine_node)
        
        return {
            "nodes": combined_nodes,
            "output": "combined_score"
        }

    def _run_in_sample_backtest(self, program, generate_plot=False):
        """Run in-sample backtest on the given program."""
        scores = evaluate_program(program, self.returns)
        ret_is = self.returns.iloc[:self.split]
        sc_is = scores.iloc[:self.split]
        
        return run_in_sample_backtest(
            ret_is, sc_is, 
            generate_plot=generate_plot,
            **self.params
        )

    def _run_oos_backtest(self, program):
        """Run out-of-sample backtest on the given program."""
        scores = evaluate_program(program, self.returns)
        ret_oos = self.returns.iloc[self.split:]
        sc_oos = scores.iloc[self.split:]
        
        return cross_sectional_ls(ret_oos, sc_oos, **self.params)

    def step(self, action: dict):
        reward = 0.0
        terminated = False
        info = {}

        # Validate the action first
        is_valid_action, action_errors = validate_action(action)
        if not is_valid_action:
            # Give bad reward for invalid actions
            reward = -2.0
            self.last_eval = {
                "oos_sharpe": 0.0,
                "turnover": 0.0,
                "tests_pass": False,
                "leak": False,
                "validation_errors": action_errors
            }
            self.steps_used += 1
            self.budget -= 1
            if self.budget <= 0:
                terminated = True
            return self._obs(), reward, terminated, False, info

        atype = action.get("type")
        
        if atype == "OBSERVE":
            tool = action.get("tool")
            if tool in self.observation_tools:
                try:
                    # Execute the observation tool
                    if tool == "analyze_factor_performance":
                        result = self.observation_tools[tool](factor_program=action.get("factor_program"))
                    else:
                        result = self.observation_tools[tool]()
                    
                    self.last_eval = {
                        "oos_sharpe": 0.0,
                        "turnover": 0.0,
                        "tests_pass": True,
                        "leak": False,
                        "observation_result": result,
                        "observation_tool": tool
                    }
                    
                    # Small positive reward for successful observation
                    reward = 0.1
                    
                except Exception as e:
                    self.last_eval = {
                        "oos_sharpe": 0.0,
                        "turnover": 0.0,
                        "tests_pass": False,
                        "leak": False,
                        "observation_error": str(e),
                        "observation_tool": tool
                    }
                    reward = -0.5
            else:
                self.last_eval = {
                    "oos_sharpe": 0.0,
                    "turnover": 0.0,
                    "tests_pass": False,
                    "leak": False,
                    "validation_errors": [f"Unknown observation tool: {tool}"]
                }
                reward = -1.0

        elif atype == "FACTOR_IMPROVE":
            try:
                new_factor = action.get("new_factor")
                weight = action.get("weight", 0.5)
                
                # Combine the current program with the new factor
                combined_program = self._combine_factors(self.current_program, new_factor, weight)
                
                # Run in-sample backtest
                is_results = self._run_in_sample_backtest(combined_program, generate_plot=True)
                
                # Update current program and performance
                self.current_program = combined_program
                self.current_performance = is_results
                
                # Calculate incremental reward
                if self.baseline_performance is None:
                    # First improvement - set baseline
                    self.baseline_performance = self._run_in_sample_backtest(self.baseline_program)
                
                baseline_sharpe = self.baseline_performance["sharpe_net"]
                current_sharpe = is_results["sharpe_net"]
                improvement = current_sharpe - baseline_sharpe
                
                # Reward based on improvement
                incremental_reward = improvement * 2.0  # Scale factor
                self.incremental_rewards.append(incremental_reward)
                
                self.last_eval = {
                    "oos_sharpe": float(is_results["sharpe_net"]),
                    "turnover": float(is_results["avg_turnover"]),
                    "tests_pass": not is_results["leakage_flag"],
                    "leak": bool(is_results["leakage_flag"]),
                    "in_sample_results": {
                        "sharpe_gross": float(is_results["sharpe_gross"]),
                        "sharpe_net": float(is_results["sharpe_net"]),
                        "sortino_net": float(is_results["sortino_net"]),
                        "max_dd": float(is_results["max_dd"]),
                        "avg_turnover": float(is_results["avg_turnover"]),
                        "plot_path": is_results.get("plot_path")
                    },
                    "improvement": float(improvement),
                    "incremental_reward": float(incremental_reward)
                }
                
                reward = incremental_reward
                
            except Exception as e:
                self.last_eval = {
                    "oos_sharpe": 0.0,
                    "turnover": 0.0,
                    "tests_pass": False,
                    "leak": False,
                    "factor_improve_error": str(e)
                }
                reward = -1.0

        elif atype == "SET_PARAMS":
            # Params already validated by validate_action
            self.params.update(action.get("params", {}))
            self.last_eval = {
                "oos_sharpe": 0.0,
                "turnover": 0.0,
                "tests_pass": True,
                "leak": False,
                "params_updated": True
            }
            reward = 0.0

        elif atype == "SET_PROGRAM":
            # Program already validated by validate_action, but double-check
            is_valid_program, program_errors = validate_program(action["program"])
            if not is_valid_program:
                reward = -2.0
                self.last_eval = {
                    "oos_sharpe": 0.0,
                    "turnover": 0.0,
                    "tests_pass": False,
                    "leak": False,
                    "validation_errors": program_errors
                }
            else:
                self.current_program = action["program"]
                Path("factors/candidate_program.json").write_text(json.dumps(action["program"], indent=2))
                self.last_eval = {
                    "oos_sharpe": 0.0,
                    "turnover": 0.0,
                    "tests_pass": True,
                    "leak": False,
                    "program_updated": True
                }
                reward = 0.0

        elif atype == "EVALUATE":
            # Run OOS backtest
            oos_results = self._run_oos_backtest(self.current_program)
            
            d_sharpe = oos_results["sharpe_net"]
            turnover = oos_results["avg_turnover"]
            leak = oos_results["leakage_flag"]
            tests_pass = not leak

            self.last_eval = {
                "oos_sharpe": float(d_sharpe),
                "turnover": float(turnover),
                "tests_pass": bool(tests_pass),
                "leak": bool(leak),
                "oos_results": {
                    "sharpe_gross": float(oos_results["sharpe_gross"]),
                    "sharpe_net": float(oos_results["sharpe_net"]),
                    "sortino_net": float(oos_results["sortino_net"]),
                    "max_dd": float(oos_results["max_dd"]),
                    "avg_turnover": float(oos_results["avg_turnover"])
                }
            }

            # Final reward calculation
            base = 0.7 * d_sharpe
            costs = 0.06 * turnover + 0.01 * self.steps_used
            guard = 0.5 if tests_pass else -1.0
            pen = -1.0 if leak else 0.0
            reward = float(np.tanh(base - costs) + guard + pen)
            
            # Add incremental rewards
            if self.incremental_rewards:
                reward += sum(self.incremental_rewards) * 0.1  # Scale down incremental rewards

        elif atype == "REFLECT":
            self.last_eval = {
                "oos_sharpe": 0.0,
                "turnover": 0.0,
                "tests_pass": True,
                "leak": False,
                "reflection": action.get("note", "")
            }
            reward = 0.0

        elif atype == "STOP":
            terminated = True
            reward = 0.0

        # Track episode rewards
        self.episode_rewards.append(reward)
        
        self.steps_used += 1
        self.budget -= 1
        if self.budget <= 0:
            terminated = True

        return self._obs(), reward, terminated, False, info