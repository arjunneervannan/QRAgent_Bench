from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import gymnasium as gym
from engine.data_loader import load_ff25_daily
from engine.backtester import cross_sectional_ls, run_in_sample_backtest
from factors.program import evaluate_program
from engine.data_analysis import describe_data, plot_returns, analyze_factor_performance
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

    def _validate_and_set_program(self, program):
        """Validate and set the new program, replacing the current one entirely."""
        # Validate the program structure
        is_valid, errors = validate_program(program)
        if not is_valid:
            raise ValueError(f"Invalid program: {errors}")
        
        # Set the new program
        self.current_program = program
        Path("factors/candidate_program.json").write_text(json.dumps(program, indent=2))
        
        return True

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
                new_program = action.get("new_program")
                
                # Validate and set the new program
                self._validate_and_set_program(new_program)
                
                # Run in-sample backtest
                is_results = self._run_in_sample_backtest(new_program, generate_plot=True)
                
                # Update current performance
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
                    "incremental_reward": float(incremental_reward),
                    "program_updated": True
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
            # Automatically run OOS evaluation when agent chooses to stop
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
                    "max_dd": float(oos_resultx_dd"]),
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
            
            terminated = True

        # Track episode rewards
        self.episode_rewards.append(reward)
        
        self.steps_used += 1
        self.budget -= 1
        if self.budget <= 0:
            terminated = True

        return self._obs(), reward, terminated, False, info
