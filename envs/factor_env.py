from __future__ import annotations
import json
import pandas as pd
import numpy as np
from pathlib import Path
import gymnasium as gym
from engine.data_loader import load_ff25_daily
from engine.backtester import cross_sectional_ls, equal_weight_baseline, plot_strategy_results
from engine.metrics import information_ratio, sharpe, sortino, max_drawdown, pure_sharpe
from factors.program import evaluate_program
from engine.data_analysis import describe_data, plot_returns, analyze_factor_performance
from factors.validate import validate_action, validate_program
from .reward_calculator import load_reward_config, calculate_reward

class FactorImproveEnv(gym.Env):
    """Enhanced environment for factor improvement with OBSERVE and FACTOR_IMPROVE actions."""
    metadata = {"render_modes": []}

    def __init__(self, data_path, test_train_split, timesteps, reward_config_path=None, plot_path=None):
        super().__init__()
        
        # Get the project root directory (where this file is located)
        project_root = Path(__file__).parent.parent
        
        # Use absolute paths
        self.data_path = str(project_root / data_path)
        self.returns = load_ff25_daily(self.data_path)
        self.split = int(test_train_split * len(self.returns))

        self.params = {
            "top_q": 0.2,
            "turnover_cap": 1.5,
            "delay_days": 1
        }
        
        # Load reward configuration
        self.reward_config = load_reward_config(reward_config_path)
        
        # Set plot path (default to current directory if not provided)
        self.plot_path = plot_path or "plots"
        
        # Create plot directory if it doesn't exist
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)

        self.timesteps = timesteps
        self.budget = timesteps
        self.steps_used = 0

        self.last_eval = {
            "information_ratio": 0.0,
            "strategy_pure_sharpe": 0.0,
            "benchmark_pure_sharpe": 0.0,
            "strategy_sortino": 0.0,
            "max_drawdown": 0.0,
            "tests_pass": True,
            "leak": False
        }
        
        # Initialize current program (will be set by first FACTOR_IMPROVE action)
        self.current_program = None
        
        # Reward tracking
        self.episode_rewards = []
        self.incremental_rewards = []
        self.current_performance = None
        
        # Equal-weight baseline performance (calculated once)
        self.equal_weight_baseline = None
        
        # Track last improvement from factor_improve actions
        self.last_improvement = 0.0
        
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
            "current_performance": self.current_performance,
            "episode_rewards": self.episode_rewards,
            "incremental_rewards": self.incremental_rewards,
            "equal_weight_baseline": self.equal_weight_baseline,
            "last_improvement": self.last_improvement
        }
        
        # Include validation errors if they exist
        if "validation_errors" in self.last_eval:
            obs["validation_errors"] = self.last_eval["validation_errors"]
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.budget = self.timesteps
        self.steps_used = 0
        self.params = {"top_q": 0.2, "turnover_cap": 1.5, "delay_days": 1}
        self.last_eval = {
            "information_ratio": 0.0,
            "strategy_pure_sharpe": 0.0,
            "benchmark_pure_sharpe": 0.0,
            "strategy_sortino": 0.0,
            "max_drawdown": 0.0,
            "tests_pass": True,
            "leak": False
        }
        self.current_program = None
        
        # Reset reward tracking
        self.episode_rewards = []
        self.incremental_rewards = []
        self.current_performance = None
        self.equal_weight_baseline = None
        self.last_improvement = 0.0
        
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
        project_root = Path(__file__).parent.parent
        candidate_path = project_root / "factors" / "candidate_program.json"
        candidate_path.write_text(json.dumps(program, indent=2))
        
        return True

    def _run_in_sample_backtest(self, program, generate_plot=False, plot_path=None):
        """Run in-sample backtest on the given program with random 10-year sampling."""
        scores = evaluate_program(program, self.returns)
        ret_is = self.returns.iloc[:self.split]
        sc_is = scores.iloc[:self.split]
        
        # Select a random 10-year period from the in-sample data
        ret_is, sc_is = self._sample_10_year_period(ret_is, sc_is)
        
        # Run factor-based backtest
        backtest_results = cross_sectional_ls(
            returns=ret_is,
            scores=sc_is,
            **self.params
        )
        
        # Calculate equal weight baseline weights
        equal_weight_weights = equal_weight_baseline(ret_is)
        backtest_results["equal_weight_weights"] = equal_weight_weights
        
        # Calculate equal weight returns for information ratio
        equal_weight_returns = (equal_weight_weights * ret_is).sum(axis=1)
        
        # Calculate information ratio
        strategy_net = backtest_results["series_net"]
        info_ratio = information_ratio(strategy_net, equal_weight_returns, "daily")
        backtest_results["information_ratio"] = info_ratio
        
        # Add plot path if requested
        if generate_plot:
            # Create title with time period information
            start_date = ret_is.index.min().strftime('%Y-%m-%d')
            end_date = ret_is.index.max().strftime('%Y-%m-%d')
            title = f"Strategy Results ({start_date} to {end_date})"
            
            # Use custom path if provided, otherwise generate default
            if plot_path is None:
                plot_path = f"strategy_results_{start_date}_{end_date}.png"
            
            plot_path = plot_strategy_results(
                strategy_weights=backtest_results["weights"],
                strategy_net_returns=backtest_results["series_net"],
                strategy_gross_returns=backtest_results["series_gross"],
                equal_weight_weights=equal_weight_weights,
                returns=ret_is,
                title=title,
                plot_path=plot_path
            )
            backtest_results["plot_path"] = plot_path
        
        return backtest_results
    
    def _sample_10_year_period(self, returns, scores):
        """Sample a random 10-year period from the data."""
        # Calculate 10 years in trading days (approximately 252 days per year)
        ten_years_days = 252 * 10
        
        # If we don't have enough data, return what we have
        if len(returns) <= ten_years_days:
            return returns, scores
        
        # Calculate the maximum start index to ensure we can get 10 years
        max_start_idx = len(returns) - ten_years_days
        
        # Select a random start index
        start_idx = np.random.randint(0, max_start_idx + 1)
        end_idx = start_idx + ten_years_days
        
        # Return the sampled data
        return returns.iloc[start_idx:end_idx], scores.iloc[start_idx:end_idx]

    def _run_oos_backtest(self, program):
        """Run out-of-sample backtest on the given program."""
        scores = evaluate_program(program, self.returns)
        ret_oos = self.returns.iloc[self.split:]
        sc_oos = scores.iloc[self.split:]
        
        # Calculate equal-weight baseline weights for out-of-sample period
        equal_weight_weights = equal_weight_baseline(ret_oos)
        
        # Run factor-based backtest
        factor_results = cross_sectional_ls(ret_oos, sc_oos, **self.params)
        
        # Add equal-weight weights to results
        factor_results["equal_weight_weights"] = equal_weight_weights
        
        # Calculate equal weight returns for information ratio
        equal_weight_returns = (equal_weight_weights * ret_oos).sum(axis=1)
        
        # Calculate information ratio
        strategy_net = factor_results["series_net"]
        info_ratio = information_ratio(strategy_net, equal_weight_returns, "daily")
        factor_results["information_ratio"] = info_ratio
        
        return factor_results

    def step(self, action: dict):
        reward = 0.0
        terminated = False
        info = {}

        # Validate the action first
        is_valid_action, action_errors = validate_action(action)
        if not is_valid_action:
            reward = calculate_reward("VALIDATION_ERROR", self.reward_config)
            self.last_eval = {
                "information_ratio": 0.0,
                "strategy_pure_sharpe": 0.0,
                "benchmark_pure_sharpe": 0.0,
                "strategy_sortino": 0.0,
                "max_drawdown": 0.0,
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
                # Execute the observation tool
                if tool == "analyze_factor_performance":
                    result = self.observation_tools[tool](factor_program=action.get("factor_program"))
                else:
                    result = self.observation_tools[tool]()
                
                reward = calculate_reward("OBSERVE", self.reward_config, success=True)
            else:
                reward = calculate_reward("OBSERVE", self.reward_config, success=False)

        elif atype == "FACTOR_IMPROVE":
            new_program = action.get("new_program")
            
            try:
                # Validate and set the new program
                self._validate_and_set_program(new_program)
                
                # Run in-sample backtest with custom plot path
                plot_path = f"{self.plot_path}/factor_improve_backtest_{self.steps_used}.png"
                is_results = self._run_in_sample_backtest(new_program, generate_plot=True, plot_path=plot_path)
                
                # Update current performance
                self.current_performance = is_results
                
                # Set equal-weight baseline if not already set
                if self.equal_weight_baseline is None:
                    self.equal_weight_baseline = is_results["equal_weight_weights"]
                
                # Calculate equal weight returns and metrics
                equal_weight_weights = is_results["equal_weight_weights"]
                equal_weight_returns = (equal_weight_weights * self.returns.iloc[:self.split]).sum(axis=1)
                
                # Calculate pure Sharpe ratios
                strategy_pure_sharpe = pure_sharpe(is_results["series_net"], "daily")
                equal_weight_pure_sharpe = pure_sharpe(equal_weight_returns, "daily")
                
                # Calculate improvement vs equal weight baseline
                current_sharpe = is_results["sharpe_net"]
                equal_weight_sharpe = sharpe(equal_weight_returns, "daily")
                improvement = current_sharpe - equal_weight_sharpe
                
                # Update last improvement
                self.last_improvement = improvement
                
                # Calculate reward
                incremental_reward = calculate_reward("FACTOR_IMPROVE", self.reward_config,
                                                    current_sharpe=current_sharpe,
                                                    equal_weight_sharpe=equal_weight_sharpe)
                self.incremental_rewards.append(incremental_reward)
                
                # Update last_eval with clean metrics for agent
                self.last_eval = {
                    # Core performance metrics
                    "information_ratio": float(is_results["information_ratio"]),
                    "strategy_pure_sharpe": float(strategy_pure_sharpe),
                    "benchmark_pure_sharpe": float(equal_weight_pure_sharpe),
                    "strategy_sortino": float(is_results["sortino_net"]),
                    "max_drawdown": float(is_results["max_dd"]),
                    
                    # Validation flags
                    "tests_pass": not is_results["leakage_flag"],
                    "leak": bool(is_results["leakage_flag"]),
                    
                    # Additional context
                    "improvement": float(improvement),
                    "plot_path": is_results.get("plot_path"),
                    "program_updated": True
                }
                
                reward = incremental_reward
                
            except ValueError as e:
                # Program validation error
                error_msg = str(e)
                reward = calculate_reward("VALIDATION_ERROR", self.reward_config)
                
                self.last_eval = {
                    "information_ratio": 0.0,
                    "strategy_pure_sharpe": 0.0,
                    "benchmark_pure_sharpe": 0.0,
                    "strategy_sortino": 0.0,
                    "max_drawdown": 0.0,
                    "tests_pass": False,
                    "leak": False,
                    "validation_errors": [error_msg],
                    "program_updated": False
                }
                
            except Exception as e:
                # Backtesting or other runtime error
                error_msg = f"Backtest error: {str(e)}"
                reward = calculate_reward("VALIDATION_ERROR", self.reward_config)
                
                self.last_eval = {
                    "information_ratio": 0.0,
                    "strategy_pure_sharpe": 0.0,
                    "benchmark_pure_sharpe": 0.0,
                    "strategy_sortino": 0.0,
                    "max_drawdown": 0.0,
                    "tests_pass": False,
                    "leak": False,
                    "runtime_errors": [error_msg],
                    "program_updated": False
                }

        elif atype == "REFLECT":
            reward = calculate_reward("REFLECT", self.reward_config)

        elif atype == "STOP":
            # Automatically run OOS evaluation when agent chooses to stop
            oos_results = self._run_oos_backtest(self.current_program)
            
            d_sharpe = oos_results["sharpe_net"]
            turnover = oos_results["avg_turnover"]
            leak = oos_results["leakage_flag"]
            tests_pass = not leak

            # Calculate OOS pure Sharpe ratios
            oos_strategy_net = oos_results["series_net"]
            oos_equal_weight_weights = oos_results["equal_weight_weights"]
            oos_equal_weight_returns = (oos_equal_weight_weights * self.returns.iloc[self.split:]).sum(axis=1)
            
            strategy_pure_sharpe = pure_sharpe(oos_strategy_net, "daily")
            equal_weight_pure_sharpe = pure_sharpe(oos_equal_weight_returns, "daily")
            
            self.last_eval = {
                # Core performance metrics
                "information_ratio": float(oos_results["information_ratio"]),
                "strategy_pure_sharpe": float(strategy_pure_sharpe),
                "benchmark_pure_sharpe": float(equal_weight_pure_sharpe),
                "strategy_sortino": float(oos_results["sortino_net"]),
                "max_drawdown": float(oos_results["max_dd"]),
                
                # Validation flags
                "tests_pass": bool(tests_pass),
                "leak": bool(leak),
                
                # Additional context
                "final_evaluation": True
            }

            # Calculate final reward
            reward = calculate_reward("STOP", self.reward_config,
                                    oos_sharpe=d_sharpe,
                                    turnover=turnover,
                                    steps_used=self.steps_used,
                                    tests_pass=tests_pass,
                                    leak=leak)
            
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
