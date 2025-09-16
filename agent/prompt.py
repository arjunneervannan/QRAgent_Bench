from __future__ import annotations
import json
from typing import Dict, Any, Optional

# System prompt that provides context about the agent's role and capabilities
SYSTEM_PROMPT = """You are an expert quantitative researcher specializing in factor-based investment strategies. Your role is to analyze financial data and develop sophisticated factor models that can predict stock returns.

CORE CAPABILITIES:
- Analyze portfolio return data using statistical tools
- Design factor models using a JSON-based Domain Specific Language (DSL)
- Evaluate factor performance through backtesting
- Iteratively improve factor strategies based on performance feedback

FACTOR MODELING EXPERTISE:
- Understand momentum, mean reversion, and other factor patterns
- Design cross-sectional and time-series factor signals
- Combine multiple signals using mathematical operations
- Apply proper normalization and risk controls

PERFORMANCE OBJECTIVES:
- Maximize out-of-sample Sharpe ratio
- Control turnover and transaction costs
- Avoid data leakage and overfitting
- Ensure factor signals are economically meaningful

You have access to observation tools for data analysis and can propose complete factor programs. Your goal is to develop robust, profitable factor strategies through systematic analysis and iteration."""

ACTION_SCHEMA = """
Valid actions (emit exactly ONE JSON object per step):
- OBSERVE:     {"type":"OBSERVE","tool":"<tool_name>","<params>":<values>}
  Available tools:
  - "describe_data": {"type":"OBSERVE","tool":"describe_data"}
  - "plot_returns": {"type":"OBSERVE","tool":"plot_returns"}
  - "analyze_factor_performance": {"type":"OBSERVE","tool":"analyze_factor_performance","factor_program":{...DSL JSON...}}
- FACTOR_IMPROVE: {"type":"FACTOR_IMPROVE","new_program":{...DSL JSON...}}
- REFLECT:     {"type":"REFLECT","note":"<brief reasoning>"}
- STOP:        {"type":"STOP"}
"""

class PromptBuilder:
    """Builder class for different types of prompts."""
    
    def __init__(self):
        self.action_schema = ACTION_SCHEMA
        self.system_prompt = SYSTEM_PROMPT
    
    def build_basic_prompt(self, task_card: str, last_obs: Dict[str, Any]) -> str:
        """Build a basic prompt that gives the agent options to OBSERVE or FACTOR_IMPROVE."""
        
        return f"""{self.system_prompt}

TASK: {task_card}

{self.action_schema}

CURRENT STATE:
- Budget remaining: {last_obs.get('budget_left', 0)}
- Current program: {json.dumps(last_obs.get('current_program', {}), indent=2)}
- Last evaluation: {json.dumps(last_obs.get('last_eval', {}), indent=2)}
- Baseline performance: {json.dumps(last_obs.get('baseline_performance', {}), indent=2)}
- Current performance: {json.dumps(last_obs.get('current_performance', {}), indent=2)}

Think about what would be most helpful given the current state. Consider:
- Do you need more information about the data or current factor?
- Can you propose a better factor program?
- Are you ready to end the episode for evaluation?

Output ONE JSON action with no extra text.
Action JSON:
"""

    def build_response_prompt(self, response: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build a prompt that takes a response and prompts the agent to reflect on it."""
        context_str = ""
        if context:
            context_str = f"\nCONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        return f"""{self.system_prompt}

You received the following response from the system:

{response}
{context_str}

Please analyze this response and provide your next action. Consider:
- What does this response tell you about the current state?
- What should be your next step?
- Do you need to adjust your approach?

{self.action_schema}

Output ONE JSON action with no extra text.
Action JSON:
"""

    def build_observation_prompt(self, tool: str, result: Any, context: Dict[str, Any]) -> str:
        """Build a prompt after an observation tool has been executed."""
        return f"""{self.system_prompt}

You just executed the {tool} tool and received the following result:

{json.dumps(result, indent=2)}

CONTEXT:
- Budget remaining: {context.get('budget_left', 0)}
- Current program: {json.dumps(context.get('current_program', {}), indent=2)}
- Episode rewards: {context.get('episode_rewards', [])}

Based on this observation, what would you like to do next?

{self.action_schema}

Output ONE JSON action with no extra text.
Action JSON:
"""

    def build_improvement_prompt(self, improvement_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build a prompt after a factor improvement has been made."""
        return f"""{self.system_prompt}

You just edited the factor and received the following results:

IMPROVEMENT RESULTS:
- In-sample Sharpe: {improvement_result.get('sharpe_net', 0):.3f}
- Improvement: {improvement_result.get('improvement', 0):.3f}
- Incremental reward: {improvement_result.get('incremental_reward', 0):.3f}
- Turnover: {improvement_result.get('avg_turnover', 0):.3f}
- Tests pass: {improvement_result.get('tests_pass', False)}

CONTEXT:
- Budget remaining: {context.get('budget_left', 0)}
- Episode rewards: {context.get('episode_rewards', [])}
- Incremental rewards: {context.get('incremental_rewards', [])}

Current program: {json.dumps(context.get('current_program', {}), indent=2)}

Based on these results, what would you like to do next?

{self.action_schema}

Output ONE JSON action with no extra text.
Action JSON:
"""

    def build_final_evaluation_prompt(self, eval_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build a prompt after the final evaluation has been completed (for reflection only)."""
        return f"""{self.system_prompt}

The episode has concluded with the following final evaluation results:

FINAL EVALUATION RESULTS:
- OOS Sharpe: {eval_result.get('oos_sharpe', 0):.3f}
- Turnover: {eval_result.get('turnover', 0):.3f}
- Tests pass: {eval_result.get('tests_pass', False)}
- Leakage detected: {eval_result.get('leak', False)}

EPISODE SUMMARY:
- Total episode rewards: {context.get('episode_rewards', [])}
- Incremental rewards: {context.get('incremental_rewards', [])}
- Final OOS Sharpe: {eval_result.get('oos_sharpe', 0):.3f}

This episode is complete. The evaluation was performed automatically when you chose to STOP.
You can use REFLECT to add any final thoughts about your strategy.

{self.action_schema}

Output ONE JSON action with no extra text.
Action JSON:
"""

# Backward compatibility function
def build_prompt(task_card: str, last_obs: dict) -> str:
    """Build a basic prompt for backward compatibility."""
    builder = PromptBuilder()
    return builder.build_basic_prompt(task_card, last_obs)