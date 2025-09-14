from __future__ import annotations
import json
from typing import Dict, Any, Optional

ACTION_SCHEMA = """
Valid actions (emit exactly ONE JSON object per step):
- OBSERVE:     {"type":"OBSERVE","tool":"<tool_name>","<params>":<values>}
  Available tools:
  - "describe_data": {"type":"OBSERVE","tool":"describe_data"}
  - "plot_returns": {"type":"OBSERVE","tool":"plot_returns"}
  - "analyze_factor_performance": {"type":"OBSERVE","tool":"analyze_factor_performance","factor_program":{...DSL JSON...}}
- FACTOR_IMPROVE: {"type":"FACTOR_IMPROVE","new_program":{...DSL JSON...}}
- EVALUATE:    {"type":"EVALUATE"}
- REFLECT:     {"type":"REFLECT","note":"<brief reasoning>"}
- STOP:        {"type":"STOP"}
"""

class PromptBuilder:
    """Builder class for different types of prompts."""
    
    def __init__(self):
        self.action_schema = ACTION_SCHEMA
    
    def build_basic_prompt(self, task_card: str, last_obs: Dict[str, Any]) -> str:
        """Build a basic prompt that gives the agent options to OBSERVE or FACTOR_IMPROVE."""
        return f"""Task: {task_card}

You are a quantitative finance agent tasked with improving factor strategies. You have access to observation tools and can propose new factor programs.

{self.action_schema}

CURRENT STATE:
- Budget remaining: {last_obs.get('budget_left', 0)}
- Current program: {json.dumps(last_obs.get('current_program', {}), indent=2)}
- Last evaluation: {json.dumps(last_obs.get('last_eval', {}), indent=2)}
- Baseline performance: {json.dumps(last_obs.get('baseline_performance', {}), indent=2)}
- Current performance: {json.dumps(last_obs.get('current_performance', {}), indent=2)}

AVAILABLE ACTIONS:
1. OBSERVE - Analyze the dataset or current factor performance
2. FACTOR_IMPROVE - Propose a new complete factor program (DAG)
3. EVALUATE - Run out-of-sample evaluation
4. REFLECT - Add reasoning notes
5. STOP - End the episode

Think about what would be most helpful given the current state. Consider:
- Do you need more information about the data or current factor?
- Can you propose a better factor program?
- Are you ready for final evaluation?

Output ONE JSON action with no extra text.
Action JSON:
"""

    def build_response_prompt(self, response: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build a prompt that takes a response and prompts the agent to reflect on it."""
        context_str = ""
        if context:
            context_str = f"\nCONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        return f"""You received the following response from the system:

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
        return f"""You just executed the {tool} tool and received the following result:

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
        return f"""You just improved the factor and received the following results:

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

    def build_evaluation_prompt(self, eval_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build a prompt after evaluation has been completed."""
        return f"""You just completed the out-of-sample evaluation with these results:

EVALUATION RESULTS:
- OOS Sharpe: {eval_result.get('oos_sharpe', 0):.3f}
- Turnover: {eval_result.get('turnover', 0):.3f}
- Tests pass: {eval_result.get('tests_pass', False)}
- Leakage detected: {eval_result.get('leak', False)}

FINAL EPISODE SUMMARY:
- Total episode rewards: {context.get('episode_rewards', [])}
- Incremental rewards: {context.get('incremental_rewards', [])}
- Final OOS Sharpe: {eval_result.get('oos_sharpe', 0):.3f}

This episode is complete. You should STOP.

{self.action_schema}

Output ONE JSON action with no extra text.
Action JSON:
"""

# Legacy function for backward compatibility
def build_prompt(task_card: str, last_obs: dict) -> str:
    """Legacy function - use PromptBuilder.build_basic_prompt instead."""
    builder = PromptBuilder()
    return builder.build_basic_prompt(task_card, last_obs)