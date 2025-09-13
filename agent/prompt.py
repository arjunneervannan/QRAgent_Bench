from __future__ import annotations
import json

ACTION_SCHEMA = """
Valid actions (emit exactly ONE JSON object per step):
- OBSERVE:     {"type":"OBSERVE","tool":"<tool_name>","<params>":<values>}
  Available tools:
  - "describe_data": {"type":"OBSERVE","tool":"describe_data"}
  - "plot_returns": {"type":"OBSERVE","tool":"plot_returns"}
  - "analyze_factor_performance": {"type":"OBSERVE","tool":"analyze_factor_performance","factor_program":{...DSL JSON...}}
- FACTOR_IMPROVE: {"type":"FACTOR_IMPROVE","new_factor":{...DSL JSON...},"weight":0.5}
"""

def build_prompt(task_card: str, last_obs: dict) -> str:
    return f"""Task: {task_card}
`
    {ACTION_SCHEMA}

    CONTEXT:
    last_eval = {json.dumps(last_obs.get('last_eval', {}))}
    params    = {json.dumps(last_obs.get('params', {}))}

    Think briefly about how to improve OOS Sharpe while keeping turnover <= 1.5 and avoiding leakage.
    Then output ONE JSON action with no extra text.
    Action JSON:
    """
