from __future__ import annotations
from typing import Dict, List, Any, Tuple, Set
import json
from pathlib import Path

# Valid operations and their required parameters
VALID_OPS = {
    "rolling_return": {"required": ["n"], "optional": []},
    "ema": {"required": ["n"], "optional": ["src"]},
    "zscore_xs": {"required": ["src"], "optional": []},
    "demean_xs": {"required": ["src"], "optional": []},
    "winsor_quantile": {"required": ["src"], "optional": ["q"]},
    "clip": {"required": ["src", "lo", "hi"], "optional": []},
    "delay": {"required": ["src", "d"], "optional": []},
    "add": {"required": ["a", "b"], "optional": []},
    "sub": {"required": ["a", "b"], "optional": []},
    "mul": {"required": ["a", "b"], "optional": []},
    "combine": {"required": ["inputs"], "optional": ["weights"]}
}

def validate_program_structure(program: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the basic structure of a JSON program.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required top-level keys
    if "nodes" not in program:
        errors.append("Missing 'nodes' key in program")
        return False, errors
    
    if "output" not in program:
        errors.append("Missing 'output' key in program")
        return False, errors
    
    if not isinstance(program["nodes"], list):
        errors.append("'nodes' must be a list")
        return False, errors
    
    if not program["nodes"]:
        errors.append("'nodes' list cannot be empty")
        return False, errors
    
    if not isinstance(program["output"], str):
        errors.append("'output' must be a string")
        return False, errors
    
    return True, errors

def validate_node_structure(node: Dict[str, Any], node_id: str) -> Tuple[bool, List[str]]:
    """
    Validate individual node structure.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required node keys
    if "id" not in node:
        errors.append(f"Node missing 'id' key")
        return False, errors
    
    if "op" not in node:
        errors.append(f"Node {node['id']} missing 'op' key")
        return False, errors
    
    if not isinstance(node["id"], str):
        errors.append(f"Node ID must be a string, got {type(node['id'])}")
        return False, errors
    
    if not isinstance(node["op"], str):
        errors.append(f"Node {node['id']} op must be a string, got {type(node['op'])}")
        return False, errors
    
    # Check if operation is valid
    if node["op"] not in VALID_OPS:
        errors.append(f"Node {node['id']} has invalid operation '{node['op']}'")
        return False, errors
    
    # Check required parameters for the operation
    op_spec = VALID_OPS[node["op"]]
    for param in op_spec["required"]:
        if param not in node:
            errors.append(f"Node {node['id']} missing required parameter '{param}' for operation '{node['op']}'")
            return False, errors
    
    # Validate parameter types
    if node["op"] in ["rolling_return", "ema", "delay"]:
        if "n" in node and not isinstance(node["n"], (int, float)):
            errors.append(f"Node {node['id']} parameter 'n' must be numeric, got {type(node['n'])}")
        if "d" in node and not isinstance(node["d"], (int, float)):
            errors.append(f"Node {node['id']} parameter 'd' must be numeric, got {type(node['d'])}")
    
    if node["op"] == "winsor_quantile":
        if "q" in node and not isinstance(node["q"], (int, float)):
            errors.append(f"Node {node['id']} parameter 'q' must be numeric, got {type(node['q'])}")
        if "q" in node and isinstance(node["q"], (int, float)) and (node["q"] <= 0 or node["q"] >= 0.5):
            errors.append(f"Node {node['id']} parameter 'q' must be between 0 and 0.5, got {node['q']}")
    
    if node["op"] == "clip":
        if "lo" in node and not isinstance(node["lo"], (int, float)):
            errors.append(f"Node {node['id']} parameter 'lo' must be numeric, got {type(node['lo'])}")
        if "hi" in node and not isinstance(node["hi"], (int, float)):
            errors.append(f"Node {node['id']} parameter 'hi' must be numeric, got {type(node['hi'])}")
    
    if node["op"] == "combine":
        if "inputs" in node and not isinstance(node["inputs"], list):
            errors.append(f"Node {node['id']} parameter 'inputs' must be a list, got {type(node['inputs'])}")
        elif "inputs" in node and isinstance(node["inputs"], list):
            # Check that all inputs are strings (node IDs)
            for i, inp in enumerate(node["inputs"]):
                if not isinstance(inp, str):
                    errors.append(f"Node {node['id']} parameter 'inputs'[{i}] must be a string, got {type(inp)}")
        
        if "weights" in node and not isinstance(node["weights"], list):
            errors.append(f"Node {node['id']} parameter 'weights' must be a list, got {type(node['weights'])}")
        elif "weights" in node and isinstance(node["weights"], list):
            # Check that all weights are numeric
            for i, w in enumerate(node["weights"]):
                if not isinstance(w, (int, float)):
                    errors.append(f"Node {node['id']} parameter 'weights'[{i}] must be numeric, got {type(w)}")
            
            # Check that weights and inputs have same length if both provided
            if "inputs" in node and isinstance(node["inputs"], list) and len(node["weights"]) != len(node["inputs"]):
                errors.append(f"Node {node['id']} 'weights' and 'inputs' must have same length, got {len(node['weights'])} vs {len(node['inputs'])}")
    
    return True, errors

def validate_dag_properties(program: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that the program forms a valid DAG and can reach the output.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Build node mapping
    nodes = {node["id"]: node for node in program["nodes"]}
    output_id = program["output"]
    
    # Check if output node exists
    if output_id not in nodes:
        errors.append(f"Output node '{output_id}' not found in nodes")
        return False, errors
    
    # Check for circular dependencies and build dependency graph
    dependencies = {}
    for node in program["nodes"]:
        node_id = node["id"]
        deps = set()
        
        # Collect dependencies based on operation
        op = node["op"]
        if op in ["ema", "zscore_xs", "demean_xs", "winsor_quantile", "clip", "delay"]:
            if "src" in node:
                deps.add(node["src"])
        elif op in ["add", "sub", "mul"]:
            if "a" in node:
                deps.add(node["a"])
            if "b" in node:
                deps.add(node["b"])
        elif op == "combine":
            if "inputs" in node:
                deps.update(node["inputs"])
        
        dependencies[node_id] = deps
        
        # Check if dependencies reference valid nodes
        for dep in deps:
            if dep not in nodes:
                errors.append(f"Node '{node_id}' references undefined node '{dep}'")
    
    if errors:
        return False, errors
    
    # Check for circular dependencies using DFS
    visited = set()
    rec_stack = set()
    
    def has_cycle(node_id: str) -> bool:
        if node_id in rec_stack:
            return True
        if node_id in visited:
            return False
        
        visited.add(node_id)
        rec_stack.add(node_id)
        
        for dep in dependencies.get(node_id, []):
            if has_cycle(dep):
                return True
        
        rec_stack.remove(node_id)
        return False
    
    # Check all nodes for cycles
    for node_id in nodes:
        if node_id not in visited:
            if has_cycle(node_id):
                errors.append(f"Circular dependency detected involving node '{node_id}'")
                return False, errors
    
    # Check if output is reachable from all nodes (optional - could be strict)
    # For now, we'll just ensure the output node exists and has no cycles
    
    return True, errors

def validate_program(program: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of a JSON program.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    all_errors = []
    
    # Step 1: Validate basic structure
    is_valid, errors = validate_program_structure(program)
    if not is_valid:
        all_errors.extend(errors)
        return False, all_errors
    
    # Step 2: Validate each node
    for node in program["nodes"]:
        is_valid, errors = validate_node_structure(node, node.get("id", "unknown"))
        if not is_valid:
            all_errors.extend(errors)
    
    # Step 3: Validate DAG properties
    is_valid, errors = validate_dag_properties(program)
    if not is_valid:
        all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors

def validate_action(action: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate different action types.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if "type" not in action:
        errors.append("Action missing 'type' key")
        return False, errors
    
    action_type = action["type"]
    
    # if action_type == "SET_PROGRAM":
    #     if "program" not in action:
    #         errors.append("SET_PROGRAM action missing 'program' key")
    #     else:
    #         is_valid, prog_errors = validate_program(action["program"])
    #         if not is_valid:
    #             errors.extend([f"Program validation failed: {err}" for err in prog_errors])
    
    # elif action_type == "SET_PARAMS":
    #     if "params" not in action:
    #         errors.append("SET_PARAMS action missing 'params' key")
    #     else:
    #         params = action["params"]
    #         if not isinstance(params, dict):
    #             errors.append("SET_PARAMS params must be a dictionary")
    #         else:
    #             # Validate parameter ranges
    #             if "top_q" in params:
    #                 q = params["top_q"]
    #                 if not isinstance(q, (int, float)) or q <= 0 or q >= 0.5:
    #                     errors.append("top_q must be between 0 and 0.5")
                
    #             if "turnover_cap" in params:
    #                 tc = params["turnover_cap"]
    #                 if not isinstance(tc, (int, float)) or tc <= 0:
    #                     errors.append("turnover_cap must be positive")
                
    #             if "delay_days" in params:
    #                 dd = params["delay_days"]
    #                 if not isinstance(dd, int) or dd < 0:
    #                     errors.append("delay_days must be a non-negative integer")
    
    # elif action_type == "EVALUATE":
    #     # No additional validation needed
    #     pass
    
    # elif action_type == "REFLECT":
    #     if "note" not in action:
    #         errors.append("REFLECT action missing 'note' key")
    
    # elif action_type == "STOP":
    #     # No additional validation needed
    #     pass
    
    if action_type == "OBSERVE":
        if "tool" not in action:
            errors.append("OBSERVE action missing 'tool' key")
        else:
            tool = action["tool"]
            valid_tools = ["describe_data", "plot_returns", "analyze_factor_performance"]
            if tool not in valid_tools:
                errors.append(f"Invalid tool '{tool}'. Valid tools: {valid_tools}")
            
            # Validate tool-specific parameters
            if tool == "analyze_factor_performance":
                if "factor_program" not in action:
                    errors.append("analyze_factor_performance requires 'factor_program' parameter")
    
    elif action_type == "FACTOR_IMPROVE":
        if "new_factor" not in action:
            errors.append("FACTOR_IMPROVE action missing 'new_factor' key")
        else:
            # Validate the new factor program
            is_valid, prog_errors = validate_program(action["new_factor"])
            if not is_valid:
                errors.extend([f"New factor validation failed: {err}" for err in prog_errors])
        
        if "weight" not in action:
            errors.append("FACTOR_IMPROVE action missing 'weight' key")
        else:
            weight = action["weight"]
            if not isinstance(weight, (int, float)) or weight <= 0 or weight > 1:
                errors.append("weight must be a number between 0 and 1")
    
    else:
        errors.append(f"Unknown action type: {action_type}")
    
    return len(errors) == 0, errors

