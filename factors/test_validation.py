#!/usr/bin/env python3
"""
Test script to demonstrate the JSON program validator.
Shows various invalid programs and how they're caught.
"""

import json
from validate import validate_program, validate_action

def test_invalid_programs():
    """Test various invalid program structures."""
    
    print("=== Testing Invalid Programs ===\n")
    
    # Test 1: Missing nodes
    invalid1 = {
        "output": "score"
    }
    print("1. Program missing 'nodes':")
    is_valid, errors = validate_program(invalid1)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 2: Missing output
    invalid2 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126}
        ]
    }
    print("2. Program missing 'output':")
    is_valid, errors = validate_program(invalid2)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 3: Invalid operation
    invalid3 = {
        "nodes": [
            {"id": "x0", "op": "invalid_op", "n": 126},
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    print("3. Program with invalid operation:")
    is_valid, errors = validate_program(invalid3)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 4: Missing required parameter
    invalid4 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return"},  # missing 'n'
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    print("4. Program missing required parameter:")
    is_valid, errors = validate_program(invalid4)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 5: Circular dependency
    invalid5 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "add", "a": "x0", "b": "x2"},
            {"id": "x2", "op": "add", "a": "x1", "b": "x0"},
            {"id": "score", "op": "zscore_xs", "src": "x1"}
        ],
        "output": "score"
    }
    print("5. Program with circular dependency:")
    is_valid, errors = validate_program(invalid5)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 6: Undefined node reference
    invalid6 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "score", "op": "zscore_xs", "src": "nonexistent"}
        ],
        "output": "score"
    }
    print("6. Program with undefined node reference:")
    is_valid, errors = validate_program(invalid6)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 7: Invalid parameter type
    invalid7 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": "not_a_number"},
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    print("7. Program with invalid parameter type:")
    is_valid, errors = validate_program(invalid7)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 8: Invalid winsor_quantile parameter
    invalid8 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "winsor_quantile", "src": "x0", "q": 0.6},  # q > 0.5
            {"id": "score", "op": "zscore_xs", "src": "x1"}
        ],
        "output": "score"
    }
    print("8. Program with invalid winsor_quantile parameter:")
    is_valid, errors = validate_program(invalid8)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()

def test_invalid_actions():
    """Test various invalid actions."""
    
    print("=== Testing Invalid Actions ===\n")
    
    # Test 1: Missing action type
    invalid_action1 = {
        "program": {"nodes": [], "output": "score"}
    }
    print("1. Action missing 'type':")
    is_valid, errors = validate_action(invalid_action1)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 2: Invalid action type
    invalid_action2 = {
        "type": "INVALID_ACTION"
    }
    print("2. Action with invalid type:")
    is_valid, errors = validate_action(invalid_action2)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 3: SET_PROGRAM with invalid program
    invalid_action3 = {
        "type": "SET_PROGRAM",
        "program": {"output": "score"}  # missing nodes
    }
    print("3. SET_PROGRAM with invalid program:")
    is_valid, errors = validate_action(invalid_action3)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 4: SET_PARAMS with invalid parameters
    invalid_action4 = {
        "type": "SET_PARAMS",
        "params": {
            "top_q": 0.6,  # > 0.5
            "turnover_cap": -1,  # negative
            "delay_days": "not_an_int"  # wrong type
        }
    }
    print("4. SET_PARAMS with invalid parameters:")
    is_valid, errors = validate_action(invalid_action4)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()
    
    # Test 5: REFLECT without note
    invalid_action5 = {
        "type": "REFLECT"
        # missing note
    }
    print("5. REFLECT without note:")
    is_valid, errors = validate_action(invalid_action5)
    print(f"   Valid: {is_valid}")
    for err in errors:
        print(f"   Error: {err}")
    print()

def test_valid_programs():
    """Test valid programs to ensure they pass validation."""
    
    print("=== Testing Valid Programs ===\n")
    
    # Test 1: Simple valid program
    valid1 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "score", "op": "zscore_xs", "src": "x0"}
        ],
        "output": "score"
    }
    print("1. Simple valid program:")
    is_valid, errors = validate_program(valid1)
    print(f"   Valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"   Error: {err}")
    else:
        print("   ✓ No errors")
    print()
    
    # Test 2: Complex valid program (like baseline)
    valid2 = {
        "nodes": [
            {"id": "x0", "op": "rolling_return", "n": 126},
            {"id": "x1", "op": "rolling_return", "n": 21},
            {"id": "x2", "op": "sub", "a": "x0", "b": "x1"},
            {"id": "x3", "op": "winsor_quantile", "src": "x2", "q": 0.02},
            {"id": "score", "op": "zscore_xs", "src": "x3"}
        ],
        "output": "score"
    }
    print("2. Complex valid program (baseline-like):")
    is_valid, errors = validate_program(valid2)
    print(f"   Valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"   Error: {err}")
    else:
        print("   ✓ No errors")
    print()

if __name__ == "__main__":
    test_valid_programs()
    test_invalid_programs()
    test_invalid_actions()
