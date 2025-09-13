"""
Skeleton for building preference pairs (A vs B) from the environment to train a small LLM via DPO.

Usage outline:
1) Sample a set of states (e.g., different initial params/programs).
2) For each state, ask the current policy/model for two candidate SET_PROGRAM (or SET_PARAMS) actions.
3) Roll each candidate for a short mini-episode ending with EVALUATE.
4) Choose the winner by higher OOS Sharpe (or full shaped reward).
5) Save (state, action_A, action_B, winner) to JSONL for DPO training.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

def save_pairs(pairs: List[Dict[str, Any]], out_path: str = "training/prefs.jsonl"):
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    print("This is a stub. Fill in sampling and evaluation with your LLM + environment.")
