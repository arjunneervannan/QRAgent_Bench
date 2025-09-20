#!/usr/bin/env python3
"""
Simple runner for debug training loop.
Usage: python run_debug.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from debug_training_loop import main

if __name__ == "__main__":
    main()
