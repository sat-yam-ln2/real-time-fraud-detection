#!/usr/bin/env python
"""
Wrapper script for analyze_telemetry.py

This script exists at the project root for easy access and calls the actual
implementation in src/utils/analyze_telemetry.py.

Usage:
    python analyze_telemetry.py [options]
    
See src/utils/analyze_telemetry.py for full documentation.
"""

import sys
import os
from pathlib import Path

# Get the absolute path to the actual analyzer script
SCRIPT_PATH = Path(__file__).parent / 'src' / 'utils' / 'analyze_telemetry.py'

if __name__ == "__main__":
    # Check if the script exists
    if not SCRIPT_PATH.exists():
        print(f"Error: Could not find analyze_telemetry.py at {SCRIPT_PATH}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Forward all arguments to the actual script
    args = ' '.join(sys.argv[1:])
    cmd = f'python "{SCRIPT_PATH}" {args}'
    
    # Execute the command
    print(f"Running: {cmd}")
    sys.exit(os.system(cmd))