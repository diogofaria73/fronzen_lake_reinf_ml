#!/usr/bin/env python3

"""
Entry point script for the Frozen Lake Q-Learning project.
This script simply imports and runs the main function from src/main.py.
"""

import os
import sys
from pathlib import Path

# Add src directory to the path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

# Import the main function
from main import main

if __name__ == "__main__":
    # Run the main function
    main() 