#!/bin/bash

# Script to run table connectivity experiments with different V and num_training_examples values
# V: 5, 20
# num_training_examples: 256, 1000000
# Each combination runs 5 times

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run the Python experiment script
python3 "$SCRIPT_DIR/run_experiments.py"

