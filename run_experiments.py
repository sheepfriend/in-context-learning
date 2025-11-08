#!/usr/bin/env python3
"""
Script to run table connectivity experiments with different V and num_training_examples values
V: 5, 20
num_training_examples: 256, 1000000
Each combination runs 5 times
"""

import os
import subprocess
import yaml
import uuid
from pathlib import Path

# Navigate to src directory
src_dir = Path(__file__).parent / "src"
os.chdir(src_dir)

# Experiment parameters
V_VALUES = [5, 20]
NUM_EXAMPLES = [1000000, 256]
NUM_RUNS = 5

# Base config file
BASE_CONFIG = "conf/table_connectivity.yaml"

def run_experiment(V, num_examples, run_idx):
    """Run a single experiment with specified parameters"""
    
    # Create a unique run name
    run_name = f"table_connectivity_V{V}_N{num_examples}_run{run_idx}"
    
    print(f"\n{'='*50}")
    print(f"Run {run_idx}/{NUM_RUNS}: V={V}, num_examples={num_examples}")
    print(f"Run name: {run_name}")
    print(f"{'='*50}")
    
    # Load base config
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update parameters
    config['training']['task_kwargs'] = {"V": V, "C": 3, "rho": 0.5}
    config['training']['num_training_examples'] = num_examples
    config['wandb']['name'] = run_name
    
    # Create temporary config file
    temp_config_path = f"conf/temp_config_{uuid.uuid4().hex[:8]}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Run training
        cmd = ["python", "train.py", "--config", temp_config_path]
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print(f"✓ Successfully completed run {run_idx}")
            return True
        else:
            print(f"✗ Failed on run {run_idx}")
            return False
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    """Run all experiments"""
    total_experiments = len(V_VALUES) * len(NUM_EXAMPLES) * NUM_RUNS
    completed = 0
    failed = 0
    
    print(f"\n{'='*50}")
    print(f"Starting {total_experiments} experiments")
    print(f"{'='*50}")
    
    for V in V_VALUES:
        for num_examples in NUM_EXAMPLES:
            print(f"\n{'#'*50}")
            print(f"Experiments with V={V}, num_training_examples={num_examples}")
            print(f"{'#'*50}")
            
            for run_idx in range(1, NUM_RUNS + 1):
                success = run_experiment(V, num_examples, run_idx)
                if success:
                    completed += 1
                else:
                    failed += 1
    
    print(f"\n{'='*50}")
    print(f"All experiments completed!")
    print(f"Successful: {completed}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()

