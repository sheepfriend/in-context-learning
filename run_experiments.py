#!/usr/bin/env python3
"""
Script to run table connectivity experiments with different configurations
- V: 5, 20
- num_training_examples: 256, 512, 1024, 2048, 4096
- model_type: gpt2 (standard), lowrank_gpt2 (low-rank)
Each combination runs 3 times
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
V_VALUES = [5]
NUM_EXAMPLES = [2**i for i in range(14,17)]
MODEL_TYPES = ["lowrank_gpt2","gpt2"]
NUM_RUNS = 5

# Base config files
BASE_CONFIG = "conf/table_connectivity.yaml"
LOWRANK_CONFIG = "conf/table_connectivity_lowrank.yaml"

def run_experiment(V, num_examples, model_type, run_idx):
    """Run a single experiment with specified parameters"""
    
    # Create a unique run name
    model_tag = "lowrank" if model_type == "lowrank_gpt2" else "standard"
    run_name = f"table_connectivity_{model_tag}_V{V}_N{num_examples}_run{run_idx}"
    
    print(f"\n{'='*50}")
    print(f"Run {run_idx}/{NUM_RUNS}: V={V}, num_examples={num_examples}, model={model_type}")
    print(f"Run name: {run_name}")
    print(f"{'='*50}")
    
    # Load base config based on model type
    base_config_file = LOWRANK_CONFIG if model_type == "lowrank_gpt2" else BASE_CONFIG
    with open(base_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update parameters
    config['training']['task_kwargs'] = {"V": V, "C": 3, "rho": 0.5}
    config['training']['num_training_examples'] = num_examples
    config['wandb']['name'] = run_name
    
    # For lowrank model, update V and C in model config
    if model_type == "lowrank_gpt2":
        config['model']['V'] = V
        config['model']['C'] = 3
        # Update n_positions based on V and C
        config['model']['n_positions'] = V * (3 + 1) + 3  # V*(C+1)+3
    
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
    total_experiments = len(V_VALUES) * len(NUM_EXAMPLES) * len(MODEL_TYPES) * NUM_RUNS
    completed = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"Starting {total_experiments} experiments")
    print(f"V values: {V_VALUES}")
    print(f"Num examples: {NUM_EXAMPLES}")
    print(f"Model types: {MODEL_TYPES}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"{'='*60}")
    
    for V in V_VALUES:
        for num_examples in NUM_EXAMPLES:
            for model_type in MODEL_TYPES:
                print(f"\n{'#'*60}")
                print(f"Experiments with V={V}, num_examples={num_examples}, model={model_type}")
                print(f"{'#'*60}")
                
                for run_idx in range(1, NUM_RUNS + 1):
                    success = run_experiment(V, num_examples, model_type, run_idx)
                    if success:
                        completed += 1
                    else:
                        failed += 1
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Successful: {completed}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

