#!/usr/bin/env python3
"""
Script to run table connectivity experiments with different configurations
- V: 5, 20
- num_training_examples: 256, 512, 1024, 2048, 4096
- model_type: gpt2 (standard), lowrank_gpt2 (low-rank)
Each combination runs 3 times
Uses 4 GPUs in parallel
"""

import os
import subprocess
import yaml
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Navigate to src directory
src_dir = Path(__file__).parent / "src"
os.chdir(src_dir)

# Experiment parameters
V_VALUES = [5]
NUM_EXAMPLES = [2**i for i in range(15,19)]
MODEL_TYPES = ["gpt2", "gpt2_fixed", "lowrank_gpt2", "lowrank_gpt2_fixed"]
NUM_RUNS = 20
NUM_GPUS = 4  # 使用4个GPU

# Base config files
BASE_CONFIG = "conf/table_connectivity.yaml"
BASE_CONFIG_FIXED = "conf/table_connectivity_fixed.yaml"
LOWRANK_CONFIG = "conf/table_connectivity_lowrank.yaml"
LOWRANK_CONFIG_FIXED = "conf/table_connectivity_lowrank_fixed.yaml"

# Create logs directory
LOGS_DIR = Path("../logs")
LOGS_DIR.mkdir(exist_ok=True)

def run_experiment(V, num_examples, model_type, run_idx, gpu_id):
    """Run a single experiment with specified parameters on a specific GPU"""
    
    # Determine model tag and sampler type
    is_lowrank = "lowrank" in model_type
    is_fixed = "fixed" in model_type
    
    model_tag = "lowrank" if is_lowrank else "standard"
    sampler_tag = "fixed" if is_fixed else "random"
    
    run_name = f"table_connectivity_{model_tag}_{sampler_tag}_V{V}_N{num_examples}_run{run_idx}"
    
    # Create log file for this experiment
    log_file = LOGS_DIR / f"{run_name}_gpu{gpu_id}.log"
    
    print(f"\n[GPU {gpu_id}] Starting: V={V}, N={num_examples}, {model_tag}, {sampler_tag}, run={run_idx}")
    print(f"[GPU {gpu_id}] Log file: {log_file}")
    
    # Load base config based on model type
    if model_type == "lowrank_gpt2":
        base_config_file = LOWRANK_CONFIG
    elif model_type == "lowrank_gpt2_fixed":
        base_config_file = LOWRANK_CONFIG_FIXED
    elif model_type == "gpt2_fixed":
        base_config_file = BASE_CONFIG_FIXED
    else:  # gpt2
        base_config_file = BASE_CONFIG
    
    with open(base_config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update parameters
    config['training']['task_kwargs'] = {"V": V, "C": 3, "rho": 0.5}
    config['training']['num_training_examples'] = num_examples
    config['wandb']['name'] = run_name
    
    # For lowrank model, update V and C in model config
    if is_lowrank:
        config['model']['V'] = V
        config['model']['C'] = 3
        # Update n_positions based on V and C
        config['model']['n_positions'] = V * (3 + 1) + 3  # V*(C+1)+3
    
    # Create temporary config file with unique name
    temp_config_path = f"conf/temp_config_{uuid.uuid4().hex[:8]}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Run training with specific GPU
        cmd = ["python", "train.py", "--config", temp_config_path]
        
        # Set environment to use specific GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Run and save output to log file
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd, 
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False
            )
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ Completed: {run_name}")
            return {'success': True, 'run_name': run_name, 'gpu': gpu_id, 'log': str(log_file)}
        else:
            print(f"[GPU {gpu_id}] ✗ Failed: {run_name}")
            return {'success': False, 'run_name': run_name, 'gpu': gpu_id, 'log': str(log_file)}
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Error: {run_name} - {e}")
        return {'success': False, 'run_name': run_name, 'gpu': gpu_id, 'error': str(e)}
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    """Run all experiments using 4 GPUs in parallel"""
    # Build list of all experiments
    experiments = []
    for V in V_VALUES:
        for num_examples in NUM_EXAMPLES:
            for model_type in MODEL_TYPES:
                for run_idx in range(1, NUM_RUNS + 1):
                    experiments.append({
                        'V': V,
                        'num_examples': num_examples,
                        'model_type': model_type,
                        'run_idx': run_idx
                    })
    
    total_experiments = len(experiments)
    
    print(f"\n{'='*80}")
    print(f"Starting {total_experiments} experiments using {NUM_GPUS} GPUs in parallel")
    print(f"V values: {V_VALUES}")
    print(f"Num examples: {NUM_EXAMPLES}")
    print(f"Model types: {MODEL_TYPES}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Logs directory: {LOGS_DIR.absolute()}")
    print(f"{'='*80}\n")
    
    results = []
    completed = 0
    failed = 0
    
    # Use ProcessPoolExecutor to run experiments in parallel
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        # Submit all experiments, cycling through GPUs
        future_to_exp = {}
        for i, exp in enumerate(experiments):
            gpu_id = i % NUM_GPUS  # Cycle through GPUs 0, 1, 2, 3
            future = executor.submit(
                run_experiment,
                exp['V'],
                exp['num_examples'],
                exp['model_type'],
                exp['run_idx'],
                gpu_id
            )
            future_to_exp[future] = exp
        
        # Collect results as they complete
        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    completed += 1
                else:
                    failed += 1
                
                # Progress update
                total_done = completed + failed
                print(f"\n{'='*80}")
                print(f"Progress: {total_done}/{total_experiments} completed "
                      f"(✓ {completed}, ✗ {failed})")
                print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"✗ Experiment failed with exception: {e}")
                failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total: {total_experiments}")
    print(f"Successful: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*completed/total_experiments:.1f}%")
    print(f"\nLogs saved to: {LOGS_DIR.absolute()}")
    print(f"{'='*80}\n")
    
    # Save summary
    summary_file = LOGS_DIR / "experiment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Experiment Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Successful: {completed}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write(f"Results:\n")
        f.write(f"{'-'*80}\n")
        for result in results:
            status = "✓" if result['success'] else "✗"
            f.write(f"{status} {result['run_name']} (GPU {result['gpu']})\n")
            f.write(f"   Log: {result.get('log', 'N/A')}\n")
            if 'error' in result:
                f.write(f"   Error: {result['error']}\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()

