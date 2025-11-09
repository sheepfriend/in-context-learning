#!/usr/bin/env python3
"""
Parse all log files from the logs directory and extract test results.
This script aggregates results from individual GPU log files.
"""

import re
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_log_file(log_file):
    """Parse a single log file to extract test results"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Regex patterns
        acc_pattern = r"Acc:\s+tensor\(([0-9.]+)(?:,\s+device='cuda:\d+')?\)"
        p_y1_pattern = r"P\(y=1\):\s+tensor\(([0-9.]+)(?:,\s+device='cuda:\d+')?\)"
        p_hat_pattern = r"P\(hat_y=1\):\s+tensor\(([0-9.]+)(?:,\s+device='cuda:\d+')?\)"
        testing_pattern = r"Testing\.\.\."
        
        # Find the last "Testing..." block
        testing_matches = list(re.finditer(testing_pattern, content))
        if not testing_matches:
            return None
        
        # Take the last testing block
        last_testing_match = testing_matches[-1]
        testing_start_pos = last_testing_match.end()
        
        # Extract the relevant lines after "Testing..."
        relevant_lines = content[testing_start_pos:].split('\n')
        
        acc, p_y1, p_hat_y1 = None, None, None
        for line in relevant_lines:
            if acc is None:
                acc_m = re.search(acc_pattern, line)
                if acc_m:
                    acc = float(acc_m.group(1))
            if p_y1 is None:
                p_y1_m = re.search(p_y1_pattern, line)
                if p_y1_m:
                    p_y1 = float(p_y1_m.group(1))
            if p_hat_y1 is None:
                p_hat_y1_m = re.search(p_hat_pattern, line)
                if p_hat_y1_m:
                    p_hat_y1 = float(p_hat_y1_m.group(1))
            
            if acc is not None and p_y1 is not None and p_hat_y1 is not None:
                break
        
        if acc is not None:
            return {
                'test_acc': acc,
                'test_p_y1': p_y1,
                'test_p_hat_y1': p_hat_y1,
            }
        
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return None

def parse_filename(log_file):
    """Extract experiment parameters from log filename"""
    # Expected format: table_connectivity_{standard|lowrank}_{fixed|random}_V{V}_N{num_examples}_run{run_idx}_gpu{gpu_id}.log
    filename = log_file.stem  # Remove .log extension
    
    pattern = r"table_connectivity_(standard|lowrank)_(fixed|random)_V(\d+)_N(\d+)_run(\d+)_gpu(\d+)"
    match = re.match(pattern, filename)
    
    if match:
        model_tag, sampler_tag, V, num_examples, run_idx, gpu_id = match.groups()
        
        # Determine model type
        is_lowrank = (model_tag == 'lowrank')
        is_fixed = (sampler_tag == 'fixed')
        
        if is_lowrank and is_fixed:
            model_type = 'lowrank_gpt2_fixed'
            model_name = 'Low-Rank-Fixed'
        elif is_lowrank:
            model_type = 'lowrank_gpt2'
            model_name = 'Low-Rank'
        elif is_fixed:
            model_type = 'gpt2_fixed'
            model_name = 'Standard-Fixed'
        else:
            model_type = 'gpt2'
            model_name = 'Standard'
        
        return {
            'model_type': model_type,
            'model_name': model_name,
            'sampler_type': sampler_tag,
            'V': int(V),
            'num_examples': int(num_examples),
            'run_idx': int(run_idx),
            'gpu_id': int(gpu_id),
        }
    
    return None

def create_summary_table(df):
    """Create summary table with mean and std"""
    summary_rows = []
    
    V_values = sorted(df['V'].unique())
    num_examples_values = sorted(df['num_examples'].unique())
    model_types = sorted(df['model_type'].unique(), reverse=True)  # lowrank first
    
    for V in V_values:
        for num_examples in num_examples_values:
            for model_type in model_types:
                subset = df[(df['V'] == V) & 
                           (df['num_examples'] == num_examples) & 
                           (df['model_type'] == model_type)]
                
                if not subset.empty:
                    model_name = subset['model_name'].iloc[0]
                    summary_rows.append({
                        'V': V,
                        'num_examples': num_examples,
                        'model': model_name,
                        'test_acc_mean': subset['test_acc'].mean(),
                        'test_acc_std': subset['test_acc'].std(),
                        'test_p_y1_mean': subset['test_p_y1'].mean(),
                        'test_p_y1_std': subset['test_p_y1'].std(),
                        'test_p_hat_y1_mean': subset['test_p_hat_y1'].mean(),
                        'test_p_hat_y1_std': subset['test_p_hat_y1'].std(),
                        'num_runs': len(subset),
                    })
    
    return pd.DataFrame(summary_rows)

def create_comparison_table(df):
    """Create comparison table between all model variants"""
    summary = create_summary_table(df)
    
    comparison_rows = []
    V_values = sorted(df['V'].unique())
    num_examples_values = sorted(df['num_examples'].unique())
    
    for V in V_values:
        for num_examples in num_examples_values:
            row_data = {'V': V, 'N': num_examples}
            
            # Get each model type
            for model_name in ['Standard', 'Standard-Fixed', 'Low-Rank', 'Low-Rank-Fixed']:
                subset = summary[(summary['V'] == V) & 
                                (summary['num_examples'] == num_examples) & 
                                (summary['model'] == model_name)]
                
                if not subset.empty:
                    mean_acc = subset['test_acc_mean'].iloc[0]
                    std_acc = subset['test_acc_std'].iloc[0]
                    row_data[model_name] = f"{mean_acc:.4f}±{std_acc:.4f}"
                else:
                    row_data[model_name] = "N/A"
            
            comparison_rows.append(row_data)
    
    df_comp = pd.DataFrame(comparison_rows)
    if not df_comp.empty:
        df_comp = df_comp.set_index(['V', 'N'])
    
    return df_comp

def main():
    parser = argparse.ArgumentParser(description="Parse log files from parallel GPU runs.")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="Directory containing log files.")
    parser.add_argument("--output", type=str, default="parsed_results",
                        help="Base name for output files (CSV, Excel).")
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return
    
    print(f"Parsing log files from: {logs_dir.absolute()}")
    
    # Find all log files
    log_files = list(logs_dir.glob("table_connectivity_*.log"))
    
    if not log_files:
        print(f"❌ No log files found in {logs_dir}")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Parse all log files
    results = []
    for log_file in log_files:
        print(f"Parsing: {log_file.name}")
        
        # Extract parameters from filename
        params = parse_filename(log_file)
        if params is None:
            print(f"  ⚠️  Could not parse filename: {log_file.name}")
            continue
        
        # Extract test results from log content
        test_results = parse_log_file(log_file)
        if test_results is None:
            print(f"  ⚠️  Could not extract test results from: {log_file.name}")
            continue
        
        # Combine parameters and results
        result = {**params, **test_results}
        results.append(result)
        print(f"  ✓ V={result['V']}, N={result['num_examples']}, "
              f"{result['model_name']}, run={result['run_idx']}, "
              f"Acc={result['test_acc']:.4f}")
    
    if not results:
        print("\n❌ No results extracted!")
        return
    
    df_results = pd.DataFrame(results)
    print(f"\n✓ Successfully parsed {len(df_results)} experiments")
    
    # Sort results
    df_results = df_results.sort_values(
        ['V', 'num_examples', 'model_type', 'run_idx']
    ).reset_index(drop=True)
    
    # Save detailed results
    detailed_output_path = f"{args.output}_detailed.csv"
    df_results.to_csv(detailed_output_path, index=False)
    print(f"\n✓ Saved detailed results: {detailed_output_path}")
    
    # Create and save summary results
    df_summary = create_summary_table(df_results)
    summary_output_path = f"{args.output}_summary.csv"
    df_summary.to_csv(summary_output_path, index=False)
    print(f"✓ Saved summary results: {summary_output_path}")
    
    # Create and save Excel file
    excel_output_path = f"{args.output}.xlsx"
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Detailed Results', index=False)
        df_summary.to_excel(writer, sheet_name='Summary Table', index=False)
        
        df_comparison = create_comparison_table(df_results)
        if not df_comparison.empty:
            df_comparison.to_excel(writer, sheet_name='Accuracy Comparison', index=True)
    print(f"✓ Saved Excel: {excel_output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary by Configuration".center(80))
    print("="*80 + "\n")
    
    V_values = sorted(df_results['V'].unique())
    num_examples_values = sorted(df_results['num_examples'].unique())
    model_types = sorted(df_results['model_type'].unique(), reverse=True)
    
    for V in V_values:
        print(f"\n{'='*80}")
        print(f"V = {V}".center(80))
        print("="*80 + "\n")
        for num_examples in num_examples_values:
            print(f"\n{'-'*80}")
            print(f"num_examples = {num_examples}".center(80))
            print("-"*80 + "\n")
            for model_type in model_types:
                subset = df_results[
                    (df_results['V'] == V) & 
                    (df_results['num_examples'] == num_examples) & 
                    (df_results['model_type'] == model_type)
                ]
                if not subset.empty:
                    model_name = subset['model_name'].iloc[0]
                    print(f"  {model_name} Transformer:")
                    for _, row in subset.iterrows():
                        print(f"    Run {row['run_idx']} (GPU {row['gpu_id']}): "
                              f"Acc={row['test_acc']:.4f}, "
                              f"P(y=1)={row['test_p_y1']:.4f}, "
                              f"P(ŷ=1)={row['test_p_hat_y1']:.4f}")
                    
                    mean_acc = subset['test_acc'].mean()
                    std_acc = subset['test_acc'].std()
                    print(f"    平均: Acc={mean_acc:.4f}±{std_acc:.4f}\n")
    
    print("\n" + "="*80)
    print("Comparison Table".center(80))
    print("="*80 + "\n")
    df_comparison = create_comparison_table(df_results)
    if not df_comparison.empty:
        print(df_comparison.to_string())
    
    print("\n" + "="*80)
    print("✓ Done!".center(80))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

