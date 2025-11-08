#!/usr/bin/env python3
"""
简单提取nohup.out中所有Testing...后面的结果
按照run_experiments.py的执行顺序排列
"""

import re
import pandas as pd
import argparse

def extract_testing_results(filepath):
    """提取所有Testing...后面的结果"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 分割成行
    lines = content.split('\n')
    
    # 找到所有Testing...的位置
    results = []
    
    for i, line in enumerate(lines):
        if "Testing..." in line:
            # 在接下来的几行中查找Acc, P(y=1), P(hat_y=1)
            acc = None
            p_y1 = None
            p_hat_y1 = None
            
            for j in range(i+1, min(i+10, len(lines))):
                # 匹配 Acc: tensor(0.7461, device='cuda:0')
                if acc is None:
                    acc_match = re.search(r"Acc:\s+tensor\(([0-9.]+)(?:,\s+device='cuda:\d+')?\)", lines[j])
                    if acc_match:
                        acc = float(acc_match.group(1))
                
                # 匹配 P(y=1): tensor(0.2695, device='cuda:0')
                if p_y1 is None:
                    p_y1_match = re.search(r"P\(y=1\):\s+tensor\(([0-9.]+)(?:,\s+device='cuda:\d+')?\)", lines[j])
                    if p_y1_match:
                        p_y1 = float(p_y1_match.group(1))
                
                # 匹配 P(hat_y=1): tensor(0.0781, device='cuda:0')
                if p_hat_y1 is None:
                    p_hat_match = re.search(r"P\(hat_y=1\):\s+tensor\(([0-9.]+)(?:,\s+device='cuda:\d+')?\)", lines[j])
                    if p_hat_match:
                        p_hat_y1 = float(p_hat_match.group(1))
                
                # 如果都找到了，或者遇到下一个Testing，停止
                if (acc is not None and p_y1 is not None and p_hat_y1 is not None) or "Testing..." in lines[j]:
                    break
            
            if acc is not None:
                results.append({
                    'acc': acc,
                    'p_y1': p_y1,
                    'p_hat_y1': p_hat_y1
                })
    
    return results

def arrange_by_experiment_order(results):
    """按照run_experiments.py的顺序排列结果"""
    
    # 根据run_experiments.py的设置
    V_VALUES = [5]
    NUM_EXAMPLES = [2**i for i in range(14, 17)]  # [16384, 32768, 65536]
    MODEL_TYPES = ["lowrank_gpt2", "gpt2"]
    NUM_RUNS = 5
    
    arranged_data = []
    idx = 0
    
    for V in V_VALUES:
        for num_examples in NUM_EXAMPLES:
            for model_type in MODEL_TYPES:
                model_name = "Low-Rank" if model_type == "lowrank_gpt2" else "Standard"
                
                for run_idx in range(1, NUM_RUNS + 1):
                    if idx < len(results):
                        result = results[idx]
                        arranged_data.append({
                            'V': V,
                            'num_examples': num_examples,
                            'model': model_name,
                            'run': run_idx,
                            'test_acc': result['acc'],
                            'test_p_y1': result['p_y1'],
                            'test_p_hat_y1': result['p_hat_y1']
                        })
                        idx += 1
                    else:
                        # 如果结果不够，用None填充
                        arranged_data.append({
                            'V': V,
                            'num_examples': num_examples,
                            'model': model_name,
                            'run': run_idx,
                            'test_acc': None,
                            'test_p_y1': None,
                            'test_p_hat_y1': None
                        })
    
    return pd.DataFrame(arranged_data), idx

def create_summary(df):
    """创建汇总统计"""
    
    summary_data = []
    
    for num_ex in sorted(df['num_examples'].unique()):
        for model in sorted(df['model'].unique()):
            subset = df[(df['num_examples'] == num_ex) & (df['model'] == model)]
            
            # 过滤掉None值
            test_accs = subset['test_acc'].dropna()
            
            if len(test_accs) > 0:
                summary_data.append({
                    'num_examples': int(num_ex),
                    'model': model,
                    'num_runs': len(test_accs),
                    'acc_mean': float(test_accs.mean()),
                    'acc_std': float(test_accs.std()),
                    'acc_min': float(test_accs.min()),
                    'acc_max': float(test_accs.max())
                })
    
    return pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nohup_file', default='nohup.out', nargs='?')
    parser.add_argument('--output', default='test_results')
    args = parser.parse_args()
    
    print(f"提取 {args.nohup_file} 中的Testing结果...\n")
    
    # 提取所有Testing...后面的结果
    results = extract_testing_results(args.nohup_file)
    
    print(f"✓ 找到 {len(results)} 个Testing结果")
    
    # 按照实验顺序排列
    df, used_count = arrange_by_experiment_order(results)
    
    print(f"✓ 排列成 {len(df)} 个实验配置")
    print(f"  (实际使用了 {used_count} 个结果)\n")
    
    # 保存详细结果
    detailed_csv = f'{args.output}_detailed.csv'
    df.to_csv(detailed_csv, index=False)
    print(f"✓ 保存详细结果: {detailed_csv}")
    
    # 创建汇总
    summary_df = create_summary(df)
    summary_csv = f'{args.output}_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ 保存汇总结果: {summary_csv}")
    
    # 尝试保存Excel
    try:
        excel_file = f'{args.output}.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detailed', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"✓ 保存Excel: {excel_file}")
    except:
        pass
    
    # 打印结果
    print("\n" + "="*80)
    print("详细结果（按实验顺序）")
    print("="*80)
    
    for num_ex in sorted(df['num_examples'].unique()):
        print(f"\n{'='*80}")
        print(f"num_examples = {int(num_ex)}")
        print(f"{'='*80}")
        
        for model in ['Low-Rank', 'Standard']:
            subset = df[(df['num_examples'] == num_ex) & (df['model'] == model)]
            
            print(f"\n{model} Transformer:")
            for _, row in subset.iterrows():
                if pd.notna(row['test_acc']):
                    p_hat_str = f"{row['test_p_hat_y1']:.4f}" if pd.notna(row['test_p_hat_y1']) else "N/A"
                    print(f"  Run {row['run']}: Acc={row['test_acc']:.4f}, "
                          f"P(y=1)={row['test_p_y1']:.4f}, "
                          f"P(ŷ=1)={p_hat_str}")
                else:
                    print(f"  Run {row['run']}: [未完成]")
            
            # 计算平均值
            test_accs = subset['test_acc'].dropna()
            if len(test_accs) > 0:
                print(f"  平均: Acc={test_accs.mean():.4f}±{test_accs.std():.4f}")
    
    # 打印汇总对比表
    print("\n" + "="*80)
    print("汇总对比表")
    print("="*80)
    print(f"\n{'num_examples':<15} {'Low-Rank':<30} {'Standard':<30}")
    print("-" * 75)
    
    for num_ex in sorted(df['num_examples'].unique()):
        row_str = f"{int(num_ex):<15}"
        
        for model in ['Low-Rank', 'Standard']:
            subset = df[(df['num_examples'] == num_ex) & (df['model'] == model)]
            test_accs = subset['test_acc'].dropna()
            
            if len(test_accs) > 0:
                mean_acc = test_accs.mean()
                std_acc = test_accs.std()
                row_str += f"{mean_acc:.4f}±{std_acc:.4f} ({len(test_accs)})    "
            else:
                row_str += f"{'N/A':<30}"
        
        print(row_str)
    
    print("\n" + "="*80)
    print("✓ 完成!")
    print("="*80)

if __name__ == '__main__':
    main()

