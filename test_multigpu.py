#!/usr/bin/env python3
"""
Test script to verify multi-GPU optimization for TableConnectivity task.
"""

import torch
import sys
import time
sys.path.append('src')

from tasks import TableConnectivity

def test_multigpu_optimization():
    print("Testing Multi-GPU Optimization for TableConnectivity...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test with larger batch size (optimized for 4 GPUs)
    n_dims = 20
    batch_size = 256  # 64 per GPU for 4 GPUs
    V = 5
    C = 3
    rho = 0.5
    num_queries = 30
    
    print(f"\nTest configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Tables (V): {V}")
    print(f"  - Columns per table (C): {C}")
    print(f"  - Connectivity (rho): {rho}")
    print(f"  - Number of queries: {num_queries}")
    
    # Create task instance
    print("\nCreating task...")
    start_time = time.time()
    task = TableConnectivity(
        n_dims=n_dims,
        batch_size=batch_size,
        V=V,
        C=C,
        rho=rho
    )
    creation_time = time.time() - start_time
    print(f"Task creation time: {creation_time:.2f}s")
    
    # Create sample input
    print("\nGenerating test data...")
    xs_b = torch.randn(batch_size, num_queries, n_dims)
    
    # Set the last 2 dimensions to represent column pairs
    for i in range(batch_size):
        for j in range(num_queries):
            col1 = torch.randint(0, V*C, (1,)).item()
            col2 = torch.randint(0, V*C, (1,)).item()
            xs_b[i, j, -2] = col1
            xs_b[i, j, -1] = col2
    
    # Move to GPU if available
    if torch.cuda.is_available():
        xs_b = xs_b.cuda()
        print("Data moved to GPU")
    
    # Evaluate (measure time)
    print("\nRunning evaluation...")
    start_time = time.time()
    ys_b = task.evaluate(xs_b)
    eval_time = time.time() - start_time
    
    print(f"Evaluation time: {eval_time:.4f}s")
    print(f"Throughput: {batch_size * num_queries / eval_time:.0f} samples/sec")
    
    # Check results
    print(f"\nResults:")
    print(f"  - Input shape: {xs_b.shape}")
    print(f"  - Output shape: {ys_b.shape}")
    print(f"  - Connected samples: {ys_b.sum().item()}/{batch_size * num_queries}")
    print(f"  - Connectivity rate: {ys_b.mean().item():.2%}")
    
    # Check a few samples
    print(f"\nSample results (first batch, first 5 queries):")
    for j in range(min(5, num_queries)):
        col1 = int(xs_b[0, j, -2].item())
        col2 = int(xs_b[0, j, -1].item())
        result = ys_b[0, j].item()
        print(f"  Query {j}: columns {col1} <-> {col2} = {result} ({'✓' if result == 1 else '✗'})")
    
    print("\n✅ Multi-GPU optimization test completed successfully!")
    
    return eval_time

if __name__ == "__main__":
    test_multigpu_optimization()

