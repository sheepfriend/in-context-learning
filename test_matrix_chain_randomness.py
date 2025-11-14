#!/usr/bin/env python3
"""
Test that MatrixChain generates different A and B matrices on each evaluate call
when no seeds are provided.
"""

import sys
sys.path.append('src')

import torch
from samplers import MatrixChainSampler
from tasks import MatrixChain

def test_randomness():
    print("="*80)
    print("Testing MatrixChain Randomness (no seeds)")
    print("="*80)
    
    L = 2
    n = 3
    b_size = 2
    n_dims = 3 * n
    
    # Create sampler
    sampler = MatrixChainSampler(n_dims=n_dims, L=L, n=n, m=n)
    
    # Create task WITHOUT seeds (should generate fresh A, B on each evaluate)
    print(f"\n1. Creating MatrixChain task WITHOUT seeds...")
    task_no_seeds = MatrixChain(n_dims=n_dims, batch_size=b_size, L=L, n=n, m=n, p=n, q=n)
    print(f"   task.A_b is None: {task_no_seeds.A_b is None}")
    print(f"   task.B_b is None: {task_no_seeds.B_b is None}")
    
    # Call evaluate multiple times and check if results differ
    print(f"\n2. Calling evaluate() 3 times and checking if Z matrices differ...")
    xs = sampler.sample_xs(n_points=L*3*n, b_size=b_size)
    
    results = []
    for i in range(3):
        xs_assembled, ys = task_no_seeds.evaluate(xs)
        # Extract Z matrix from first block, first batch item
        z_start = 2 * n
        z_end = 3 * n
        Z = xs_assembled[0, z_start:z_end, 2*n:3*n]
        results.append(Z.clone())
        print(f"\n   Call {i+1} - Z matrix:")
        print(f"   {Z.numpy()}")
    
    # Check that results are different
    print(f"\n3. Verifying that results differ across calls...")
    diff_0_1 = (results[0] - results[1]).abs().max().item()
    diff_1_2 = (results[1] - results[2]).abs().max().item()
    diff_0_2 = (results[0] - results[2]).abs().max().item()
    
    print(f"   Max diff between call 1 and 2: {diff_0_1:.4f}")
    print(f"   Max diff between call 2 and 3: {diff_1_2:.4f}")
    print(f"   Max diff between call 1 and 3: {diff_0_2:.4f}")
    
    if diff_0_1 > 0.01 and diff_1_2 > 0.01 and diff_0_2 > 0.01:
        print(f"\n   ✓ Results differ significantly - randomness is working!")
    else:
        print(f"\n   ✗ Results are too similar - randomness may not be working!")
        return False
    
    # Now test with seeds - should be reproducible
    print(f"\n4. Creating MatrixChain task WITH seeds...")
    task_with_seeds = MatrixChain(n_dims=n_dims, batch_size=b_size, seeds=[42, 43], L=L, n=n, m=n, p=n, q=n)
    print(f"   task.A_b is None: {task_with_seeds.A_b is None}")
    
    print(f"\n5. Calling evaluate() 3 times with seeds...")
    results_seeded = []
    for i in range(3):
        xs_assembled, ys = task_with_seeds.evaluate(xs)
        # Extract Z matrix from first block, first batch item
        Z = xs_assembled[0, z_start:z_end, 2*n:3*n]
        results_seeded.append(Z.clone())
        print(f"\n   Call {i+1} - Z matrix:")
        print(f"   {Z.numpy()}")
    
    # Check that seeded results are identical
    print(f"\n6. Verifying that seeded results are identical...")
    diff_seed_0_1 = (results_seeded[0] - results_seeded[1]).abs().max().item()
    diff_seed_1_2 = (results_seeded[1] - results_seeded[2]).abs().max().item()
    
    print(f"   Max diff between call 1 and 2: {diff_seed_0_1:.2e}")
    print(f"   Max diff between call 2 and 3: {diff_seed_1_2:.2e}")
    
    if diff_seed_0_1 < 1e-6 and diff_seed_1_2 < 1e-6:
        print(f"\n   ✓ Seeded results are identical - reproducibility is working!")
    else:
        print(f"\n   ✗ Seeded results differ - reproducibility may not be working!")
        return False
    
    print(f"\n{'='*80}")
    print("✓ All randomness tests passed!")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_randomness()
    sys.exit(0 if success else 1)

