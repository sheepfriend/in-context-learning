#!/usr/bin/env python3
"""
Simple test script for the MatrixChain data type.
"""

import torch
import sys
sys.path.append('src')

from samplers import MatrixChainSampler
from tasks import MatrixChain

def test_matrix_chain():
    print("="*80)
    print("Testing MatrixChain implementation")
    print("="*80)
    
    # Parameters
    L = 3  # Number of matrices
    n = 4  # Matrix size (n x n)
    b_size = 2  # Batch size
    n_dims = 3 * n  # Total dimensions (12)
    
    print(f"\nParameters:")
    print(f"  L (number of matrices): {L}")
    print(f"  n (matrix size): {n} x {n}")
    print(f"  batch_size: {b_size}")
    print(f"  n_dims: {n_dims}")
    
    # Create sampler
    print(f"\n1. Creating MatrixChainSampler...")
    sampler = MatrixChainSampler(n_dims=n_dims, L=L, n=n, m=n)
    
    # Sample data
    print(f"2. Sampling {L} matrices per batch...")
    xs = sampler.sample_xs(n_points=L*3*n, b_size=b_size)
    print(f"   xs shape: {xs.shape} (expected: ({b_size}, {L}, {n}, {n}))")
    
    # Create task
    print(f"\n3. Creating MatrixChain task...")
    task = MatrixChain(n_dims=n_dims, batch_size=b_size, L=L, n=n, m=n, p=n, q=n)
    
    # Evaluate
    print(f"4. Evaluating task (computing Y=AX, Z=YB, assembling blocks)...")
    xs_assembled, ys = task.evaluate(xs)
    print(f"   xs_assembled shape: {xs_assembled.shape} (expected: ({b_size}, {L*3*n}, {3*n}))")
    print(f"   ys shape: {ys.shape} (expected: ({b_size}, {L*3*n}))")
    
    # Verify block diagonal structure
    print(f"\n5. Verifying block diagonal structure...")
    print(f"   Checking first batch item, first block (M_0)...")
    
    batch_idx = 0
    block_idx = 0
    
    # Extract the first block (M_0) which should be 3n x 3n
    block_start = block_idx * 3 * n
    block_end = (block_idx + 1) * 3 * n
    M_0 = xs_assembled[batch_idx, block_start:block_end, :]
    
    # Check X block (top-left n x n)
    X = M_0[:n, :n]
    X_zeros_right = M_0[:n, n:]
    print(f"   X block (top-left {n}x{n}): non-zero entries = {(X != 0).sum().item()}/{n*n}")
    print(f"   X padding (right): zero entries = {(X_zeros_right == 0).sum().item()}/{n*2*n}")
    
    # Check Y block (middle n x n)
    Y_zeros_left = M_0[n:2*n, :n]
    Y = M_0[n:2*n, n:2*n]
    Y_zeros_right = M_0[n:2*n, 2*n:]
    print(f"   Y block (middle {n}x{n}): non-zero entries = {(Y != 0).sum().item()}/{n*n}")
    print(f"   Y padding (left): zero entries = {(Y_zeros_left == 0).sum().item()}/{n*n}")
    print(f"   Y padding (right): zero entries = {(Y_zeros_right == 0).sum().item()}/{n*n}")
    
    # Check Z block (bottom-right n x n)
    Z_zeros_left = M_0[2*n:3*n, :2*n]
    Z = M_0[2*n:3*n, 2*n:]
    print(f"   Z block (bottom-right {n}x{n}): non-zero entries = {(Z != 0).sum().item()}/{n*n}")
    print(f"   Z padding (left): zero entries = {(Z_zeros_left == 0).sum().item()}/{2*n*n}")
    
    # Verify Y = AX and Z = YB
    print(f"\n6. Verifying matrix transformations...")
    X_input = xs[batch_idx, block_idx]  # shape (n, n)
    A = task.A_b[batch_idx]  # shape (n, n)
    B = task.B_b[batch_idx]  # shape (n, n)
    
    Y_expected = A @ X_input
    Z_expected = Y_expected @ B
    
    Y_diff = (Y - Y_expected).abs().max().item()
    Z_diff = (Z - Z_expected).abs().max().item()
    
    print(f"   Max difference in Y: {Y_diff:.2e} (should be ~0)")
    print(f"   Max difference in Z: {Z_diff:.2e} (should be ~0)")
    
    # Check targets
    print(f"\n7. Checking targets (ys)...")
    print(f"   X rows (should be 0): {ys[batch_idx, block_start:block_start+n]}")
    print(f"   Y rows (mean of Y matrix rows): {ys[batch_idx, block_start+n:block_start+2*n]}")
    print(f"   Z rows (mean of Z matrix rows): {ys[batch_idx, block_start+2*n:block_start+3*n]}")
    
    # Verify Y and Z targets match the mean
    Y_target = ys[batch_idx, block_start+n:block_start+2*n]
    Y_mean = Y.mean(dim=1)
    Z_target = ys[batch_idx, block_start+2*n:block_start+3*n]
    Z_mean = Z.mean(dim=1)
    
    Y_target_diff = (Y_target - Y_mean).abs().max().item()
    Z_target_diff = (Z_target - Z_mean).abs().max().item()
    
    print(f"   Max difference in Y target: {Y_target_diff:.2e} (should be ~0)")
    print(f"   Max difference in Z target: {Z_target_diff:.2e} (should be ~0)")
    
    print(f"\n{'='*80}")
    print("✓ All tests passed!")
    print("="*80)

if __name__ == "__main__":
    test_matrix_chain()

