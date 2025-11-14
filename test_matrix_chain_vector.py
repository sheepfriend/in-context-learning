#!/usr/bin/env python3
"""
Test the MatrixChainVector implementation.
"""

import sys
sys.path.append('src')

import torch
from samplers import MatrixChainVectorSampler
from tasks import MatrixChainVector

def test_matrix_chain_vector():
    print("="*80)
    print("Testing MatrixChainVector")
    print("="*80)
    
    # Parameters
    L = 3
    n = 4
    b_size = 2
    
    print(f"\nParameters:")
    print(f"  L={L}, n={n}")
    print(f"  batch_size={b_size}")
    print(f"  Expected rows per block: n^2 + 2n = {n*n + 2*n}")
    print(f"  Expected total rows: L * (n^2 + 2n) = {L * (n*n + 2*n)}")
    print(f"  n_dims (columns): {n}")
    
    # Create sampler
    print(f"\n1. Creating sampler...")
    n_dims = n  # Column dimension
    sampler = MatrixChainVectorSampler(n_dims=n_dims, L=L, n=n, m=n)
    
    # Sample data
    print(f"\n2. Sampling X matrices...")
    xs = sampler.sample_xs(n_points=L*(n*n+2*n), b_size=b_size, seeds=[42, 43])
    print(f"   xs shape: {xs.shape} (expected: [{b_size}, {L}, {n}, {n}])")
    
    # Create task
    print(f"\n3. Creating task...")
    task = MatrixChainVector(n_dims=n_dims, batch_size=b_size, seeds=[42, 43], L=L, n=n, m=n, p=n, q=n)
    
    # Evaluate
    print(f"\n4. Evaluating task...")
    xs_assembled, ys = task.evaluate(xs)
    
    print(f"   xs_assembled shape: {xs_assembled.shape}")
    print(f"   ys shape: {ys.shape}")
    print(f"   Expected shape: [{b_size}, {L*(n*n+2*n)}, {n}]")
    
    # Check the structure
    print(f"\n5. Checking structure...")
    rows_per_block = n*n + 2*n
    for block_idx in range(L):
        block_start = block_idx * rows_per_block
        
        # x part (column vector)
        x_start = block_start
        x_end = block_start + n*n
        x_part = xs_assembled[0, x_start:x_end, :]
        print(f"\n   Block {block_idx}:")
        print(f"   - x part shape: {x_part.shape} (rows {x_start}-{x_end})")
        print(f"   - x non-zero in column 0: {(x_part[:, 0] != 0).sum().item()} / {n*n}")
        print(f"   - x zero in other columns: {(x_part[:, 1:] == 0).all().item()}")
        
        # Y part
        y_start = block_start + n*n
        y_end = y_start + n
        y_part = xs_assembled[0, y_start:y_end, :]
        print(f"   - Y part shape: {y_part.shape} (rows {y_start}-{y_end})")
        print(f"   - Y norm: {y_part.norm().item():.4f}")
        
        # Z part
        z_start = y_end
        z_end = z_start + n
        z_part = xs_assembled[0, z_start:z_end, :]
        print(f"   - Z part shape: {z_part.shape} (rows {z_start}-{z_end})")
        print(f"   - Z norm: {z_part.norm().item():.4f}")
    
    # Verify transformations
    print(f"\n6. Verifying transformations...")
    A = task.A_b[0]
    B = task.B_b[0]
    
    for block_idx in [0]:  # Just check first block
        block_start = block_idx * rows_per_block
        
        # Get x and reconstruct X
        x_vec = xs_assembled[0, block_start:block_start+n*n, 0]
        X_reconstructed = x_vec.view(n, n).T  # Reshape back (column-wise flatten was used)
        
        # Get Y and Z
        y_start = block_start + n*n
        Y = xs_assembled[0, y_start:y_start+n, :n]
        
        z_start = y_start + n
        Z = xs_assembled[0, z_start:z_start+n, :n]
        
        # Check Y = AX (or XA)
        Y_check1 = A @ X_reconstructed
        Y_check2 = X_reconstructed @ A
        
        diff1 = (Y - Y_check1).norm().item()
        diff2 = (Y - Y_check2).norm().item()
        
        print(f"   Block {block_idx}:")
        print(f"   - Original X (from sampler) norm: {xs[0, block_idx].norm().item():.4f}")
        print(f"   - Reconstructed X norm: {X_reconstructed.norm().item():.4f}")
        print(f"   - Y norm: {Y.norm().item():.4f}")
        print(f"   - ||Y - AX||: {diff1:.6f}")
        print(f"   - ||Y - XA||: {diff2:.6f}")
        print(f"   - Y is likely from: {'AX' if diff1 < diff2 else 'XA'}")
        
        # Check Z = YB (or BY)
        Z_check1 = Y @ B
        Z_check2 = B @ Y
        
        diff1 = (Z - Z_check1).norm().item()
        diff2 = (Z - Z_check2).norm().item()
        
        print(f"   - Z norm: {Z.norm().item():.4f}")
        print(f"   - ||Z - YB||: {diff1:.6f}")
        print(f"   - ||Z - BY||: {diff2:.6f}")
        print(f"   - Z is likely from: {'YB' if diff1 < diff2 else 'BY'}")
    
    print(f"\n{'='*80}")
    print("✓ Test completed!")
    print("="*80)

if __name__ == "__main__":
    test_matrix_chain_vector()

