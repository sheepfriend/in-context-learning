#!/usr/bin/env python3
"""
Example: Using MatrixChainTransformer for Matrix Chain Task
This script demonstrates how to use the custom MatrixChainTransformer architecture.
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
from samplers import MatrixChainSampler
from tasks import MatrixChain
from models import MatrixChainTransformer

def mean_squared_error(output, target):
    """MSE loss function."""
    return ((output - target) ** 2).mean()

def main():
    print("="*80)
    print("MatrixChainTransformer Example")
    print("="*80)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    L = 3           # Number of M_i blocks
    n = 4           # Matrix size (each sub-matrix is n×n)
    n_dims = 12     # Input dimension (must be 3*n)
    n_embd = 128    # Embedding dimension
    n_head = 4      # Number of attention heads
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    
    print(f"\nConfiguration:")
    print(f"  Task: Matrix Chain (Y=AX, Z=YB)")
    print(f"  Architecture: Custom Transformer with 4 stages")
    print(f"  L={L} (number of blocks), n={n} (matrix size)")
    print(f"  n_dims={n_dims}, n_embd={n_embd}, n_head={n_head}")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    print(f"\n1. Creating MatrixChainTransformer...")
    model = MatrixChainTransformer(
        n_dims=n_dims,
        n_embd=n_embd,
        n_head=n_head,
        L=L,
        n=n
    )
    
    print(f"   Model name: {model.name}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n   Architecture overview:")
    print(f"   - Stage 1: Two 1-layer transformers (original + transposed)")
    print(f"   - Stage 2: Two 1-layer transformers (no positional encoding)")
    print(f"   - Stage 3: MLP for Y and Z prediction")
    
    # =========================================================================
    # Setup Training
    # =========================================================================
    print(f"\n2. Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sampler = MatrixChainSampler(n_dims=n_dims, L=L, n=n, m=n)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n3. Training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        # Generate training batch
        xs = sampler.sample_xs(n_points=L*3*n, b_size=batch_size)
        task = MatrixChain(
            n_dims=n_dims,
            batch_size=batch_size,
            seeds=None,  # Random A and B for each batch
            L=L, n=n, m=n, p=n, q=n
        )
        xs_assembled, ys = task.evaluate(xs)
        
        # Training step with custom procedure
        optimizer.zero_grad()
        
        # Calculate position indices
        last_block_start = (L - 1) * 3 * n
        y_start = last_block_start + n
        y_end = last_block_start + 2 * n
        z_start = last_block_start + 2 * n
        z_end = last_block_start + 3 * n
        
        # Step 1: Predict Y (with Y masked)
        xs_masked_y = xs_assembled.clone()
        xs_masked_y[:, y_start:y_end, n:2*n] = 0
        output_y = model(xs_masked_y, ys)
        y_pred = output_y[:, y_start:y_end, n:2*n]
        y_target = ys[:, y_start:y_end, n:2*n]
        y_loss = mean_squared_error(y_pred, y_target)
        
        # Step 2: Predict Z (with ground truth Y)
        xs_with_y = xs_assembled.clone()
        xs_with_y[:, z_start:z_end, 2*n:3*n] = 0
        output_z = model(xs_with_y, ys)
        z_pred = output_z[:, z_start:z_end, 2*n:3*n]
        z_target = ys[:, z_start:z_end, 2*n:3*n]
        z_loss = mean_squared_error(z_pred, z_target)
        
        # Combined loss
        loss = (y_loss + z_loss) / 2
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
                  f"Y_loss={y_loss.item():.4f}, Z_loss={z_loss.item():.4f}")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    print(f"\n4. Evaluating on test set...")
    model.eval()
    
    with torch.no_grad():
        # Fixed seed for reproducible evaluation
        test_seeds = [42 + i for i in range(8)]
        xs_test = sampler.sample_xs(n_points=L*3*n, b_size=8, seeds=test_seeds)
        task_test = MatrixChain(
            n_dims=n_dims,
            batch_size=8,
            seeds=test_seeds,
            L=L, n=n, m=n, p=n, q=n
        )
        xs_test_assembled, ys_test = task_test.evaluate(xs_test)
        
        # Get transformation matrices for analysis
        A = task_test.A_b[0]  # First batch item's A matrix
        B = task_test.B_b[0]  # First batch item's B matrix
        
        # Predict Y
        xs_test_masked_y = xs_test_assembled.clone()
        xs_test_masked_y[:, y_start:y_end, n:2*n] = 0
        output_test_y = model(xs_test_masked_y, ys_test)
        y_pred_test = output_test_y[0, y_start:y_end, n:2*n]  # First batch item
        y_target_test = ys_test[0, y_start:y_end, n:2*n]
        
        # Predict Z
        xs_test_with_y = xs_test_assembled.clone()
        xs_test_with_y[:, z_start:z_end, 2*n:3*n] = 0
        output_test_z = model(xs_test_with_y, ys_test)
        z_pred_test = output_test_z[0, z_start:z_end, 2*n:3*n]  # First batch item
        z_target_test = ys_test[0, z_start:z_end, 2*n:3*n]
        
        # Calculate errors
        y_mse = ((y_pred_test - y_target_test) ** 2).mean().item()
        z_mse = ((z_pred_test - z_target_test) ** 2).mean().item()
        
        print(f"\n   Test Results (first batch item):")
        print(f"   Y MSE: {y_mse:.4f}")
        print(f"   Z MSE: {z_mse:.4f}")
        print(f"   Total MSE: {(y_mse + z_mse) / 2:.4f}")
        
        # Show sample predictions
        print(f"\n   Sample Predictions:")
        print(f"   Y predicted (row 0): {y_pred_test[0]}")
        print(f"   Y target (row 0):    {y_target_test[0]}")
        print(f"   Y error (row 0):     {(y_pred_test[0] - y_target_test[0]).abs()}")
        
        print(f"\n   Z predicted (row 0): {z_pred_test[0]}")
        print(f"   Z target (row 0):    {z_target_test[0]}")
        print(f"   Z error (row 0):     {(z_pred_test[0] - z_target_test[0]).abs()}")
        
        # Show transformation matrices
        print(f"\n   Transformation matrices (first test item):")
        print(f"   A matrix norm: {A.norm().item():.4f}")
        print(f"   B matrix norm: {B.norm().item():.4f}")
        
        # Get X from last block for verification
        X_start = last_block_start
        X_end = last_block_start + n
        X_test = xs_test_assembled[0, X_start:X_end, :n]
        
        # Verify the transformation (depends on the random order)
        print(f"\n   Ground Truth Verification:")
        print(f"   X matrix (row 0): {X_test[0]}")
        print(f"   Y should be A@X or X@A")
        print(f"   Z should be Y@B or B@Y")
    
    print(f"\n{'='*80}")
    print("✓ Example completed!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. MatrixChainTransformer uses a 4-stage architecture")
    print("2. Training predicts Y first, then Z given true Y")
    print("3. Loss is only computed on the last M_i block")
    print("4. The model processes both original and transposed sequences")
    print("\nFor production training, use:")
    print("  cd src && python train.py --config conf/matrix_chain_custom.yaml")

if __name__ == "__main__":
    main()

