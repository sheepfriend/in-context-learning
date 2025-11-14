#!/usr/bin/env python3
"""
Simplified end-to-end test for MatrixChainTransformer.
"""

import sys
sys.path.append('src')

import torch
from samplers import MatrixChainSampler
from tasks import MatrixChain
from models import MatrixChainTransformer

def mean_squared_error(output, target):
    """MSE loss function."""
    return ((output - target) ** 2).mean()

def train_step_custom(model, xs, ys, optimizer, loss_func):
    """Custom training step for MatrixChainTransformer."""
    optimizer.zero_grad()
    
    batch_size = xs.shape[0]
    n = model.n
    L = model.L
    block_size = 3 * n
    last_block_start = (L - 1) * block_size
    
    # Step 1: Mask Y and Z, then predict Y
    # (Z should also be masked because Z depends on Y)
    xs_masked_y = xs.clone()
    y_start = last_block_start + n
    y_end = last_block_start + 2 * n
    z_start = last_block_start + 2 * n
    z_end = last_block_start + 3 * n
    xs_masked_y[:, y_start:y_end, n:2*n] = 0  # Mask Y
    xs_masked_y[:, z_start:z_end, 2*n:3*n] = 0  # Mask Z (because Z depends on Y)
    
    output_y = model(xs_masked_y, ys)
    y_pred = output_y[:, y_start:y_end, n:2*n]
    y_target = ys[:, y_start:y_end, n:2*n]
    y_loss = loss_func(y_pred, y_target)
    
    # Step 2: Use ground truth Y to predict Z
    xs_with_true_y = xs.clone()
    xs_with_true_y[:, z_start:z_end, 2*n:3*n] = 0  # Mask Z
    
    output_z = model(xs_with_true_y, ys)
    z_pred = output_z[:, z_start:z_end, 2*n:3*n]
    z_target = ys[:, z_start:z_end, 2*n:3*n]
    z_loss = loss_func(z_pred, z_target)
    
    # Total loss
    loss = (y_loss + z_loss) / 2
    
    loss.backward()
    optimizer.step()
    
    return loss.detach().item(), y_loss.detach().item(), z_loss.detach().item()

def main():
    print("="*80)
    print("Simplified E2E Test: MatrixChainTransformer")
    print("="*80)
    
    # Parameters matching config
    L = 3
    n = 4
    n_dims = 12
    n_embd = 128
    n_head = 4
    batch_size = 64
    learning_rate = 0.0003
    train_steps = 500
    
    print(f"\nConfiguration:")
    print(f"  L={L}, n={n}, n_dims={n_dims}")
    print(f"  n_embd={n_embd}, n_head={n_head}")
    print(f"  batch_size={batch_size}, lr={learning_rate}")
    print(f"  train_steps={train_steps}")
    
    # Create model
    print(f"\n1. Creating MatrixChainTransformer...")
    model = MatrixChainTransformer(
        n_dims=n_dims,
        n_embd=n_embd,
        n_head=n_head,
        L=L,
        n=n
    )
    print(f"   Model: {model.name}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sampler
    print(f"\n2. Creating data sampler...")
    sampler = MatrixChainSampler(n_dims=n_dims, L=L, n=n, m=n)
    
    # Setup training
    print(f"\n3. Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = mean_squared_error
    
    # Training loop
    print(f"\n4. Training for {train_steps} steps...")
    model.train()
    
    loss_history = []
    y_loss_history = []
    z_loss_history = []
    
    for step in range(train_steps):
        # Generate batch
        xs = sampler.sample_xs(n_points=L*3*n, b_size=batch_size)
        task = MatrixChain(n_dims=n_dims, batch_size=batch_size, seeds=None, L=L, n=n, m=n, p=n, q=n)
        xs_assembled, ys = task.evaluate(xs)
        
        # Train step
        loss, y_loss, z_loss = train_step_custom(model, xs_assembled, ys, optimizer, loss_func)
        
        loss_history.append(loss)
        y_loss_history.append(y_loss)
        z_loss_history.append(z_loss)
        
        # Print progress
        if (step + 1) % 50 == 0:
            avg_loss = sum(loss_history[-50:]) / 50
            avg_y_loss = sum(y_loss_history[-50:]) / 50
            avg_z_loss = sum(z_loss_history[-50:]) / 50
            print(f"   Step {step+1:4d}: Loss={avg_loss:.4f}, Y={avg_y_loss:.4f}, Z={avg_z_loss:.4f}")
    
    # Evaluation
    print(f"\n5. Evaluating on test set...")
    model.eval()
    
    test_losses = []
    test_y_losses = []
    test_z_losses = []
    
    with torch.no_grad():
        for _ in range(10):  # 10 test batches
            test_seeds = [1000 + i for i in range(batch_size)]
            xs_test = sampler.sample_xs(n_points=L*3*n, b_size=batch_size, seeds=test_seeds)
            task_test = MatrixChain(n_dims=n_dims, batch_size=batch_size, seeds=test_seeds, L=L, n=n, m=n, p=n, q=n)
            xs_test_assembled, ys_test = task_test.evaluate(xs_test)
            
            last_block_start = (L - 1) * 3 * n
            y_start = last_block_start + n
            y_end = last_block_start + 2 * n
            z_start = last_block_start + 2 * n
            z_end = last_block_start + 3 * n
            
            # Y prediction (with Y and Z masked)
            xs_test_masked_y = xs_test_assembled.clone()
            xs_test_masked_y[:, y_start:y_end, n:2*n] = 0  # Mask Y
            xs_test_masked_y[:, z_start:z_end, 2*n:3*n] = 0  # Mask Z
            output_test_y = model(xs_test_masked_y, ys_test)
            y_pred_test = output_test_y[:, y_start:y_end, n:2*n]
            y_target_test = ys_test[:, y_start:y_end, n:2*n]
            y_mse = ((y_pred_test - y_target_test) ** 2).mean().item()
            
            # Z prediction
            xs_test_with_y = xs_test_assembled.clone()
            xs_test_with_y[:, z_start:z_end, 2*n:3*n] = 0
            output_test_z = model(xs_test_with_y, ys_test)
            z_pred_test = output_test_z[:, z_start:z_end, 2*n:3*n]
            z_target_test = ys_test[:, z_start:z_end, 2*n:3*n]
            z_mse = ((z_pred_test - z_target_test) ** 2).mean().item()
            
            test_losses.append((y_mse + z_mse) / 2)
            test_y_losses.append(y_mse)
            test_z_losses.append(z_mse)
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_y_loss = sum(test_y_losses) / len(test_y_losses)
    avg_test_z_loss = sum(test_z_losses) / len(test_z_losses)
    
    print(f"   Average Test Loss: {avg_test_loss:.4f}")
    print(f"   Average Test Y Loss: {avg_test_y_loss:.4f}")
    print(f"   Average Test Z Loss: {avg_test_z_loss:.4f}")
    
    # Show sample predictions
    print(f"\n6. Sample predictions (first batch item):")
    print(f"   Y pred (row 0): {y_pred_test[0, 0]}")
    print(f"   Y target (row 0): {y_target_test[0, 0]}")
    print(f"   Z pred (row 0): {z_pred_test[0, 0]}")
    print(f"   Z target (row 0): {z_target_test[0, 0]}")
    
    # Training curve summary
    print(f"\n7. Training Summary:")
    print(f"   Initial loss (avg first 50): {sum(loss_history[:50])/50:.4f}")
    print(f"   Final loss (avg last 50): {sum(loss_history[-50:])/50:.4f}")
    print(f"   Best loss: {min(loss_history):.4f}")
    print(f"   Improvement: {(sum(loss_history[:50])/50 - sum(loss_history[-50:])/50):.4f}")
    
    print(f"\n{'='*80}")
    print("✓ End-to-end test completed successfully!")
    print("="*80)
    print("\nThe MatrixChainTransformer is ready to use!")
    print("\nTo train with full configuration, run:")
    print("  cd src && python train.py --config conf/matrix_chain_custom.yaml")

if __name__ == "__main__":
    main()

