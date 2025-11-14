#!/usr/bin/env python3
"""
Example script showing how to train with MatrixChain task.

This demonstrates:
1. Creating a simple configuration
2. Running a short training loop
3. Checking the model's predictions
"""

import os
import sys
sys.path.append('src')

import torch
from samplers import MatrixChainSampler
from tasks import MatrixChain
from models import build_model

def create_simple_config():
    """Create a minimal configuration for testing."""
    config = {
        'model': {
            'family': 'gpt2',
            'n_dims': 12,  # 3 * n = 3 * 4
            'n_positions': 36,  # L * 3 * n = 3 * 3 * 4
            'n_embd': 128,
            'n_layer': 4,
            'n_head': 4,
        },
        'training': {
            'task': 'matrix_chain',
            'data': 'matrix_chain',
            'task_kwargs': {'L': 3, 'n': 4, 'm': 4, 'p': 4, 'q': 4},
            'num_tasks': None,
            'num_training_examples': None,
            'batch_size': 4,
            'learning_rate': 3e-4,
            'train_steps': 100,
            'save_every_steps': 1000,
            'keep_every_steps': -1,
            'resume_id': None,
            'curriculum': {
                'dims': {
                    'start': 12,
                    'end': 12,
                    'inc': 0,
                    'interval': 2000
                },
                'points': {
                    'start': 36,
                    'end': 36,
                    'inc': 0,
                    'interval': 2000
                }
            }
        },
        'wandb': {
            'project': 'in-context-training',
            'entity': 'test',
            'notes': 'Matrix chain example',
            'name': 'matrix_chain_example',
            'log_every_steps': 10
        },
        'out_dir': '../models/matrix_chain_example',
        'test_run': True  # Set to True to avoid wandb logging
    }
    return config

def simple_training_loop():
    """Run a simple training loop to demonstrate the task."""
    print("="*80)
    print("Matrix Chain Training Example")
    print("="*80)
    
    # Create configuration
    config = create_simple_config()
    
    # Extract parameters
    n_dims = config['model']['n_dims']
    n_positions = config['model']['n_positions']
    batch_size = config['training']['batch_size']
    task_kwargs = config['training']['task_kwargs']
    L = task_kwargs['L']
    n = task_kwargs['n']
    
    print(f"\nConfiguration:")
    print(f"  Model dims: {n_dims}")
    print(f"  Max positions: {n_positions}")
    print(f"  Batch size: {batch_size}")
    print(f"  Task: L={L}, n={n}×{n} matrices")
    
    # Create sampler and task
    print(f"\n1. Creating data sampler and task...")
    data_sampler = MatrixChainSampler(n_dims=n_dims, **task_kwargs)
    task_sampler = MatrixChain(n_dims=n_dims, batch_size=batch_size, **task_kwargs)
    
    # Sample data
    print(f"2. Sampling data...")
    xs = data_sampler.sample_xs(n_points=L*3*n, b_size=batch_size)
    xs_assembled, ys = task_sampler.evaluate(xs)
    
    print(f"   Input shape: {xs_assembled.shape}")
    print(f"   Target shape: {ys.shape}")
    
    # Build model
    print(f"\n3. Building model...")
    model_config = config['model']
    # Convert config dict to object with attributes for build_model
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    model_conf = ConfigObj(model_config)
    model = build_model(model_conf)
    print(f"   Model: {model_config['family']}")
    print(f"   Layers: {model_config['n_layer']}, Heads: {model_config['n_head']}")
    print(f"   Embedding dim: {model_config['n_embd']}")
    
    # Create optimizer
    print(f"\n4. Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_func = torch.nn.MSELoss()
    
    # Training loop
    print(f"\n5. Training for {config['training']['train_steps']} steps...")
    model.train()
    
    for step in range(config['training']['train_steps']):
        # Sample new data
        xs = data_sampler.sample_xs(n_points=L*3*n, b_size=batch_size)
        task = MatrixChain(n_dims=n_dims, batch_size=batch_size, **task_kwargs)
        xs_assembled, ys = task.evaluate(xs)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(xs_assembled, ys)
        
        # Compute loss only on last position (full embedding)
        # output[:, -1] shape: (batch_size, n_dims)
        # ys[:, -1] shape: (batch_size, n_dims)
        loss = loss_func(output[:, -1], ys[:, -1])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log progress
        if step % 10 == 0:
            print(f"   Step {step:3d}, Loss: {loss.item():.4f}")
    
    print(f"\n6. Final evaluation...")
    model.eval()
    with torch.no_grad():
        # Sample test data with a fixed seed for evaluation
        test_seed = 999
        xs = data_sampler.sample_xs(n_points=L*3*n, b_size=1)
        task = MatrixChain(n_dims=n_dims, batch_size=1, seeds=[test_seed], **task_kwargs)
        xs_assembled, ys = task.evaluate(xs)
        
        # Display A and B matrices (now available because we used seeds)
        A = task.A_b[0]  # shape (n, n)
        B = task.B_b[0]  # shape (n, n)
        print(f"   Transformation matrices A and B:")
        print(f"   A (shape {A.shape}): mean={A.mean().item():.3f}, std={A.std().item():.3f}")
        print(f"   A:\n{A.numpy()}")
        print(f"   B (shape {B.shape}): mean={B.mean().item():.3f}, std={B.std().item():.3f}")
        print(f"   B:\n{B.numpy()}")
        
        # Extract the complete Z matrix from the last M_i
        # Note: The actual order is random in training, here we compute all 4 possibilities
        last_block_idx = L - 1
        X_last = xs[0, last_block_idx]  # The input X matrix
        
        print(f"\n   Computing all possible transformations for last M_{last_block_idx}:")
        print(f"   Order 0: Y = AX, Z = YB")
        Y_0 = A @ X_last
        Z_0 = Y_0 @ B
        print(f"   Z_0:\n{Z_0.numpy()}\n")
        
        print(f"   Order 1: Y = AX, Z = BY")
        Y_1 = A @ X_last
        Z_1 = B @ Y_1
        print(f"   Z_1:\n{Z_1.numpy()}\n")
        
        print(f"   Order 2: Y = XA, Z = YB")
        Y_2 = X_last @ A
        Z_2 = Y_2 @ B
        print(f"   Z_2:\n{Z_2.numpy()}\n")
        
        print(f"   Order 3: Y = XA, Z = BY")
        Y_3 = X_last @ A
        Z_3 = B @ Y_3
        print(f"   Z_3:\n{Z_3.numpy()}")
        
        # Get prediction
        output = model(xs_assembled, ys)
        
        # Calculate MSE on the Z part of the last M_i
        # Last M_i: positions [(L-1)*3*n : L*3*n]
        # Z part of last M_i: positions [(L-1)*3*n + 2*n : L*3*n]
        last_z_start = (L-1) * 3 * n + 2 * n  # Start of Z in last block
        last_z_end = L * 3 * n  # End of last block
        
        # Extract predictions and targets for Z part of last M_i
        # Now ys contains full embeddings, shape (1, seq_len, n_dims)
        # Z part embeddings: positions [last_z_start:last_z_end] in the 2n:3n dimension range
        z_pred_embeddings = output[0, last_z_start:last_z_end]  # shape (n, 3*n)
        z_target_embeddings = ys[0, last_z_start:last_z_end]  # shape (n, 3*n)
        
        # Extract only the Z block part (columns 2n:3n)
        z_pred_block = z_pred_embeddings[:, 2*n:3*n]  # shape (n, n)
        z_target_block = z_target_embeddings[:, 2*n:3*n]  # shape (n, n)
        
        # Calculate MSE on the full Z matrix
        mse_full = ((z_pred_block - z_target_block) ** 2).mean().item()
        
        print(f"\n   Z target encoding: full embedding vectors (no aggregation)")
        print(f"\n   Z target matrix (from embeddings, the actual ground truth used):")
        print(f"{z_target_block.numpy()}")
        print(f"\n   Z predicted matrix:")
        print(f"{z_pred_block.numpy()}")
        print(f"\n   Average MSE on full Z matrix: {mse_full:.4f}")
        
        # Try to identify which order was used by comparing with all possibilities
        print(f"\n   Identifying which transformation order was used:")
        mses = []
        for i, Z in enumerate([Z_0, Z_1, Z_2, Z_3]):
            mse = ((z_target_block - Z) ** 2).mean().item()
            mses.append(mse)
            print(f"   Order {i} MSE: {mse:.6f}")
        
        best_order = torch.tensor(mses).argmin().item()
        print(f"   Best match: Order {best_order} (MSE ≈ {mses[best_order]:.2e})")
        
        # Also show per-element errors
        element_errors = ((z_pred_block - z_target_block) ** 2).numpy()
        print(f"\n   Per-element squared errors:")
        print(f"{element_errors}")
    
    print(f"\n{'='*80}")
    print("✓ Training example completed!")
    print("="*80)
    print(f"\nTo run full training with wandb logging:")
    print(f"  cd src")
    print(f"  python train.py --config conf/matrix_chain.yaml")

if __name__ == "__main__":
    simple_training_loop()

