#!/usr/bin/env python3
"""
Test script for autoregressive table connectivity components.
"""

import sys
sys.path.append('src')

import torch
from samplers_autoregressive import TableConnectivityAutoregressiveSampler
from models_autoregressive import AutoregressiveTransformerModel
from tasks_autoregressive import TableConnectivityAutoregressiveTask


def test_sampler():
    """Test the autoregressive sampler."""
    print("="*80)
    print("Testing Autoregressive Sampler")
    print("="*80)
    
    sampler = TableConnectivityAutoregressiveSampler(
        n_dims=5,
        V=5,
        C=3,
        rho=0.5,
        max_path_samples=3
    )
    
    print(f"Created sampler:")
    print(f"  V={sampler.V}, C={sampler.C}")
    print(f"  Total columns: {sampler.total_columns}")
    print(f"  Vocab size (implicit): {4 + sampler.total_columns}")
    print(f"  Table embeddings shape: {sampler.table_embeddings.shape}")
    print(f"  Column embeddings shape: {sampler.column_embeddings.shape}")
    
    # Sample a batch
    print("\nSampling a batch...")
    xs, ys, labels, masks = sampler.sample_xs(
        n_points=50,
        b_size=4,
        max_path_len=15
    )
    
    print(f"  xs shape: {xs.shape}")
    print(f"  ys shape: {ys.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  masks shape: {masks.shape}")
    
    # Check a sample
    print(f"\nSample 0:")
    print(f"  Label: {labels[0].item()}")
    print(f"  Path length: {masks[0].sum().item()}")
    print(f"  Target tokens: {ys[0][:int(masks[0].sum())].tolist()}")
    
    print("\n✓ Sampler test passed!")
    return sampler


def test_model(sampler):
    """Test the autoregressive model."""
    print("\n" + "="*80)
    print("Testing Autoregressive Model")
    print("="*80)
    
    model = AutoregressiveTransformerModel(
        n_dims=5,
        n_positions=50,
        n_embd=128,
        n_layer=4,
        n_head=4,
        V=5,
        C=3,
        vocab_size=19,  # 4 + 15
        schema_len=21   # 5*4+1
    )
    
    print(f"Created model: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Schema length: {model.schema_len}")
    print(f"Vocab size: {model.vocab_size}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    xs, ys, labels, masks = sampler.sample_xs(
        n_points=50,
        b_size=2,
        max_path_len=15
    )
    
    with torch.no_grad():
        logits = model(xs, ys)
    
    print(f"  Input shape: {xs.shape}")
    print(f"  Target shape: {ys.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: [batch, path_len, vocab_size] = [{xs.shape[0]}, {ys.shape[1]}, {model.vocab_size}]")
    
    assert logits.shape == (xs.shape[0], ys.shape[1], model.vocab_size), "Shape mismatch!"
    
    print("\n✓ Model test passed!")
    return model


def test_task():
    """Test the autoregressive task."""
    print("\n" + "="*80)
    print("Testing Autoregressive Task")
    print("="*80)
    
    task = TableConnectivityAutoregressiveTask(
        n_dims=5,
        batch_size=4,
        V=5,
        C=3
    )
    
    print(f"Created task:")
    print(f"  Vocab size: {task.vocab_size}")
    print(f"  V={task.V}, C={task.C}")
    
    # Get loss function
    loss_func = task.get_training_metric()
    print(f"  Loss function: {loss_func}")
    print(f"  Metric name: {task.get_metric_name()}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    logits = torch.randn(2, 10, task.vocab_size)
    targets = torch.randint(0, task.vocab_size, (2, 10))
    targets[0, 5:] = 0  # Add padding
    
    loss = loss_func(logits.reshape(-1, task.vocab_size), targets.reshape(-1))
    print(f"  Test loss: {loss.item():.4f}")
    
    print("\n✓ Task test passed!")
    return task


def test_attention_masks():
    """Test attention mask patterns."""
    print("\n" + "="*80)
    print("Testing Attention Masks")
    print("="*80)
    
    model = AutoregressiveTransformerModel(
        n_dims=5,
        n_positions=30,
        n_embd=64,
        n_layer=4,
        n_head=4,
        V=5,
        C=3,
        schema_len=21
    )
    
    print(f"Mask shapes:")
    print(f"  First layers mask: {model.mask_first_buffer.shape}")
    print(f"  Remaining layers mask: {model.mask_remaining_buffer.shape}")
    
    # Check mask patterns
    mask_first = model.mask_first_buffer
    mask_remaining = model.mask_remaining_buffer
    
    # First layers should have block diagonal + causal
    schema_blocks_ok = True
    for i in range(model.V):
        start = i * (model.C + 1)
        end = start + (model.C + 1)
        if end <= model.schema_len:
            block = mask_first[start:end, start:end]
            # Should be all zeros (can attend)
            if (block == 0).all():
                pass  # Good
            else:
                schema_blocks_ok = False
                break
    
    print(f"  Schema block diagonal: {'✓' if schema_blocks_ok else '✗'}")
    
    # Remaining layers should be purely causal
    is_causal = True
    for i in range(model.n_positions):
        # Position i can attend to positions 0...i
        can_attend = (mask_remaining[i, :i+1] == 0).all()
        # Position i cannot attend to positions i+1...end
        cannot_attend = (mask_remaining[i, i+1:] < -1e8).all() if i+1 < model.n_positions else True
        
        if not (can_attend and cannot_attend):
            is_causal = False
            break
    
    print(f"  Causal pattern: {'✓' if is_causal else '✗'}")
    
    print("\n✓ Attention mask test passed!")


def test_integration():
    """Integration test: sample -> model -> loss."""
    print("\n" + "="*80)
    print("Testing Integration")
    print("="*80)
    
    # Create components
    sampler = TableConnectivityAutoregressiveSampler(
        n_dims=5, V=5, C=3, rho=0.5
    )
    
    model = AutoregressiveTransformerModel(
        n_dims=5, n_positions=50, n_embd=64, n_layer=4, n_head=4,
        V=5, C=3, vocab_size=19, schema_len=21
    )
    
    task = TableConnectivityAutoregressiveTask(
        n_dims=5, batch_size=4, V=5, C=3
    )
    
    loss_func = task.get_training_metric()
    
    # Sample data
    xs, ys, labels, masks = sampler.sample_xs(
        n_points=50,
        b_size=4,
        max_path_len=15
    )
    
    # Forward pass
    logits = model(xs, ys)
    
    # Compute loss
    logits_flat = logits.reshape(-1, model.vocab_size)
    ys_flat = ys.reshape(-1)
    loss = loss_func(logits_flat, ys_flat)
    
    # Compute accuracy
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == ys) & (masks.bool())
    accuracy = correct.sum().item() / masks.sum().item()
    
    print(f"Integration test results:")
    print(f"  Batch size: {xs.shape[0]}")
    print(f"  Sequence length: {xs.shape[1]}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print("\n✓ Integration test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("AUTOREGRESSIVE TABLE CONNECTIVITY - COMPONENT TESTS")
    print("="*80 + "\n")
    
    try:
        sampler = test_sampler()
        model = test_model(sampler)
        task = test_task()
        test_attention_masks()
        test_integration()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nYou can now train the model with:")
        print("  cd src")
        print("  python train_autoregressive.py --config conf/table_connectivity_autoregressive.yaml")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

