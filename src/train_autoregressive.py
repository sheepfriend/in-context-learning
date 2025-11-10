"""
Training script for autoregressive table connectivity path search.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from quinine import QuinineArgumentParser
from schema import schema
from models_autoregressive import build_autoregressive_model
from samplers_autoregressive import get_autoregressive_sampler
from tasks_autoregressive import get_autoregressive_task
from beam_search import evaluate_with_beam_search


def train_step(model, xs, ys, masks, optimizer, loss_func):
    """
    Single training step.
    
    Args:
        model: Autoregressive model
        xs: [batch, seq_len, n_dims] input
        ys: [batch, path_len] target tokens
        masks: [batch, path_len] valid position masks
        optimizer: Optimizer
        loss_func: Loss function (CrossEntropyLoss)
    
    Returns:
        loss: Scalar loss
        accuracy: Token-level accuracy
    """
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(xs, ys)  # [batch, path_len, vocab_size]
    
    # Reshape for loss computation
    batch_size, path_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)  # [batch*path_len, vocab_size]
    ys_flat = ys.reshape(-1)  # [batch*path_len]
    
    # Compute loss (CrossEntropyLoss ignores padding automatically)
    loss = loss_func(logits_flat, ys_flat)
    
    # Compute accuracy
    preds = torch.argmax(logits, dim=-1)  # [batch, path_len]
    correct = (preds == ys) & (masks.bool())
    accuracy = correct.sum().item() / masks.sum().item() if masks.sum() > 0 else 0
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item(), accuracy


def train(model, args):
    """Main training loop."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    
    # Loss function
    loss_func = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    
    # Data sampler
    sampler = get_autoregressive_sampler(
        args.training.data,
        n_dims=args.model.n_dims,
        **args.training.task_kwargs
    )
    
    # Training loop
    pbar = tqdm(range(args.training.train_steps))
    
    for step in pbar:
        # Sample batch
        xs, ys, labels, masks = sampler.sample_xs(
            n_points=args.model.n_positions,
            b_size=args.training.batch_size,
        )
        
        xs = xs.to(device)
        ys = ys.to(device)
        masks = masks.to(device)
        
        # Train step
        loss, accuracy = train_step(model, xs, ys, masks, optimizer, loss_func)
        
        # Update progress bar
        pbar.set_description(f"Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        # Log to wandb
        if step % args.wandb.log_every_steps == 0:
            wandb.log({
                "train/loss": loss,
                "train/accuracy": accuracy,
                "train/step": step,
            })
        
        # Save checkpoint
        if step > 0 and step % args.training.save_every_steps == 0:
            save_path = os.path.join(args.out_dir, f"model_step_{step}.pt")
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"\nSaved checkpoint to {save_path}")
    
    # Final save
    final_path = os.path.join(args.out_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")


def test(model, args):
    """Test with beam search."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print("\nTesting with Beam Search...")
    
    # Data sampler
    sampler = get_autoregressive_sampler(
        args.training.data,
        n_dims=args.model.n_dims,
        **args.training.task_kwargs
    )
    
    # Sample test batch
    test_batch_size = min(100, args.training.batch_size)
    xs, ys, labels, masks = sampler.sample_xs(
        n_points=args.model.n_positions,
        b_size=test_batch_size,
    )
    
    xs = xs.to(device)
    ys = ys.to(device)
    labels = labels.to(device)
    
    # Get column embeddings for beam search
    column_embeddings = sampler.column_embeddings.to(device)
    
    # Evaluate with beam search
    accuracy, exact_match = evaluate_with_beam_search(
        model=model,
        xs_batch=xs,
        ys_batch=ys,
        labels_batch=labels,
        column_embeddings=column_embeddings,
        beam_width=5,
        device=device
    )
    
    print(f"\nTest Results:")
    print(f"  Label Accuracy: {accuracy:.4f}")
    print(f"  Exact Match Rate: {exact_match:.4f}")
    
    # Log to wandb
    wandb.log({
        "test/label_accuracy": accuracy,
        "test/exact_match": exact_match,
    })
    
    return accuracy, exact_match


def main(args):
    """Main function."""
    
    # Initialize wandb (disabled mode)
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        name=args.wandb.name,
        notes=args.wandb.notes,
        config=vars(args),
        mode="disabled"  # Disable wandb
    )
    
    # Build model
    print("Building model...")
    model = build_autoregressive_model(args)
    print(f"Model: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nStarting training...")
    train(model, args)
    
    # Test
    print("\nStarting testing...")
    test(model, args)
    
    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    
    assert args.model.family == "autoregressive_gpt2", \
        "This script is for autoregressive_gpt2 models only"
    
    print(f"Running with: {args}")
    
    main(args)

