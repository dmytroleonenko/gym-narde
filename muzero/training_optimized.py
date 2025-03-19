#!/usr/bin/env python3
"""
Optimized MuZero training script that uses pre-generated batches and MPS acceleration.
"""

import torch
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing
from muzero.models import MuZeroNetwork
from muzero.batch_collector import AsyncBatchCollector

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set or couldn't be set


def train_muzero_optimized(
    num_epochs=1000,
    batch_size=2048,
    buffer_size=8192,
    learning_rate=0.001,
    weight_decay=1e-4,
    checkpoint_path="muzero_model",
    log_dir="logs",
    num_parallel_envs=16,
    device_str=None,
    enable_profiling=False,
    save_interval=100,
):
    """
    Train MuZero with optimized batch processing for MPS acceleration.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Number of samples per batch (using large batches for MPS efficiency)
        buffer_size: Size of the experience buffer
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        checkpoint_path: Path to save model checkpoints
        log_dir: Directory for TensorBoard logs
        num_parallel_envs: Number of parallel environments for data collection
        device_str: Device to use for training ("cuda", "mps", or "cpu")
        enable_profiling: Whether to enable PyTorch profiling
        save_interval: How often to save model checkpoints (in epochs)
    """
    # Set up device
    if device_str is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for training")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS for training")
        else:
            device = torch.device("cpu")
            print("Using CPU for training (no GPU available)")
    else:
        device = torch.device(device_str)
        print(f"Using {device_str} for training")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create model
    collector = AsyncBatchCollector(
        batch_size=batch_size,
        buffer_size=buffer_size,
        num_parallel_envs=num_parallel_envs,
        device=device,
    )
    
    # Get input and action dimensions from collector
    input_dim = collector.collector.input_dim
    action_dim = collector.collector.action_dim
    
    # Create model
    model = MuZeroNetwork(input_dim, action_dim).to(device)
    # Convert model to bfloat16
    model = model.to(torch.bfloat16)
    
    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate / 10,
    )
    
    # Start asynchronous data collection
    collector.start_collection()
    
    # Wait for some initial data
    print("Waiting for initial data collection...")
    time.sleep(5)
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Get a batch
        observations, actions, rewards, next_observations, dones = collector.get_batch()
        
        # Enable profiling if requested
        if enable_profiling and epoch == 0:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                train_on_batch(model, optimizer, observations, actions, rewards, next_observations, dones)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            # Train on batch
            loss_components = train_on_batch(model, optimizer, observations, actions, rewards, next_observations, dones)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to tensorboard
        if loss_components:
            total_loss, value_loss, reward_loss, policy_loss = loss_components
            writer.add_scalar('Loss/total', total_loss, epoch)
            writer.add_scalar('Loss/value', value_loss, epoch)
            writer.add_scalar('Loss/reward', reward_loss, epoch)
            writer.add_scalar('Loss/policy', policy_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Calculate training time
        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('Time/epoch_seconds', epoch_duration, epoch)
        
        # Print progress
        if epoch % 10 == 0:
            if loss_components:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss:.4f}, Value: {value_loss:.4f}, "
                      f"Reward: {reward_loss:.4f}, Policy: {policy_loss:.4f}, "
                      f"Time: {epoch_duration:.2f}s, LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch}/{num_epochs}, Time: {epoch_duration:.2f}s, LR: {current_lr:.2e}")
        
        # Save model checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            save_path = f"{checkpoint_path}_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    save_path = f"{checkpoint_path}_final.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"Final model saved to {save_path}")
    
    # Close collector
    collector.close()
    writer.close()
    
    return model


def train_on_batch(model, optimizer, observations, actions, rewards, next_observations, dones):
    """Train model on a batch of data using large batch processing for MPS efficiency."""
    # Enable training mode
    model.train()
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Get batch size
    batch_size = observations.shape[0]
    
    # Ensure inputs are bfloat16
    observations = observations.to(torch.bfloat16)
    rewards = rewards.to(torch.bfloat16)
    
    # Get initial hidden state and predictions
    hidden, value_pred, policy_logits = model.initial_inference(observations)
    
    # Create one-hot encoding for actions
    action_one_hot = torch.zeros(batch_size, model.action_dim, device=actions.device, dtype=torch.bfloat16)
    for i in range(batch_size):
        action_one_hot[i, actions[i]] = 1.0
    
    # Get recurrent predictions
    next_hidden, reward_pred, next_value_pred, next_policy_logits = model.recurrent_inference(hidden, action_one_hot)
    
    # Calculate value loss
    value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(-1), rewards)
    next_value_loss = torch.nn.functional.mse_loss(next_value_pred.squeeze(-1), torch.zeros_like(rewards))
    combined_value_loss = (value_loss + next_value_loss) / 2.0
    
    # Calculate reward loss
    reward_loss = torch.nn.functional.mse_loss(reward_pred.squeeze(-1), rewards)
    
    # Calculate policy loss
    # For this example, we'll use a simple uniform policy target
    # In a real implementation, you would use the MCTS policy as the target
    uniform_policy = torch.ones_like(policy_logits) / model.action_dim
    policy_loss = torch.nn.functional.cross_entropy(policy_logits, uniform_policy)
    
    # Calculate total loss
    total_loss = combined_value_loss + reward_loss + policy_loss
    
    # Backpropagate
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Return loss components
    return total_loss.item(), combined_value_loss.item(), reward_loss.item(), policy_loss.item()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MuZero with optimized batch processing")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--buffer-size", type=int, default=8192, help="Size of experience buffer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--checkpoint", type=str, default="muzero_model", help="Checkpoint path")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--parallel-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, or cpu)")
    parser.add_argument("--profiling", action="store_true", help="Enable PyTorch profiling")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval in epochs")
    
    args = parser.parse_args()
    
    train_muzero_optimized(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
        num_parallel_envs=args.parallel_envs,
        device_str=args.device,
        enable_profiling=args.profiling,
        save_interval=args.save_interval,
    ) 