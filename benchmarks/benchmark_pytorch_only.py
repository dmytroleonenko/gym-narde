#!/usr/bin/env python3
"""
Benchmark script to compare PyTorch performance on CPU vs MPS (Metal Performance Shaders).
"""

import time
import torch
import numpy as np
import gymnasium as gym
import argparse
import gym_narde  # Import to register the environment
from muzero.models import MuZeroNetwork
from muzero.training import train_muzero


def benchmark_forward_passes(num_iterations=1000, batch_size=32, device_type="cpu"):
    """
    Benchmark forward passes through the PyTorch model.
    """
    print(f"\n=== Benchmarking Forward Passes on {device_type.upper()} ===")
    
    # Create environment to get input dimensions
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    
    # Create random inputs
    observations = np.random.randn(batch_size, input_dim).astype(np.bfloat16)
    
    # Create PyTorch model
    torch_device = torch.device(device_type)
    torch_model = MuZeroNetwork(input_dim, action_dim).to(torch_device)
    torch_model = torch_model.to(torch.bfloat16)  # Convert model to bfloat16
    torch_observations = torch.FloatTensor(observations).to(torch_device).to(torch.bfloat16)
    
    # Warm-up
    print(f"Warming up with {50} iterations...")
    for _ in range(50):
        with torch.no_grad():
            torch_model.initial_inference(torch_observations)
    
    # Synchronize before timing
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    print(f"Running {num_iterations} iterations with batch size {batch_size}...")
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            result = torch_model.initial_inference(torch_observations)
    
    # Ensure computation is completed
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
        
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    # Print results
    print(f"PyTorch on {device_type.upper()}: {avg_time * 1000:.2f} ms per batch (batch size: {batch_size})")
    
    return avg_time


def benchmark_recurrent_inference(num_iterations=100, batch_size=32, device_type="cpu"):
    """
    Benchmark recurrent inference through the PyTorch model.
    """
    print(f"\n=== Benchmarking Recurrent Inference on {device_type.upper()} ===")
    
    # Create environment to get input dimensions
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    hidden_dim = 256
    
    # Create random inputs
    hidden_states = np.random.randn(batch_size, hidden_dim).astype(np.bfloat16)
    actions = np.random.randint(0, action_dim, size=batch_size)
    
    # Create PyTorch model
    torch_device = torch.device(device_type)
    torch_model = MuZeroNetwork(input_dim, action_dim).to(torch_device)
    torch_model = torch_model.to(torch.bfloat16)  # Convert model to bfloat16
    torch_hidden_states = torch.FloatTensor(hidden_states).to(torch_device).to(torch.bfloat16)
    torch_actions = torch.LongTensor(actions).to(torch_device)
    
    # Create batched action one-hot for PyTorch model
    action_onehots = torch.zeros(batch_size, action_dim, device=torch_device, dtype=torch.bfloat16)
    for i in range(batch_size):
        action_onehots[i, torch_actions[i]] = 1.0
    
    # Warm-up
    print(f"Warming up with {50} iterations...")
    for _ in range(50):
        with torch.no_grad():
            # Process all examples as a batch when possible
            for i in range(batch_size):
                torch_model.recurrent_inference(torch_hidden_states[i:i+1], action_onehots[i:i+1])
    
    # Synchronize before timing
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    print(f"Running {num_iterations} iterations with batch size {batch_size}...")
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            for i in range(batch_size):
                torch_model.recurrent_inference(torch_hidden_states[i:i+1], action_onehots[i:i+1])
    
    # Ensure computation is completed
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
        
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    # Print results
    print(f"PyTorch on {device_type.upper()}: {avg_time * 1000:.2f} ms per batch (batch size: {batch_size})")
    
    return avg_time


def benchmark_training(num_episodes=5, device_type="cpu"):
    """
    Benchmark a small training run using PyTorch.
    """
    print(f"\n=== Benchmarking Training on {device_type.upper()} ===")
    
    # Run PyTorch training
    start_time = time.time()
    train_muzero(
        num_episodes=num_episodes,
        replay_buffer_size=1000,
        batch_size=16,
        num_simulations=10,
        temperature_init=1.0,
        temperature_final=0.1,
        device_str=device_type,
        enable_profiling=False,
        save_interval=num_episodes
    )
    total_time = time.time() - start_time
    
    print(f"PyTorch training on {device_type.upper()}: {total_time:.2f} seconds for {num_episodes} episodes")
    
    return total_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch on CPU vs GPU/MPS")
    parser.add_argument("--forward-iterations", type=int, default=1000, 
                       help="Number of iterations for forward pass benchmark")
    parser.add_argument("--recurrent-iterations", type=int, default=100, 
                       help="Number of iterations for recurrent inference benchmark")
    parser.add_argument("--batch-size", type=int, default=128, 
                       help="Batch size for benchmarks")
    parser.add_argument("--train-episodes", type=int, default=5, 
                       help="Number of episodes for training benchmark")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip the training benchmark")
    parser.add_argument("--skip-cpu", action="store_true",
                       help="Skip CPU benchmarks and only run GPU/MPS tests")
    
    args = parser.parse_args()
    
    # Check available devices
    if torch.cuda.is_available():
        print("CUDA is available")
        gpu_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS is available")
        gpu_device = "mps"
    else:
        print("No GPU acceleration available, falling back to CPU")
        gpu_device = "cpu"
    
    # Run benchmarks on CPU if not skipped
    if not args.skip_cpu:
        cpu_forward_time = benchmark_forward_passes(
            num_iterations=args.forward_iterations, 
            batch_size=args.batch_size,
            device_type="cpu"
        )
        
        cpu_recurrent_time = benchmark_recurrent_inference(
            num_iterations=args.recurrent_iterations, 
            batch_size=args.batch_size,
            device_type="cpu"
        )
    else:
        cpu_forward_time = None
        cpu_recurrent_time = None
    
    # Run benchmarks on GPU/MPS if available
    if gpu_device != "cpu":
        gpu_forward_time = benchmark_forward_passes(
            num_iterations=args.forward_iterations, 
            batch_size=args.batch_size,
            device_type=gpu_device
        )
        
        gpu_recurrent_time = benchmark_recurrent_inference(
            num_iterations=args.recurrent_iterations, 
            batch_size=args.batch_size,
            device_type=gpu_device
        )
    
    # Run training benchmark if not skipped
    if not args.skip_training:
        cpu_train_time = benchmark_training(
            num_episodes=args.train_episodes,
            device_type="cpu"
        )
        
        if gpu_device != "cpu":
            gpu_train_time = benchmark_training(
                num_episodes=args.train_episodes,
                device_type=gpu_device
            )
    
    # Summary
    print("\n=== Benchmark Summary ===")
    if not args.skip_cpu:
        print(f"Forward Pass - CPU: {cpu_forward_time * 1000:.2f} ms per batch")
    
    if gpu_device != "cpu":
        print(f"Forward Pass - {gpu_device.upper()}: {gpu_forward_time * 1000:.2f} ms per batch")
        if not args.skip_cpu:
            print(f"Forward Pass Speedup: {cpu_forward_time / gpu_forward_time:.2f}x")
    
    if not args.skip_cpu:
        print(f"Recurrent Inference - CPU: {cpu_recurrent_time * 1000:.2f} ms per batch")
    
    if gpu_device != "cpu":
        print(f"Recurrent Inference - {gpu_device.upper()}: {gpu_recurrent_time * 1000:.2f} ms per batch")
        if not args.skip_cpu:
            print(f"Recurrent Inference Speedup: {cpu_recurrent_time / gpu_recurrent_time:.2f}x")
    
    if not args.skip_training:
        if not args.skip_cpu:
            print(f"Training - CPU: {cpu_train_time:.2f} seconds for {args.train_episodes} episodes")
        
        if gpu_device != "cpu":
            print(f"Training - {gpu_device.upper()}: {gpu_train_time:.2f} seconds for {args.train_episodes} episodes")
            if not args.skip_cpu:
                print(f"Training Speedup: {cpu_train_time / gpu_train_time:.2f}x")


if __name__ == "__main__":
    main() 