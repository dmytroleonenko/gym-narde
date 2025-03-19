#!/usr/bin/env python3
"""
Benchmark script to compare PyTorch performance on CPU vs MPS with different batch sizes.
"""

import time
import torch
import numpy as np
import gymnasium as gym
import gym_narde  # Import to register the environment
from muzero.models import MuZeroNetwork
import matplotlib.pyplot as plt

def benchmark_batch_size(batch_size, num_iterations=500, device_type="cpu"):
    """
    Benchmark forward passes through the PyTorch model with a specific batch size.
    """
    # Create environment to get input dimensions
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    
    # Create random inputs
    observations = np.random.randn(batch_size, input_dim).astype(np.bfloat16)
    
    # Create PyTorch model
    torch_device = torch.device(device_type)
    torch_model = MuZeroNetwork(input_dim, action_dim).to(torch_device)
    torch_observations = torch.FloatTensor(observations).to(torch_device).to(torch.bfloat16)
    
    # Warm-up
    for _ in range(50):
        with torch.no_grad():
            torch_model.initial_inference(torch_observations)
    
    # Synchronize before timing
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    
    # Benchmark
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
    
    return avg_time


def main():
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
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    cpu_times = []
    gpu_times = []
    speedups = []
    
    print("\n=== Benchmarking Different Batch Sizes ===")
    print(f"{'Batch Size':<10} {'CPU (ms)':<10} {'MPS (ms)':<10} {'Speedup':<10}")
    print("-" * 40)
    
    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")
        
        # CPU benchmark
        cpu_time = benchmark_batch_size(batch_size, device_type="cpu")
        cpu_times.append(cpu_time * 1000)  # Convert to ms
        
        # GPU benchmark
        if gpu_device != "cpu":
            gpu_time = benchmark_batch_size(batch_size, device_type=gpu_device)
            gpu_times.append(gpu_time * 1000)  # Convert to ms
            speedup = cpu_time / gpu_time
            speedups.append(speedup)
        else:
            gpu_times.append(None)
            speedups.append(None)
        
        print(f"{batch_size:<10} {cpu_time * 1000:.2f} ms{' ' * 2} {gpu_time * 1000:.2f} ms{' ' * 2} {speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Times
    plt.subplot(2, 1, 1)
    plt.plot(batch_sizes, cpu_times, 'o-', label='CPU')
    if gpu_device != "cpu":
        plt.plot(batch_sizes, gpu_times, 'o-', label=gpu_device.upper())
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.title('Inference Time vs Batch Size')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Plot 2: Speedup
    if gpu_device != "cpu":
        plt.subplot(2, 1, 2)
        plt.plot(batch_sizes, speedups, 'o-', color='green')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup (CPU time / GPU time)')
        plt.title(f'Speedup ({gpu_device.upper()} vs CPU)')
        plt.xscale('log', base=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('batch_size_benchmark.png')
    print(f"Results saved to batch_size_benchmark.png")


if __name__ == "__main__":
    main() 