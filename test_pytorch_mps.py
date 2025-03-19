#!/usr/bin/env python3
"""
Benchmark PyTorch with MPS (Metal Performance Shaders) acceleration.
"""

import torch
import numpy as np
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")

# Determine the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# CPU device for comparison
cpu_device = torch.device("cpu")

# Matrix multiplication benchmark
def run_matmul_benchmark(batch_sizes):
    """Benchmark matrix multiplication with PyTorch on CPU vs MPS."""
    print("\n=== Matrix Multiplication Benchmark ===")
    
    cpu_times = []
    mps_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random matrices
        a_np = np.random.random((batch_size, 128, 128)).astype(np.float32)
        b_np = np.random.random((batch_size, 128, 128)).astype(np.float32)
        
        # Convert to PyTorch tensors (CPU)
        a_cpu = torch.tensor(a_np, device=cpu_device)
        b_cpu = torch.tensor(b_np, device=cpu_device)
        
        # Convert to device tensors (MPS/CUDA/CPU)
        a_device = torch.tensor(a_np, device=device)
        b_device = torch.tensor(b_np, device=device)
        
        # Warm up
        _ = torch.matmul(a_device, b_device)
        if device.type in ["cuda", "mps"]:
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        
        # Benchmark CPU
        start_time = time.time()
        for _ in range(10):
            _ = torch.matmul(a_cpu, b_cpu)
        cpu_time = (time.time() - start_time) / 10
        print(f"PyTorch CPU matrix multiplication: {cpu_time:.6f} seconds")
        
        # Benchmark device (MPS/CUDA/CPU)
        start_time = time.time()
        for _ in range(10):
            _ = torch.matmul(a_device, b_device)
            if device.type in ["cuda", "mps"]:
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        device_time = (time.time() - start_time) / 10
        print(f"PyTorch {device.type.upper()} matrix multiplication: {device_time:.6f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / device_time if device_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        cpu_times.append(cpu_time)
        mps_times.append(device_time)
        
    return cpu_times, mps_times

# Neural network benchmark (similar to MuZero)
def run_neural_network_benchmark(batch_sizes):
    """Benchmark a simple neural network with PyTorch on CPU vs MPS."""
    print("\n=== Neural Network Benchmark ===")
    
    # Define a simple network (similar to what might be used in MuZero)
    class SimpleNetwork(torch.nn.Module):
        def __init__(self, input_dim=24, hidden_dim=128):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create models
    input_dim = 24  # Board size for Narde
    hidden_dim = 128
    
    cpu_model = SimpleNetwork(input_dim, hidden_dim).to(cpu_device)
    device_model = SimpleNetwork(input_dim, hidden_dim).to(device)
    
    cpu_times = []
    mps_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random input
        x_np = np.random.random((batch_size, input_dim)).astype(np.float32)
        
        # Convert to PyTorch tensors
        x_cpu = torch.tensor(x_np, device=cpu_device)
        x_device = torch.tensor(x_np, device=device)
        
        # Warm up
        _ = device_model(x_device)
        if device.type in ["cuda", "mps"]:
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        
        # Benchmark CPU
        start_time = time.time()
        for _ in range(100):
            _ = cpu_model(x_cpu)
        cpu_time = (time.time() - start_time) / 100
        print(f"PyTorch CPU neural network: {cpu_time:.6f} seconds")
        
        # Benchmark device (MPS/CUDA/CPU)
        start_time = time.time()
        for _ in range(100):
            _ = device_model(x_device)
            if device.type in ["cuda", "mps"]:
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        device_time = (time.time() - start_time) / 100
        print(f"PyTorch {device.type.upper()} neural network: {device_time:.6f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / device_time if device_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        cpu_times.append(cpu_time)
        mps_times.append(device_time)
        
    return cpu_times, mps_times

# Vector operations benchmark (similar to game logic)
def run_vector_ops_benchmark(batch_sizes):
    """Benchmark vector operations with PyTorch on CPU vs MPS."""
    print("\n=== Vector Operations Benchmark ===")
    
    cpu_times = []
    mps_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random vectors
        a_np = np.random.random((batch_size, 1024)).astype(np.float32) * 2 - 1  # -1 to 1
        
        # Convert to PyTorch tensors
        a_cpu = torch.tensor(a_np, device=cpu_device)
        a_device = torch.tensor(a_np, device=device)
        
        # Define operations (similar to game logic)
        def vector_ops(a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = torch.exp(a * 0.01)  # Scale to avoid overflow
            e = torch.tanh(a)
            return b + c + d + e
        
        # Warm up
        _ = vector_ops(a_device)
        if device.type in ["cuda", "mps"]:
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        
        # Benchmark CPU
        start_time = time.time()
        for _ in range(10):
            _ = vector_ops(a_cpu)
        cpu_time = (time.time() - start_time) / 10
        print(f"PyTorch CPU vector operations: {cpu_time:.6f} seconds")
        
        # Benchmark device (MPS/CUDA/CPU)
        start_time = time.time()
        for _ in range(10):
            _ = vector_ops(a_device)
            if device.type in ["cuda", "mps"]:
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        device_time = (time.time() - start_time) / 10
        print(f"PyTorch {device.type.upper()} vector operations: {device_time:.6f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / device_time if device_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        cpu_times.append(cpu_time)
        mps_times.append(device_time)
        
    return cpu_times, mps_times

def main():
    # Define batch sizes to test
    small_batch_sizes = [1, 8, 32, 128]
    large_batch_sizes = [512, 2048]
    extra_large_batch_sizes = [4096, 8192]
    batch_sizes = small_batch_sizes + large_batch_sizes + extra_large_batch_sizes
    
    # Run benchmarks
    cpu_matmul_times, mps_matmul_times = run_matmul_benchmark(batch_sizes)
    cpu_nn_times, mps_nn_times = run_neural_network_benchmark(batch_sizes)
    cpu_vector_times, mps_vector_times = run_vector_ops_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("\nMatrix Multiplication:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = cpu_matmul_times[i] / mps_matmul_times[i] if mps_matmul_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, CPU: {cpu_matmul_times[i]:.6f}s, {device.type.upper()}: {mps_matmul_times[i]:.6f}s, Speedup: {speedup:.2f}x")
    
    print("\nNeural Network:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = cpu_nn_times[i] / mps_nn_times[i] if mps_nn_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, CPU: {cpu_nn_times[i]:.6f}s, {device.type.upper()}: {mps_nn_times[i]:.6f}s, Speedup: {speedup:.2f}x")
    
    print("\nVector Operations:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = cpu_vector_times[i] / mps_vector_times[i] if mps_vector_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, CPU: {cpu_vector_times[i]:.6f}s, {device.type.upper()}: {mps_vector_times[i]:.6f}s, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 