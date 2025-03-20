#!/usr/bin/env python3
"""
Benchmark PyTorch performance on NVIDIA T4 GPU vs CPU with different batch sizes.
Uses similar operations to the JAX benchmark for comparison.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

class SimpleMLPNetwork(torch.nn.Module):
    """A simple MLP network similar to the one in the JAX benchmark."""
    def __init__(self, input_dim=24, hidden_dim=128, output_dim=24*24):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 64)
        
        # Policy and value heads
        self.policy_head = torch.nn.Linear(64, output_dim)
        self.value_head = torch.nn.Linear(64, 1)
        
        # Activation functions
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        policy = self.softmax(self.policy_head(x))
        value = self.tanh(self.value_head(x))
        
        return policy, value

def matrix_op_pytorch(x, device):
    """Perform matrix operations similar to the JAX benchmark."""
    x = torch.abs(x)
    x = torch.matmul(x, x.transpose(-1, -2))
    x = torch.mean(x, dim=2)
    x = torch.nn.functional.relu(x)
    x = torch.sum(x, dim=1)
    return x

def matrix_op_numpy(x):
    """Perform matrix operations with NumPy."""
    x = np.abs(x)
    x = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
    x = np.mean(x, axis=2)
    x = np.maximum(0, x)  # ReLU
    x = np.sum(x, axis=1)
    return x

def benchmark_matrix_ops(batch_sizes, num_iterations=100):
    """Benchmark matrix operations on PyTorch vs NumPy."""
    print("\n=== Matrix Operations Benchmark (PyTorch vs NumPy) ===")
    
    cuda_times = []
    cpu_times = []
    numpy_times = []
    cuda_speedups = []  # CUDA vs CPU
    
    # Results for markdown report
    results = []
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_device = torch.device("cuda")
        print(f"Running on: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Warm up CUDA
    if cuda_available:
        x_cuda = torch.randn(1, 24, 24, device=cuda_device)
        _ = matrix_op_pytorch(x_cuda, cuda_device)
        torch.cuda.synchronize()
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random data
        x_np = np.random.randn(batch_size, 24, 24).astype(np.float32)
        x_cpu = torch.FloatTensor(x_np)
        
        if cuda_available:
            x_cuda = torch.FloatTensor(x_np).to(cuda_device)
        
        # Benchmark NumPy
        start = time.time()
        for _ in range(num_iterations):
            _ = matrix_op_numpy(x_np)
        numpy_time = (time.time() - start) / num_iterations
        print(f"NumPy: {numpy_time:.6f} seconds")
        
        # Benchmark PyTorch CPU
        start = time.time()
        for _ in range(num_iterations):
            _ = matrix_op_pytorch(x_cpu, "cpu")
        cpu_time = (time.time() - start) / num_iterations
        print(f"PyTorch CPU: {cpu_time:.6f} seconds")
        
        # Benchmark PyTorch CUDA
        if cuda_available:
            start = time.time()
            for _ in range(num_iterations):
                _ = matrix_op_pytorch(x_cuda, cuda_device)
                torch.cuda.synchronize()
            cuda_time = (time.time() - start) / num_iterations
            print(f"PyTorch CUDA: {cuda_time:.6f} seconds")
            
            # Calculate speedups
            cpu_speedup = numpy_time / cpu_time
            cuda_speedup = numpy_time / cuda_time
            print(f"CPU speedup over NumPy: {cpu_speedup:.2f}x")
            print(f"CUDA speedup over NumPy: {cuda_speedup:.2f}x")
            
            cuda_times.append(cuda_time)
            cuda_speedups.append(cuda_speedup)
        else:
            cuda_time = None
            cuda_speedup = None
            
        cpu_times.append(cpu_time)
        numpy_times.append(numpy_time)
        
        # Save results
        results.append({
            "batch_size": batch_size,
            "numpy_time": numpy_time,
            "cpu_time": cpu_time,
            "cuda_time": cuda_time,
            "cpu_speedup": numpy_time / cpu_time,
            "cuda_speedup": cuda_speedup
        })
    
    return results

def benchmark_neural_network(batch_sizes, num_iterations=10):
    """Benchmark neural network forward pass."""
    print("\n=== Neural Network Benchmark (PyTorch vs NumPy) ===")
    
    cuda_times = []
    cpu_times = []
    cuda_speedups = []
    
    # Results for markdown report
    results = []
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    # Create models
    cpu_model = SimpleMLPNetwork()
    if cuda_available:
        cuda_model = SimpleMLPNetwork().to("cuda")
        
    # Set to eval mode
    cpu_model.eval()
    if cuda_available:
        cuda_model.eval()
    
    # Warm up
    with torch.no_grad():
        x_cpu = torch.randn(1, 24)
        _ = cpu_model(x_cpu)
        
        if cuda_available:
            x_cuda = torch.randn(1, 24, device="cuda")
            _ = cuda_model(x_cuda)
            torch.cuda.synchronize()
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random data
        x_np = np.random.randn(batch_size, 24).astype(np.float32)
        x_cpu = torch.FloatTensor(x_np)
        
        if cuda_available:
            x_cuda = torch.FloatTensor(x_np).to("cuda")
        
        # Benchmark PyTorch CPU
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _, _ = cpu_model(x_cpu)
        cpu_time = (time.time() - start) / num_iterations
        print(f"PyTorch CPU: {cpu_time:.6f} seconds")
        
        # Benchmark PyTorch CUDA
        if cuda_available:
            start = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _, _ = cuda_model(x_cuda)
                    torch.cuda.synchronize()
            cuda_time = (time.time() - start) / num_iterations
            print(f"PyTorch CUDA: {cuda_time:.6f} seconds")
            
            # Calculate speedup
            cuda_speedup = cpu_time / cuda_time
            print(f"CUDA speedup over CPU: {cuda_speedup:.2f}x")
            
            cuda_times.append(cuda_time)
            cuda_speedups.append(cuda_speedup)
        else:
            cuda_time = None
            cuda_speedup = None
            
        cpu_times.append(cpu_time)
        
        # Save results
        results.append({
            "batch_size": batch_size,
            "cpu_time": cpu_time,
            "cuda_time": cuda_time,
            "cuda_speedup": cuda_speedup
        })
    
    return results

def generate_report(matrix_results, nn_results):
    """Generate a markdown report with the benchmark results."""
    report = "# PyTorch on NVIDIA T4 GPU Benchmark Report\n\n"
    
    # Test environment info
    report += "## Test Environment\n\n"
    report += f"- PyTorch Version: {torch.__version__}\n"
    report += f"- CUDA Available: {torch.cuda.is_available()}\n"
    if torch.cuda.is_available():
        report += f"- GPU: {torch.cuda.get_device_name(0)}\n"
        report += f"- CUDA Version: {torch.version.cuda}\n"
    report += "\n"
    
    # Matrix operations results
    report += "## Matrix Operations Benchmark\n\n"
    report += "| Batch Size | NumPy (seconds) | PyTorch CPU (seconds) | PyTorch CUDA (seconds) | CPU vs NumPy | CUDA vs NumPy |\n"
    report += "|------------|-----------------|------------------------|--------------------------|--------------|---------------|\n"
    
    for result in matrix_results:
        cuda_time = f"{result['cuda_time']:.6f}" if result['cuda_time'] is not None else "N/A"
        cuda_speedup = f"{result['cuda_speedup']:.2f}x" if result['cuda_speedup'] is not None else "N/A"
        
        report += f"| {result['batch_size']:<10} | {result['numpy_time']:.6f} | {result['cpu_time']:.6f} | {cuda_time} | {result['cpu_speedup']:.2f}x | {cuda_speedup} |\n"
    
    # Neural network results
    report += "\n## Neural Network Benchmark\n\n"
    report += "| Batch Size | PyTorch CPU (seconds) | PyTorch CUDA (seconds) | CUDA vs CPU |\n"
    report += "|------------|------------------------|--------------------------|------------|\n"
    
    for result in nn_results:
        cuda_time = f"{result['cuda_time']:.6f}" if result['cuda_time'] is not None else "N/A"
        cuda_speedup = f"{result['cuda_speedup']:.2f}x" if result['cuda_speedup'] is not None else "N/A"
        
        report += f"| {result['batch_size']:<10} | {result['cpu_time']:.6f} | {cuda_time} | {cuda_speedup} |\n"
    
    # Add analysis section comparing with JAX results
    report += "\n## Comparison with JAX (XLA) Results\n\n"
    report += "This benchmark evaluates PyTorch's performance on the NVIDIA T4 GPU compared to CPU, performing similar operations to those in the JAX benchmark. "
    report += "From these results, we can make the following observations:\n\n"
    
    # Find max speedups
    max_cuda_matrix_speedup = max([r["cuda_speedup"] for r in matrix_results if r["cuda_speedup"] is not None], default=0)
    max_cuda_nn_speedup = max([r["cuda_speedup"] for r in nn_results if r["cuda_speedup"] is not None], default=0)
    
    report += "1. For matrix operations, PyTorch with CUDA achieves a maximum speedup of "
    report += f"{max_cuda_matrix_speedup:.2f}x over NumPy on CPU\n"
    report += "2. For neural network operations, PyTorch with CUDA achieves a maximum speedup of "
    report += f"{max_cuda_nn_speedup:.2f}x over PyTorch on CPU\n"
    report += "3. PyTorch with CUDA shows excellent scaling with batch size, especially for larger batches\n"
    
    # Save the report
    with open("pytorch_t4_gpu_benchmark_report.md", "w") as f:
        f.write(report)
    print(f"Report saved to pytorch_t4_gpu_benchmark_report.md")

def plot_results(batch_sizes, matrix_results, nn_results):
    """Plot the benchmark results."""
    plt.figure(figsize=(15, 15))
    
    # Matrix operations plots
    plt.subplot(2, 2, 1)
    plt.plot(batch_sizes, [r["numpy_time"] for r in matrix_results], 'o-', label='NumPy')
    plt.plot(batch_sizes, [r["cpu_time"] for r in matrix_results], 'o-', label='PyTorch CPU')
    if all(r["cuda_time"] is not None for r in matrix_results):
        plt.plot(batch_sizes, [r["cuda_time"] for r in matrix_results], 'o-', label='PyTorch CUDA')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Operations Time vs Batch Size')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(batch_sizes, [r["cpu_speedup"] for r in matrix_results], 'o-', label='CPU vs NumPy')
    if all(r["cuda_speedup"] is not None for r in matrix_results):
        plt.plot(batch_sizes, [r["cuda_speedup"] for r in matrix_results], 'o-', label='CUDA vs NumPy')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup')
    plt.title('Matrix Operations Speedup')
    plt.xscale('log', base=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Neural network plots
    plt.subplot(2, 2, 3)
    plt.plot(batch_sizes, [r["cpu_time"] for r in nn_results], 'o-', label='PyTorch CPU')
    if all(r["cuda_time"] is not None for r in nn_results):
        plt.plot(batch_sizes, [r["cuda_time"] for r in nn_results], 'o-', label='PyTorch CUDA')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Neural Network Time vs Batch Size')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    if all(r["cuda_speedup"] is not None for r in nn_results):
        plt.plot(batch_sizes, [r["cuda_speedup"] for r in nn_results], 'o-', label='CUDA vs CPU')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup (CPU time / CUDA time)')
        plt.title('Neural Network Speedup (CUDA vs CPU)')
        plt.xscale('log', base=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('pytorch_t4_gpu_benchmark.png')
    print(f"Plot saved to pytorch_t4_gpu_benchmark.png")

def main():
    # Use the same batch sizes as the JAX benchmark
    batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]
    
    # Run matrix operations benchmark
    print("Running matrix operations benchmark...")
    matrix_results = benchmark_matrix_ops(batch_sizes)
    
    # Run neural network benchmark
    print("Running neural network benchmark...")
    nn_results = benchmark_neural_network(batch_sizes)
    
    # Generate report
    generate_report(matrix_results, nn_results)
    
    # Plot results
    plot_results(batch_sizes, matrix_results, nn_results)

if __name__ == "__main__":
    main() 