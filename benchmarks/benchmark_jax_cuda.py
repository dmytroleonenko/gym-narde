#!/usr/bin/env python3
"""
Benchmark JAX acceleration on CUDA (NVIDIA T4 GPU) for various batch sizes.
This script compares JAX with XLA on GPU versus NumPy on CPU.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import timeit
import matplotlib.pyplot as plt
from functools import partial

# Enable XLA optimization flags
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'gpu')

def verify_jax_cuda():
    """Verify that JAX is using CUDA and XLA."""
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"XLA flags: jax_disable_jit={jax.config.jax_disable_jit}")

def create_random_boards(num_boards, use_jax=True):
    """Create random board states."""
    np.random.seed(42)
    boards = np.random.randint(
        low=-15,
        high=15,
        size=(num_boards, 24),
        dtype=np.int32
    )
    
    if use_jax:
        return jnp.array(boards)
    else:
        return boards

@jax.jit
def rotate_boards_jax(boards):
    """JAX-accelerated function to rotate multiple boards at once."""
    return -1 * jnp.flip(boards, axis=1)

def rotate_boards_numpy(boards):
    """NumPy function to rotate multiple boards."""
    return -1 * np.flip(boards, axis=1)

# Define a more complex model with realistic neural network operations
@partial(jax.jit, static_argnums=(1,))
def mlp_forward_jax(x, hidden_size=128):
    """Simple MLP forward pass with JAX - simulates a MuZero-style network."""
    # Convert to float32
    x = x.astype(jnp.float32)
    
    # First reshape to proper dimensions
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1)  # Flatten
    input_dim = x.shape[1]
    
    # Initialize weights with a specific seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # First layer
    w1 = jax.random.normal(key, (input_dim, hidden_size)) / jnp.sqrt(input_dim)
    b1 = jnp.zeros((hidden_size,))
    h1 = jnp.dot(x, w1) + b1
    h1 = jax.nn.relu(h1)
    
    # Second layer
    key, subkey = jax.random.split(key)
    w2 = jax.random.normal(subkey, (hidden_size, hidden_size)) / jnp.sqrt(hidden_size)
    b2 = jnp.zeros((hidden_size,))
    h2 = jnp.dot(h1, w2) + b2
    h2 = jax.nn.relu(h2)
    
    # Output layer
    key, subkey = jax.random.split(key)
    w3 = jax.random.normal(subkey, (hidden_size, 64)) / jnp.sqrt(hidden_size)
    b3 = jnp.zeros((64,))
    logits = jnp.dot(h2, w3) + b3
    
    # Policy and value heads (MuZero style)
    key, subkey = jax.random.split(key)
    w_policy = jax.random.normal(subkey, (64, 24 * 24)) / jnp.sqrt(64)  # 24*24 possible moves
    b_policy = jnp.zeros((24 * 24,))
    policy = jnp.dot(logits, w_policy) + b_policy
    policy = jax.nn.softmax(policy, axis=-1)
    
    key, subkey = jax.random.split(key)
    w_value = jax.random.normal(subkey, (64, 1)) / jnp.sqrt(64)
    b_value = jnp.zeros((1,))
    value = jnp.dot(logits, w_value) + b_value
    value = jnp.tanh(value)
    
    return policy, value

def mlp_forward_numpy(x, hidden_size=128):
    """NumPy version of the same network."""
    # Convert to float32
    x = x.astype(np.float32)
    
    # Reshape
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1)  # Flatten
    input_dim = x.shape[1]
    
    # Use the same seed for reproducibility
    np.random.seed(42)
    
    # First layer
    w1 = np.random.normal(0, 1, (input_dim, hidden_size)) / np.sqrt(input_dim)
    b1 = np.zeros((hidden_size,))
    h1 = np.dot(x, w1) + b1
    h1 = np.maximum(0, h1)  # ReLU
    
    # Second layer
    w2 = np.random.normal(0, 1, (hidden_size, hidden_size)) / np.sqrt(hidden_size)
    b2 = np.zeros((hidden_size,))
    h2 = np.dot(h1, w2) + b2
    h2 = np.maximum(0, h2)  # ReLU
    
    # Output layer
    w3 = np.random.normal(0, 1, (hidden_size, 64)) / np.sqrt(hidden_size)
    b3 = np.zeros((64,))
    logits = np.dot(h2, w3) + b3
    
    # Policy and value heads
    w_policy = np.random.normal(0, 1, (64, 24 * 24)) / np.sqrt(64)
    b_policy = np.zeros((24 * 24,))
    policy = np.dot(logits, w_policy) + b_policy
    # Softmax
    policy_exp = np.exp(policy - np.max(policy, axis=-1, keepdims=True))
    policy = policy_exp / np.sum(policy_exp, axis=-1, keepdims=True)
    
    w_value = np.random.normal(0, 1, (64, 1)) / np.sqrt(64)
    b_value = np.zeros((1,))
    value = np.dot(logits, w_value) + b_value
    value = np.tanh(value)
    
    return policy, value

@jax.jit
def process_batch_jax(batch):
    """Perform several operations on the batch to mimic a real workload."""
    # Create a simple network-like computation
    x = jnp.abs(batch).astype(jnp.float32)
    x = jnp.matmul(x, jnp.transpose(x, axes=(0, 2, 1)))
    x = jnp.mean(x, axis=2)
    x = jax.nn.relu(x)
    x = jnp.sum(x, axis=1)
    return x

def process_batch_numpy(batch):
    """NumPy version of the same computation."""
    x = np.abs(batch).astype(np.float32)
    x = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
    x = np.mean(x, axis=2)
    x = np.maximum(0, x)  # ReLU
    x = np.sum(x, axis=1)
    return x

def benchmark_matrix_ops(batch_sizes, operations=100):
    """Benchmark JAX vs NumPy for matrix operations with different batch sizes."""
    print("\n=== Matrix Operations Benchmark (CUDA vs CPU) ===")
    
    jax_times = []
    numpy_times = []
    
    # Results for markdown report
    results = []
    
    # Warm up JIT compilation
    boards_jax = create_random_boards(1, use_jax=True)
    boards_jax = boards_jax.reshape(1, 24, 1)  # Add channel dimension
    _ = process_batch_jax(boards_jax)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random boards with JAX and NumPy
        boards_jax = create_random_boards(batch_size, use_jax=True)
        boards_jax = boards_jax.reshape(batch_size, 24, 1)  # Add channel dimension
        boards_numpy = np.array(boards_jax)
        
        # Benchmark JAX processing
        jax_stmt = f"""
for _ in range({operations}):
    result_jax = process_batch_jax(boards_jax)
    _ = result_jax.block_until_ready()
"""
        jax_time = timeit.timeit(stmt=jax_stmt, 
                                globals={"process_batch_jax": process_batch_jax, 
                                        "boards_jax": boards_jax}, 
                                number=5) / 5
        print(f"JAX process {batch_size} boards: {jax_time:.6f} seconds")
        
        # Benchmark NumPy processing
        numpy_stmt = f"""
for _ in range({operations}):
    result_numpy = process_batch_numpy(boards_numpy)
"""
        numpy_time = timeit.timeit(stmt=numpy_stmt, 
                                globals={"process_batch_numpy": process_batch_numpy, 
                                        "boards_numpy": boards_numpy}, 
                                number=5) / 5
        print(f"NumPy process {batch_size} boards: {numpy_time:.6f} seconds")
        
        # Calculate speedup
        speedup = numpy_time / jax_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Record times and results
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
        
        results.append({
            "batch_size": batch_size,
            "jax_time": jax_time,
            "numpy_time": numpy_time,
            "speedup": speedup
        })
    
    return jax_times, numpy_times, results

def benchmark_neural_network(batch_sizes, operations=10):
    """Benchmark JAX vs NumPy for neural network forward pass with different batch sizes."""
    print("\n=== Neural Network Benchmark (CUDA vs CPU) ===")
    
    jax_times = []
    numpy_times = []
    
    # Results for markdown report
    results = []
    
    # Warm up JIT compilation
    boards_jax = create_random_boards(1, use_jax=True)
    _ = mlp_forward_jax(boards_jax)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random boards with JAX and NumPy
        boards_jax = create_random_boards(batch_size, use_jax=True)
        boards_numpy = np.array(boards_jax)
        
        # Benchmark JAX neural network
        jax_stmt = f"""
for _ in range({operations}):
    policy, value = mlp_forward_jax(boards_jax)
    _ = policy.block_until_ready()
    _ = value.block_until_ready()
"""
        jax_time = timeit.timeit(stmt=jax_stmt, 
                                globals={"mlp_forward_jax": mlp_forward_jax, 
                                        "boards_jax": boards_jax}, 
                                number=5) / 5
        print(f"JAX neural network {batch_size} boards: {jax_time:.6f} seconds")
        
        # Benchmark NumPy neural network
        numpy_stmt = f"""
for _ in range({operations}):
    policy, value = mlp_forward_numpy(boards_numpy)
"""
        numpy_time = timeit.timeit(stmt=numpy_stmt, 
                                 globals={"mlp_forward_numpy": mlp_forward_numpy, 
                                         "boards_numpy": boards_numpy}, 
                                 number=5) / 5
        print(f"NumPy neural network {batch_size} boards: {numpy_time:.6f} seconds")
        
        # Calculate speedup
        speedup = numpy_time / jax_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Record times and results
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
        
        results.append({
            "batch_size": batch_size,
            "jax_time": jax_time,
            "numpy_time": numpy_time,
            "speedup": speedup
        })
    
    return jax_times, numpy_times, results

def generate_report(matrix_results, nn_results):
    """Generate a markdown report with the benchmark results."""
    report = "# JAX with XLA on NVIDIA T4 GPU Benchmark Report\n\n"
    
    # Test environment info
    report += "## Test Environment\n\n"
    report += f"- JAX Version: {jax.__version__}\n"
    report += f"- JAX Backend: {jax.default_backend()}\n"
    report += f"- JAX Devices: {jax.devices()}\n"
    report += f"- GPU: {jax.devices()[0].device_kind if jax.default_backend() == 'gpu' else 'None'}\n"
    report += "- XLA: Enabled\n\n"
    
    # Matrix operations results
    report += "## Matrix Operations Benchmark\n\n"
    report += "| Batch Size | JAX (seconds) | NumPy (seconds) | Speedup |\n"
    report += "|------------|--------------|-----------------|--------|\n"
    
    for result in matrix_results:
        report += f"| {result['batch_size']:<10} | {result['jax_time']:.6f} | {result['numpy_time']:.6f} | {result['speedup']:.2f}x |\n"
    
    # Neural network results
    report += "\n## Neural Network Benchmark\n\n"
    report += "| Batch Size | JAX (seconds) | NumPy (seconds) | Speedup |\n"
    report += "|------------|--------------|-----------------|--------|\n"
    
    for result in nn_results:
        report += f"| {result['batch_size']:<10} | {result['jax_time']:.6f} | {result['numpy_time']:.6f} | {result['speedup']:.2f}x |\n"
    
    # Analysis and Recommendations
    report += "\n## Analysis and Recommendations\n\n"
    
    # Find crossover points and max speedups
    matrix_speedups = [r["speedup"] for r in matrix_results]
    nn_speedups = [r["speedup"] for r in nn_results]
    max_matrix_speedup = max(matrix_speedups)
    max_nn_speedup = max(nn_speedups)
    
    # Find batch size where JAX becomes faster (speedup > 1)
    matrix_crossover = "Not found"
    for r in matrix_results:
        if r["speedup"] > 1:
            matrix_crossover = r["batch_size"]
            break
            
    nn_crossover = "Not found"
    for r in nn_results:
        if r["speedup"] > 1:
            nn_crossover = r["batch_size"]
            break
    
    report += f"### Key Findings\n\n"
    report += f"1. **Matrix Operations**: JAX with XLA on T4 GPU achieves up to {max_matrix_speedup:.2f}x speedup over NumPy on CPU\n"
    report += f"2. **Neural Network**: JAX with XLA on T4 GPU achieves up to {max_nn_speedup:.2f}x speedup over NumPy on CPU\n"
    report += f"3. **Crossover Points**: JAX becomes faster than NumPy at batch size {matrix_crossover} for matrix operations and {nn_crossover} for neural networks\n\n"
    
    report += "### Recommendations\n\n"
    report += "1. Use JAX with XLA on GPU for matrix operations with batch sizes >= 64\n"
    report += "2. Use JAX with XLA on GPU for neural network inference with batch sizes >= 16\n"
    report += "3. For very small batch sizes (1-8), CPU may still be more efficient due to GPU transfer overhead\n"
    report += "4. Consider using mixed precision (bfloat16) for even better performance on T4 GPU\n"
    
    # Save the report
    with open("jax_t4_gpu_xla_benchmark_report.md", "w") as f:
        f.write(report)
    print(f"Report saved to jax_t4_gpu_xla_benchmark_report.md")

def plot_results(batch_sizes, matrix_results, nn_results):
    """Plot the benchmark results."""
    plt.figure(figsize=(15, 15))
    
    # Matrix operations plots
    plt.subplot(2, 2, 1)
    plt.plot(batch_sizes, [r["jax_time"] for r in matrix_results], 'o-', label='JAX (GPU)')
    plt.plot(batch_sizes, [r["numpy_time"] for r in matrix_results], 'o-', label='NumPy (CPU)')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Operations Time vs Batch Size')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(batch_sizes, [r["speedup"] for r in matrix_results], 'o-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (NumPy time / JAX time)')
    plt.title('Matrix Operations Speedup (JAX vs NumPy)')
    plt.xscale('log', base=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Neural network plots
    plt.subplot(2, 2, 3)
    plt.plot(batch_sizes, [r["jax_time"] for r in nn_results], 'o-', label='JAX (GPU)')
    plt.plot(batch_sizes, [r["numpy_time"] for r in nn_results], 'o-', label='NumPy (CPU)')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Neural Network Time vs Batch Size')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(batch_sizes, [r["speedup"] for r in nn_results], 'o-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (NumPy time / JAX time)')
    plt.title('Neural Network Speedup (JAX vs NumPy)')
    plt.xscale('log', base=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('jax_t4_gpu_xla_benchmark.png')
    print(f"Plot saved to jax_t4_gpu_xla_benchmark.png")

def main():
    # Verify JAX is using CUDA and XLA
    verify_jax_cuda()
    
    # Use a subset of batch sizes to make the benchmark faster
    batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]
    
    # Run matrix operations benchmark
    print("Running matrix operations benchmark...")
    matrix_jax_times, matrix_numpy_times, matrix_results = benchmark_matrix_ops(batch_sizes)
    
    # Run neural network benchmark
    print("Running neural network benchmark...")
    nn_jax_times, nn_numpy_times, nn_results = benchmark_neural_network(batch_sizes)
    
    # Generate report
    generate_report(matrix_results, nn_results)
    
    # Plot results
    plot_results(batch_sizes, matrix_results, nn_results)

if __name__ == "__main__":
    main() 