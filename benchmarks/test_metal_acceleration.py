#!/usr/bin/env python3
"""
Simplified benchmark for JAX with Metal backend acceleration.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import timeit
import time

print(f"JAX version: {jax.__version__}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available JAX devices: {jax.devices()}")

# Simple matrix multiplication benchmark
def run_matmul_benchmark(batch_sizes):
    """Benchmark matrix multiplication with different batch sizes and frameworks."""
    print("\n=== Matrix Multiplication Benchmark ===")
    
    jax_times = []
    numpy_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random matrices with NumPy
        np.random.seed(42)
        a_np = np.random.random((batch_size, 128, 128)).astype(np.float32)
        b_np = np.random.random((batch_size, 128, 128)).astype(np.float32)
        
        # JAX implementation
        @jax.jit
        def matmul_jax(a, b):
            return jnp.matmul(a, b)
        
        # NumPy implementation
        def matmul_numpy(a, b):
            return np.matmul(a, b)
        
        # Convert to JAX arrays
        a_jax = jnp.array(a_np)
        b_jax = jnp.array(b_np)
        
        # Warm up JAX JIT compilation
        _ = matmul_jax(a_jax, b_jax).block_until_ready()
        
        # Benchmark JAX
        start_time = time.time()
        for _ in range(10):
            result = matmul_jax(a_jax, b_jax)
            _ = result.block_until_ready()
        jax_time = (time.time() - start_time) / 10
        print(f"JAX matrix multiplication: {jax_time:.6f} seconds")
        
        # Benchmark NumPy
        start_time = time.time()
        for _ in range(10):
            _ = matmul_numpy(a_np, b_np)
        numpy_time = (time.time() - start_time) / 10
        print(f"NumPy matrix multiplication: {numpy_time:.6f} seconds")
        
        # Calculate speedup
        speedup = numpy_time / jax_time if jax_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
        
    return jax_times, numpy_times

# Simple vector operations benchmark
def run_vector_ops_benchmark(batch_sizes):
    """Benchmark vector operations with different batch sizes and frameworks."""
    print("\n=== Vector Operations Benchmark ===")
    
    jax_times = []
    numpy_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random vectors with NumPy
        np.random.seed(42)
        a_np = np.random.random((batch_size, 1024)).astype(np.float32)
        
        # JAX implementation
        @jax.jit
        def vector_ops_jax(a):
            # Perform a series of vector operations
            b = jnp.sin(a)
            c = jnp.cos(a)
            d = jnp.exp(a * 0.01)  # Scale to avoid overflow
            e = jnp.tanh(a)
            return b + c + d + e
        
        # NumPy implementation
        def vector_ops_numpy(a):
            # Same operations with NumPy
            b = np.sin(a)
            c = np.cos(a)
            d = np.exp(a * 0.01)  # Scale to avoid overflow
            e = np.tanh(a)
            return b + c + d + e
        
        # Convert to JAX arrays
        a_jax = jnp.array(a_np)
        
        # Warm up JAX JIT compilation
        _ = vector_ops_jax(a_jax).block_until_ready()
        
        # Benchmark JAX
        start_time = time.time()
        for _ in range(10):
            result = vector_ops_jax(a_jax)
            _ = result.block_until_ready()
        jax_time = (time.time() - start_time) / 10
        print(f"JAX vector operations: {jax_time:.6f} seconds")
        
        # Benchmark NumPy
        start_time = time.time()
        for _ in range(10):
            _ = vector_ops_numpy(a_np)
        numpy_time = (time.time() - start_time) / 10
        print(f"NumPy vector operations: {numpy_time:.6f} seconds")
        
        # Calculate speedup
        speedup = numpy_time / jax_time if jax_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
        
    return jax_times, numpy_times

def main():
    # Define batch sizes to test
    small_batch_sizes = [1, 8, 32, 128]
    large_batch_sizes = [512, 2048]
    extra_large_batch_sizes = [4096, 8192]
    batch_sizes = small_batch_sizes + large_batch_sizes + extra_large_batch_sizes
    
    # Run benchmarks
    jax_matmul_times, numpy_matmul_times = run_matmul_benchmark(batch_sizes)
    jax_vector_times, numpy_vector_times = run_vector_ops_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("\nMatrix Multiplication:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = numpy_matmul_times[i] / jax_matmul_times[i] if jax_matmul_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, JAX: {jax_matmul_times[i]:.6f}s, NumPy: {numpy_matmul_times[i]:.6f}s, Speedup: {speedup:.2f}x")
    
    print("\nVector Operations:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = numpy_vector_times[i] / jax_vector_times[i] if jax_vector_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, JAX: {jax_vector_times[i]:.6f}s, NumPy: {numpy_vector_times[i]:.6f}s, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 