#!/usr/bin/env python3
"""
Benchmark JAX acceleration with bfloat16 precision for neural network operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import timeit
import flax.linen as nn

# Configure JAX
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available JAX devices: {jax.devices()}")

# Define a simple neural network
class SimpleNetwork(nn.Module):
    """A simple neural network similar to what might be used in MuZero."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

class SimpleNetworkPyTorch:
    """PyTorch equivalent of SimpleNetwork for comparison."""
    def __init__(self, hidden_dim, input_dim):
        import torch
        import torch.nn as nn
        
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Move to the best available device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                "cuda" if torch.cuda.is_available() else 
                                "cpu")
        self.model = self.model.to(self.device)
        print(f"PyTorch using device: {self.device}")
    
    def __call__(self, x):
        import torch
        
        # Move input to device
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        result = self.model(x)
        
        # Move result back to CPU for comparison
        return result.detach().cpu().numpy()


def create_random_input(batch_size, input_dim, use_jax=True, dtype=None):
    """Create random input for neural network."""
    if use_jax:
        key = jax.random.PRNGKey(42)
        # Generate random input
        if dtype is None:
            dtype = jnp.float32
        x = jax.random.normal(key, shape=(batch_size, input_dim), dtype=dtype)
        return x
    else:
        np.random.seed(42)
        return np.random.normal(size=(batch_size, input_dim)).astype(np.float32)


def run_neural_network_benchmark(batch_sizes, input_dim=24, hidden_dim=128):
    """
    Benchmark neural network forward pass with different batch sizes and precisions.
    """
    print(f"\n=== Neural Network Benchmark (input_dim={input_dim}, hidden_dim={hidden_dim}) ===")
    
    jax_fp32_times = []
    jax_bf16_times = []
    torch_times = []
    
    # Create models
    model_fp32 = SimpleNetwork(hidden_dim=hidden_dim)
    model_bf16 = SimpleNetwork(hidden_dim=hidden_dim)
    
    # Generate a key for parameter initialization
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    
    # Initialize with a small batch to get parameters
    init_batch = create_random_input(1, input_dim)
    params_fp32 = model_fp32.init(key1, init_batch)
    
    # Initialize BF16 model by casting parameters to bfloat16
    def cast_to_bfloat16(params):
        return jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    
    params_bf16 = cast_to_bfloat16(model_fp32.init(key2, init_batch))
    
    # Initialize PyTorch model
    try:
        model_torch = SimpleNetworkPyTorch(hidden_dim=hidden_dim, input_dim=input_dim)
    except ImportError:
        print("PyTorch not available, skipping PyTorch benchmarks")
        model_torch = None
    
    # Define JIT-compiled forward pass functions
    @jax.jit
    def forward_fp32(params, x):
        return model_fp32.apply(params, x)
    
    @jax.jit
    def forward_bf16(params, x):
        # Cast input to bfloat16
        x_bf16 = x.astype(jnp.bfloat16)
        return model_bf16.apply(params, x_bf16)
    
    # Warm up JIT compilation
    x_small = create_random_input(1, input_dim)
    _ = forward_fp32(params_fp32, x_small)
    _ = forward_bf16(params_bf16, x_small)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random input
        x_jax = create_random_input(batch_size, input_dim)
        x_np = np.array(x_jax)  # For PyTorch
        
        # Benchmark JAX FP32 forward pass
        jax_fp32_stmt = """
for _ in range(100):
    output = forward_fp32(params_fp32, x_jax)
    _ = output.block_until_ready()
"""
        jax_fp32_time = timeit.timeit(
            stmt=jax_fp32_stmt, 
            globals={
                "forward_fp32": forward_fp32, 
                "params_fp32": params_fp32, 
                "x_jax": x_jax
            }, 
            number=5
        ) / 5
        print(f"JAX FP32 forward pass for {batch_size} samples: {jax_fp32_time:.6f} seconds")
        
        # Benchmark JAX BF16 forward pass
        jax_bf16_stmt = """
for _ in range(100):
    output = forward_bf16(params_bf16, x_jax)
    _ = output.block_until_ready()
"""
        jax_bf16_time = timeit.timeit(
            stmt=jax_bf16_stmt, 
            globals={
                "forward_bf16": forward_bf16, 
                "params_bf16": params_bf16, 
                "x_jax": x_jax
            }, 
            number=5
        ) / 5
        print(f"JAX BF16 forward pass for {batch_size} samples: {jax_bf16_time:.6f} seconds")
        
        # Benchmark PyTorch forward pass if available
        torch_time = 0
        if model_torch is not None:
            torch_stmt = """
for _ in range(100):
    output = model_torch(x_np)
"""
            try:
                import torch
                
                torch_time = timeit.timeit(
                    stmt=torch_stmt, 
                    globals={
                        "model_torch": model_torch, 
                        "x_np": x_np
                    }, 
                    number=5
                ) / 5
                print(f"PyTorch forward pass for {batch_size} samples: {torch_time:.6f} seconds")
                
                # Synchronize for proper timing if using CUDA or MPS
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    
            except ImportError:
                print("PyTorch not available for benchmarking")
        
        # Record times
        jax_fp32_times.append(jax_fp32_time)
        jax_bf16_times.append(jax_bf16_time)
        torch_times.append(torch_time)
        
        # Verify results (on smaller batch sizes)
        if batch_size <= 32:
            output_fp32 = np.array(forward_fp32(params_fp32, x_jax))
            output_bf16 = np.array(forward_bf16(params_bf16, x_jax))
            
            # We expect some difference between FP32 and BF16, but they should be close
            mean_abs_diff = np.mean(np.abs(output_fp32 - output_bf16))
            print(f"Mean absolute difference between FP32 and BF16: {mean_abs_diff:.6f}")
            
            # Check if outputs are reasonably close (allow more difference for BF16)
            if mean_abs_diff < 0.1:
                print("✅ BF16 results reasonably match FP32 results")
            else:
                print("❌ BF16 results differ significantly from FP32 results")
            
            # Check PyTorch vs JAX if available
            if model_torch is not None:
                try:
                    output_torch = model_torch(x_np)
                    # Just print the shapes, the values will be different due to different initialization
                    print(f"PyTorch output shape: {output_torch.shape}, JAX output shape: {output_fp32.shape}")
                except Exception as e:
                    print(f"Error running PyTorch model: {e}")
    
    return jax_fp32_times, jax_bf16_times, torch_times


def main():
    # Define batch sizes to test
    small_batch_sizes = [1, 8, 32]
    large_batch_sizes = [128, 512]
    extra_large_batch_sizes = [4096, 8192]  # Exclude 16384 for neural networks to avoid memory issues
    batch_sizes = small_batch_sizes + large_batch_sizes + extra_large_batch_sizes
    
    # Run benchmarks with a small network first
    jax_fp32_times1, jax_bf16_times1, torch_times1 = run_neural_network_benchmark(
        batch_sizes, input_dim=24, hidden_dim=128
    )
    
    # Run benchmarks with a larger network
    jax_fp32_times2, jax_bf16_times2, torch_times2 = run_neural_network_benchmark(
        batch_sizes, input_dim=128, hidden_dim=512
    )
    
    # Print summary
    print("\n=== Performance Summary ===")
    
    print("\nSmall Network (input_dim=24, hidden_dim=128):")
    for i, batch_size in enumerate(batch_sizes):
        if torch_times1[i] > 0:
            fp32_vs_torch = torch_times1[i] / jax_fp32_times1[i] if jax_fp32_times1[i] > 0 else 0
            bf16_vs_torch = torch_times1[i] / jax_bf16_times1[i] if jax_bf16_times1[i] > 0 else 0
        else:
            fp32_vs_torch = 0
            bf16_vs_torch = 0
            
        bf16_vs_fp32 = jax_fp32_times1[i] / jax_bf16_times1[i] if jax_bf16_times1[i] > 0 else 0
        
        print(f"Batch size: {batch_size}, "
              f"JAX FP32: {jax_fp32_times1[i]:.6f}s, "
              f"JAX BF16: {jax_bf16_times1[i]:.6f}s, "
              f"PyTorch: {torch_times1[i]:.6f}s")
        print(f"  • BF16 Speedup vs FP32: {bf16_vs_fp32:.2f}x, "
              f"FP32 Speedup vs PyTorch: {fp32_vs_torch:.2f}x, "
              f"BF16 Speedup vs PyTorch: {bf16_vs_torch:.2f}x")
    
    print("\nLarge Network (input_dim=128, hidden_dim=512):")
    for i, batch_size in enumerate(batch_sizes):
        if torch_times2[i] > 0:
            fp32_vs_torch = torch_times2[i] / jax_fp32_times2[i] if jax_fp32_times2[i] > 0 else 0
            bf16_vs_torch = torch_times2[i] / jax_bf16_times2[i] if jax_bf16_times2[i] > 0 else 0
        else:
            fp32_vs_torch = 0
            bf16_vs_torch = 0
            
        bf16_vs_fp32 = jax_fp32_times2[i] / jax_bf16_times2[i] if jax_bf16_times2[i] > 0 else 0
        
        print(f"Batch size: {batch_size}, "
              f"JAX FP32: {jax_fp32_times2[i]:.6f}s, "
              f"JAX BF16: {jax_bf16_times2[i]:.6f}s, "
              f"PyTorch: {torch_times2[i]:.6f}s")
        print(f"  • BF16 Speedup vs FP32: {bf16_vs_fp32:.2f}x, "
              f"FP32 Speedup vs PyTorch: {fp32_vs_torch:.2f}x, "
              f"BF16 Speedup vs PyTorch: {bf16_vs_torch:.2f}x")


if __name__ == "__main__":
    main() 