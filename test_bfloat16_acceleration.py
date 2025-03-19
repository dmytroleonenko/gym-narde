#!/usr/bin/env python3
"""
Benchmark JAX acceleration with bfloat16 precision for Narde game operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import timeit

# Configure JAX
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available JAX devices: {jax.devices()}")


def create_random_boards(num_boards, use_jax=True, use_bfloat16=False):
    """Create random Narde board states."""
    if use_jax:
        # Use JAX random to generate board states
        key = jax.random.PRNGKey(42)
        # Generate random boards with random pieces (-15 to 15)
        boards = jax.random.randint(
            key, 
            shape=(num_boards, 24), 
            minval=-15, 
            maxval=15,
            dtype=jnp.int32
        )
        # Convert to bfloat16 if needed
        if use_bfloat16:
            boards = boards.astype(jnp.bfloat16)
        return boards
    else:
        # Use NumPy random to generate board states
        np.random.seed(42)
        # Generate random boards with random pieces
        boards = np.random.randint(
            low=-15,
            high=15,
            size=(num_boards, 24),
            dtype=np.int32
        )
        return boards


def run_board_rotation_benchmark(batch_sizes):
    """
    Benchmark board rotation performance with different batch sizes and precisions.
    Board rotation is a fundamental operation that occurs after every player's turn.
    """
    print("\n=== Board Rotation Benchmark ===")
    
    jax_fp32_times = []
    jax_bf16_times = []
    numpy_times = []
    
    @jax.jit
    def rotate_boards_jax_fp32(boards):
        """JAX-accelerated function to rotate multiple boards at once (FP32)."""
        return -1 * jnp.flip(boards, axis=1)
    
    @jax.jit
    def rotate_boards_jax_bf16(boards):
        """JAX-accelerated function to rotate multiple boards at once (BF16)."""
        # Convert to bfloat16, perform operation, and convert back
        boards_bf16 = boards.astype(jnp.bfloat16)
        result = -1 * jnp.flip(boards_bf16, axis=1)
        return result
    
    def rotate_boards_numpy(boards):
        """NumPy function to rotate multiple boards."""
        return -1 * np.flip(boards, axis=1)
    
    # Warm up JIT compilation
    boards_jax = create_random_boards(1, use_jax=True)
    boards_jax_bf16 = create_random_boards(1, use_jax=True, use_bfloat16=True)
    _ = rotate_boards_jax_fp32(boards_jax)
    _ = rotate_boards_jax_bf16(boards_jax_bf16)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random boards with JAX and NumPy
        boards_jax = create_random_boards(batch_size, use_jax=True)
        boards_jax_bf16 = create_random_boards(batch_size, use_jax=True, use_bfloat16=True)
        boards_numpy = create_random_boards(batch_size, use_jax=False)
        
        # Benchmark JAX FP32 rotation
        jax_fp32_stmt = """
for _ in range(100):
    rotated_jax = rotate_boards_jax_fp32(boards_jax)
    _ = rotated_jax.block_until_ready()
"""
        jax_fp32_time = timeit.timeit(
            stmt=jax_fp32_stmt, 
            globals={"rotate_boards_jax_fp32": rotate_boards_jax_fp32, "boards_jax": boards_jax}, 
            number=5
        ) / 5
        print(f"JAX FP32 rotate {batch_size} boards: {jax_fp32_time:.6f} seconds")
        
        # Benchmark JAX BF16 rotation
        jax_bf16_stmt = """
for _ in range(100):
    rotated_jax = rotate_boards_jax_bf16(boards_jax_bf16)
    _ = rotated_jax.block_until_ready()
"""
        jax_bf16_time = timeit.timeit(
            stmt=jax_bf16_stmt, 
            globals={"rotate_boards_jax_bf16": rotate_boards_jax_bf16, "boards_jax_bf16": boards_jax_bf16}, 
            number=5
        ) / 5
        print(f"JAX BF16 rotate {batch_size} boards: {jax_bf16_time:.6f} seconds")
        
        # Benchmark NumPy rotation
        numpy_stmt = """
for _ in range(100):
    rotated_numpy = rotate_boards_numpy(boards_numpy)
"""
        numpy_time = timeit.timeit(
            stmt=numpy_stmt, 
            globals={"rotate_boards_numpy": rotate_boards_numpy, "boards_numpy": boards_numpy}, 
            number=5
        ) / 5
        print(f"NumPy rotate {batch_size} boards: {numpy_time:.6f} seconds")
        
        # Record times
        jax_fp32_times.append(jax_fp32_time)
        jax_bf16_times.append(jax_bf16_time)
        numpy_times.append(numpy_time)
        
        # Verify results are correct
        jax_fp32_rotated = np.array(rotate_boards_jax_fp32(boards_jax))
        jax_bf16_rotated = np.array(rotate_boards_jax_bf16(boards_jax_bf16))
        numpy_rotated = rotate_boards_numpy(boards_numpy)
        
        # Check a sample board for each
        sample_idx = 0
        jax_fp32_sample = jax_fp32_rotated[sample_idx]
        jax_bf16_sample = jax_bf16_rotated[sample_idx]
        numpy_sample = numpy_rotated[sample_idx]
        
        # Original board samples
        jax_fp32_original = np.array(boards_jax)[sample_idx]
        jax_bf16_original = np.array(boards_jax_bf16)[sample_idx]
        numpy_original = boards_numpy[sample_idx]
        
        # Calculate expected results
        jax_fp32_expected = -1 * np.flip(jax_fp32_original)
        jax_bf16_expected = -1 * np.flip(jax_bf16_original)
        numpy_expected = -1 * np.flip(numpy_original)
        
        # Check if results match expected
        jax_fp32_correct = np.allclose(jax_fp32_sample, jax_fp32_expected)
        jax_bf16_correct = np.allclose(jax_bf16_sample, jax_bf16_expected, rtol=1e-2)  # Higher tolerance for BF16
        numpy_correct = np.allclose(numpy_sample, numpy_expected)
        
        # Report verification results
        if jax_fp32_correct:
            print("✅ JAX FP32 rotation is correct")
        else:
            print("❌ JAX FP32 rotation has issues")
            
        if jax_bf16_correct:
            print("✅ JAX BF16 rotation is correct") 
        else:
            print("❌ JAX BF16 rotation has issues")
            
        if numpy_correct:
            print("✅ NumPy rotation is correct") 
        else:
            print("❌ NumPy rotation has issues")
    
    return jax_fp32_times, jax_bf16_times, numpy_times


def main():
    # Define batch sizes to test
    small_batch_sizes = [1, 8, 32, 128]
    large_batch_sizes = [512, 2048]
    extra_large_batch_sizes = [4096, 8192, 16384]
    batch_sizes = small_batch_sizes + large_batch_sizes + extra_large_batch_sizes
    
    # Run benchmarks
    jax_fp32_times, jax_bf16_times, numpy_times = run_board_rotation_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("\nBoard Rotation:")
    for i, batch_size in enumerate(batch_sizes):
        fp32_speedup = numpy_times[i] / jax_fp32_times[i] if jax_fp32_times[i] > 0 else 0
        bf16_speedup = numpy_times[i] / jax_bf16_times[i] if jax_bf16_times[i] > 0 else 0
        bf16_vs_fp32 = jax_fp32_times[i] / jax_bf16_times[i] if jax_bf16_times[i] > 0 else 0
        
        print(f"Batch size: {batch_size}, "
              f"JAX FP32: {jax_fp32_times[i]:.6f}s, "
              f"JAX BF16: {jax_bf16_times[i]:.6f}s, "
              f"NumPy: {numpy_times[i]:.6f}s")
        print(f"  • FP32 Speedup vs NumPy: {fp32_speedup:.2f}x, "
              f"BF16 Speedup vs NumPy: {bf16_speedup:.2f}x, "
              f"BF16 Speedup vs FP32: {bf16_vs_fp32:.2f}x")


if __name__ == "__main__":
    main() 