#!/usr/bin/env python3
"""
Test JAX board rotation only to ensure it works correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import timeit

# Configure JAX 
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available JAX devices: {jax.devices()}")


def create_random_boards(num_boards, use_jax=True):
    """Create random Narde board states."""
    if use_jax:
        # Use JAX random to generate board states
        key = jax.random.PRNGKey(42)
        # Generate random boards with random pieces
        boards = jax.random.randint(
            key, 
            shape=(num_boards, 24), 
            minval=-15, 
            maxval=15,
            dtype=jnp.int32
        )
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
    Benchmark board rotation performance with different batch sizes.
    Board rotation is a fundamental operation that occurs after every player's turn.
    """
    print("\n=== Board Rotation Benchmark ===")
    
    jax_times = []
    numpy_times = []
    
    @jax.jit
    def rotate_boards_jax(boards):
        """JAX-accelerated function to rotate multiple boards at once."""
        return -1 * jnp.flip(boards, axis=1)
    
    def rotate_boards_numpy(boards):
        """NumPy function to rotate multiple boards."""
        return -1 * np.flip(boards, axis=1)
    
    # Warm up JIT compilation
    boards_jax = create_random_boards(1, use_jax=True)
    _ = rotate_boards_jax(boards_jax)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random boards with JAX and NumPy
        boards_jax = create_random_boards(batch_size, use_jax=True)
        boards_numpy = create_random_boards(batch_size, use_jax=False)
        
        # Print the first board from each to check if they're the same
        print(f"JAX first board: {np.array(boards_jax)[0]}")
        print(f"NumPy first board: {boards_numpy[0]}")
        
        # See what they look like after rotation
        rotated_jax = rotate_boards_jax(boards_jax)
        rotated_numpy = rotate_boards_numpy(boards_numpy)
        
        print(f"JAX rotated first board: {np.array(rotated_jax)[0]}")
        print(f"NumPy rotated first board: {rotated_numpy[0]}")
        
        # Benchmark JAX rotation
        jax_stmt = """
for _ in range(100):
    rotated_jax = rotate_boards_jax(boards_jax)
    _ = rotated_jax.block_until_ready()
"""
        jax_time = timeit.timeit(stmt=jax_stmt, globals={"rotate_boards_jax": rotate_boards_jax, "boards_jax": boards_jax}, number=5) / 5
        print(f"JAX rotate {batch_size} boards: {jax_time:.6f} seconds")
        
        # Benchmark NumPy rotation
        numpy_stmt = """
for _ in range(100):
    rotated_numpy = rotate_boards_numpy(boards_numpy)
"""
        numpy_time = timeit.timeit(stmt=numpy_stmt, globals={"rotate_boards_numpy": rotate_boards_numpy, "boards_numpy": boards_numpy}, number=5) / 5
        print(f"NumPy rotate {batch_size} boards: {numpy_time:.6f} seconds")
        
        # Record times
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
        
        # Verify results are the same
        rotated_jax_np = np.array(rotate_boards_jax(boards_jax))
        rotated_numpy_np = rotate_boards_numpy(boards_numpy)
        
        # Print shapes for debugging
        print(f"JAX rotated shape: {rotated_jax_np.shape}, NumPy rotated shape: {rotated_numpy_np.shape}")
        
        # Check operation correctness
        jax_correctly_rotated = np.array_equal(-1 * np.flip(np.array(boards_jax)[0]), rotated_jax_np[0])
        numpy_correctly_rotated = np.array_equal(-1 * np.flip(boards_numpy[0]), rotated_numpy_np[0])
        
        if jax_correctly_rotated:
            print("✅ JAX rotation is correct")
        else:
            print("❌ JAX rotation has issues")
            
        if numpy_correctly_rotated:
            print("✅ NumPy rotation is correct") 
        else:
            print("❌ NumPy rotation has issues")
    
    return jax_times, numpy_times


def main():
    # Define batch sizes to test (use fewer sizes for faster execution)
    batch_sizes = [1, 32, 2048]
    
    # Run benchmarks
    jax_rotation_times, numpy_rotation_times = run_board_rotation_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    for i, batch_size in enumerate(batch_sizes):
        speedup = numpy_rotation_times[i] / jax_rotation_times[i] if jax_rotation_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, JAX: {jax_rotation_times[i]:.6f}s, NumPy: {numpy_rotation_times[i]:.6f}s, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main() 