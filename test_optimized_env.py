#!/usr/bin/env python3
"""
Test the optimized Narde environment with selective JAX acceleration.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import timeit

# Import directly from the standalone file
from optimized_narde_env import OptimizedNardeEnv, simulate_moves_batch

# Show JAX configuration
print(f"JAX default backend: {jax.default_backend()}")
print(f"Available JAX devices: {jax.devices()}")

def test_basic_functionality():
    """Test basic functionality of the environment."""
    print("\n=== Testing Basic Environment Functionality ===")
    env = OptimizedNardeEnv()
    obs, info = env.reset(seed=42)
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial valid actions: {info['valid_actions']}")
    
    # Take a random action
    action = np.random.choice(info['valid_actions'])
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"After action {action}:")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Valid actions: {len(info['valid_actions'])}")
    
    # Render the environment
    env.render()
    
    return env


def test_block_rule_single():
    """Test the block rule implementation for a single board."""
    print("\n=== Testing Block Rule (Single Board) ===")
    env = OptimizedNardeEnv()
    
    # Create a test board where player 1 violates the block rule
    board = np.zeros(24, dtype=np.int32)
    # Player 2 has all pieces in player 1's home (positions 18-23)
    board[18:24] = np.array([-3, -3, -3, -2, -2, -2], dtype=np.int32)
    
    # Check with NumPy (should be faster for single board)
    start_time = time.time()
    violates_numpy = env.check_block_rule(board, player=1, use_jax=False)
    numpy_time = time.time() - start_time
    
    # Check with JAX
    start_time = time.time()
    violates_jax = env.check_block_rule(board, player=1, use_jax=True)
    jax_time = time.time() - start_time
    
    print(f"Block rule violated (NumPy): {violates_numpy}, Time: {numpy_time:.6f}s")
    print(f"Block rule violated (JAX): {violates_jax}, Time: {jax_time:.6f}s")
    
    if violates_numpy == violates_jax:
        print("✅ Both methods give the same result")
    else:
        print("❌ Results differ between NumPy and JAX")


def benchmark_block_rule():
    """Benchmark the block rule checking with various batch sizes."""
    print("\n=== Benchmarking Block Rule Checking ===")
    env = OptimizedNardeEnv()
    
    # Test batch sizes
    batch_sizes = [1, 8, 32, 128, 512, 2048]
    
    # Warm up JIT compilation first to avoid measuring compilation time
    print("Warming up JAX JIT compilation...")
    warmup_boards = np.random.randint(-5, 5, size=(10, 24), dtype=np.int32)
    _ = env.check_block_rule_batch(warmup_boards, 1)
    # Do a few more calls to ensure it's fully compiled
    for _ in range(5):
        _ = env.check_block_rule_batch(warmup_boards, 1)
    print("JIT compilation completed.")
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create random boards
        np.random.seed(42)
        boards = np.random.randint(-5, 5, size=(batch_size, 24), dtype=np.int32)
        
        # Benchmark NumPy version (loop over each board)
        numpy_stmt = """
for _ in range(100):  # Increased from 1 to 100 iterations
    result = np.array([env._check_block_rule_numpy(board, 1) for board in boards])
"""
        numpy_time = timeit.timeit(
            stmt=numpy_stmt, 
            globals={
                "env": env, 
                "boards": boards,
                "np": np
            }, 
            number=5
        ) / 5
        print(f"NumPy check (loop, 100 iterations): {numpy_time:.6f} seconds")
        
        # Benchmark JAX batch version
        jax_stmt = """
for _ in range(100):  # Increased from 1 to 100 iterations
    result = env.check_block_rule_batch(boards, 1)
    # Force JAX to complete the computation (equivalent to block_until_ready)
    _ = jax.device_get(result)
"""
        jax_time = timeit.timeit(
            stmt=jax_stmt, 
            globals={
                "env": env, 
                "boards": boards,
                "np": np,
                "jax": jax
            }, 
            number=5
        ) / 5
        print(f"JAX batch check (100 iterations): {jax_time:.6f} seconds")
        
        # Calculate speedup
        speedup = numpy_time / jax_time if jax_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results match
        numpy_result = np.array([env._check_block_rule_numpy(board, 1) for board in boards])
        jax_result = env.check_block_rule_batch(boards, 1)
        
        if np.array_equal(numpy_result, jax_result):
            print("✅ Results match between NumPy and JAX")
        else:
            print("❌ Results differ between NumPy and JAX")
            mismatches = np.sum(numpy_result != jax_result)
            print(f"   {mismatches}/{batch_size} mismatches")


def test_batch_simulation():
    """Test batched simulation for MCTS."""
    print("\n=== Testing Batch Simulation ===")
    env = OptimizedNardeEnv()
    
    # Test multiple batch sizes
    batch_sizes = [32, 128, 512, 2048]
    
    # Warm up JAX
    print("Warming up JAX JIT compilation...")
    warmup_boards = np.array([env.board.copy() for _ in range(10)])
    warmup_actions = np.random.randint(0, 30, size=10)
    _, _ = simulate_moves_batch(env, warmup_boards, 1, warmup_actions, use_jax=True)
    print("JIT compilation completed.")
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create batch of boards
        boards = np.array([env.board.copy() for _ in range(batch_size)])
        
        # Create random actions
        actions = np.random.randint(0, 30, size=batch_size)
        
        # Benchmark JAX simulation (with timeit for accuracy)
        jax_stmt = """
for _ in range(10):
    new_boards, violates = simulate_moves_batch(env, boards, 1, actions, use_jax=True)
    # Force JAX to complete the computation
    _ = jax.device_get(violates)
"""
        jax_time = timeit.timeit(
            stmt=jax_stmt, 
            globals={
                "simulate_moves_batch": simulate_moves_batch,
                "env": env, 
                "boards": boards,
                "actions": actions,
                "jax": jax
            }, 
            number=5
        ) / 5
        
        # Benchmark NumPy simulation
        numpy_stmt = """
for _ in range(10):
    new_boards, violates = simulate_moves_batch(env, boards, 1, actions, use_jax=False)
"""
        numpy_time = timeit.timeit(
            stmt=numpy_stmt, 
            globals={
                "simulate_moves_batch": simulate_moves_batch,
                "env": env, 
                "boards": boards,
                "actions": actions
            }, 
            number=5
        ) / 5
        
        print(f"JAX batch simulation (10 iterations): {jax_time:.6f}s")
        print(f"NumPy batch simulation (10 iterations): {numpy_time:.6f}s")
        
        # Calculate speedup both ways
        if jax_time > 0:
            numpy_vs_jax = numpy_time / jax_time
            print(f"JAX Speedup vs NumPy: {numpy_vs_jax:.2f}x")
        else:
            print("JAX time was too small to calculate speedup")
            
        # Verify results match
        new_boards_jax, violates_jax = simulate_moves_batch(env, boards, 1, actions, use_jax=True)
        new_boards_numpy, violates_numpy = simulate_moves_batch(env, boards, 1, actions, use_jax=False)
        
        boards_match = np.array_equal(new_boards_jax, new_boards_numpy)
        rules_match = np.array_equal(violates_jax, violates_numpy)
        
        if boards_match and rules_match:
            print("✅ Results match between JAX and NumPy")
        else:
            print("❌ Results differ between JAX and NumPy")
            if not boards_match:
                print("   Board states differ")
            if not rules_match:
                print("   Block rule violations differ")


def main():
    # Test environment functionality
    env = test_basic_functionality()
    
    # Test block rule checking for a single board
    test_block_rule_single()
    
    # Benchmark block rule checking with different batch sizes
    benchmark_block_rule()
    
    # Test batch simulation for MCTS
    test_batch_simulation()


if __name__ == "__main__":
    main() 