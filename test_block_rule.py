#!/usr/bin/env python3
"""
Test JAX block rule checking to ensure it works correctly.
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
        # Generate random boards with random pieces (-1 = opponent, 1 = current player)
        boards = jax.random.randint(
            key, 
            shape=(num_boards, 24), 
            minval=-5, 
            maxval=5,
            dtype=jnp.int32
        )
        return boards
    else:
        # Use NumPy random to generate board states
        np.random.seed(42)
        # Generate random boards with random pieces
        boards = np.random.randint(
            low=-5,
            high=5,
            size=(num_boards, 24),
            dtype=np.int32
        )
        return boards


def create_test_board(use_jax=True):
    """Create a specific board state where the block rule would be relevant."""
    # Create a board where player 1 (positive values) has all pieces in opponent's home
    # and player 2 (negative values) has all pieces in player 1's home
    if use_jax:
        board = jnp.zeros((24,), dtype=jnp.int32)
        # Player 1 has 15 pieces in opponent's home (positions 0-5)
        board = board.at[0:6].set(jnp.array([3, 3, 3, 2, 2, 2], dtype=jnp.int32))
        # Player 2 has 15 pieces in player 1's home (positions 18-23)
        board = board.at[18:24].set(jnp.array([-3, -3, -3, -2, -2, -2], dtype=jnp.int32))
        return board
    else:
        board = np.zeros((24,), dtype=np.int32)
        # Player 1 has 15 pieces in opponent's home (positions 0-5)
        board[0:6] = np.array([3, 3, 3, 2, 2, 2], dtype=np.int32)
        # Player 2 has 15 pieces in player 1's home (positions 18-23)
        board[18:24] = np.array([-3, -3, -3, -2, -2, -2], dtype=np.int32)
        return board


def check_block_rule_numpy(board, player=1):
    """Check if player is violating the block rule (NumPy version)."""
    # For player 1, home positions are 18-23
    # For player 2, home positions are 0-5
    home_positions = range(18, 24) if player == 1 else range(0, 6)
    
    # Count pieces of the opponent in our home
    opponent_val = -1 if player == 1 else 1
    opponent_pieces_in_home = sum(1 for pos in home_positions if board[pos] * opponent_val > 0)
    
    # Count total pieces of the opponent
    opponent_total_pieces = sum(1 for val in board if val * opponent_val > 0)
    
    # The block rule is violated if all of the opponent's pieces are in our home
    return opponent_pieces_in_home == opponent_total_pieces and opponent_total_pieces > 0


@jax.jit
def check_block_rule_jax(board, player=1):
    """Check if player is violating the block rule (JAX version)."""
    # Use static masks for both home areas
    player1_home_mask = jnp.zeros_like(board, dtype=jnp.bool_)
    player1_home_mask = player1_home_mask.at[18:24].set(True)
    
    player2_home_mask = jnp.zeros_like(board, dtype=jnp.bool_)
    player2_home_mask = player2_home_mask.at[0:6].set(True)
    
    # Select the appropriate home mask based on player
    home_mask = jnp.where(player == 1, player1_home_mask, player2_home_mask)
    
    # Determine the opponent's piece sign
    opponent_sign = jnp.where(player == 1, -1, 1)
    
    # Create a mask for opponent's pieces (pieces with opponent's sign)
    is_opponent_piece = jnp.sign(board) == opponent_sign
    
    # Count opponent pieces in home
    opponent_pieces_in_home = jnp.sum(jnp.logical_and(is_opponent_piece, home_mask))
    
    # Count total opponent pieces
    opponent_total_pieces = jnp.sum(is_opponent_piece)
    
    # Check if all opponent pieces are in our home and there's at least one opponent piece
    return jnp.logical_and(
        opponent_pieces_in_home == opponent_total_pieces,
        opponent_total_pieces > 0
    )


def run_block_rule_benchmark(batch_sizes):
    """
    Benchmark block rule checking performance with different batch sizes.
    Block rule checking is necessary for determining legal moves.
    """
    print("\n=== Block Rule Checking Benchmark ===")
    
    jax_times = []
    numpy_times = []
    
    # Creating a batch function for NumPy version
    def check_block_rule_numpy_batch(boards, player=1):
        return np.array([check_block_rule_numpy(board, player) for board in boards])
    
    # JAX version batched through vmap
    check_block_rule_jax_batch = jax.vmap(check_block_rule_jax, in_axes=(0, None))
    
    # Warm up JIT compilation with a single board
    test_board_jax = create_test_board(use_jax=True).reshape(1, 24)
    _ = check_block_rule_jax_batch(test_board_jax, 1)
    
    # Also create the numpy version for verification
    test_board_numpy = create_test_board(use_jax=False).reshape(1, 24)
    numpy_result = check_block_rule_numpy_batch(test_board_numpy, 1)
    jax_result = np.array(check_block_rule_jax_batch(test_board_jax, 1))
    
    print(f"\nTest board verification:")
    print(f"NumPy result: {numpy_result}, JAX result: {jax_result}")
    
    if np.array_equal(numpy_result, jax_result):
        print("✅ Test board verification successful - both implementations give same result")
    else:
        print("❌ Test board verification failed - implementations give different results")
        
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random boards with JAX and NumPy
        boards_jax = create_random_boards(batch_size, use_jax=True)
        boards_numpy = np.array(boards_jax)  # Use same boards for fair comparison
        
        # Benchmark JAX block rule checking
        jax_stmt = """
for _ in range(100):
    result_jax = check_block_rule_jax_batch(boards_jax, 1)
    _ = result_jax.block_until_ready()
"""
        jax_time = timeit.timeit(
            stmt=jax_stmt, 
            globals={
                "check_block_rule_jax_batch": check_block_rule_jax_batch, 
                "boards_jax": boards_jax
            }, 
            number=5
        ) / 5
        print(f"JAX check block rule for {batch_size} boards: {jax_time:.6f} seconds")
        
        # Benchmark NumPy block rule checking
        numpy_stmt = """
for _ in range(100):
    result_numpy = check_block_rule_numpy_batch(boards_numpy, 1)
"""
        numpy_time = timeit.timeit(
            stmt=numpy_stmt, 
            globals={
                "check_block_rule_numpy_batch": check_block_rule_numpy_batch, 
                "boards_numpy": boards_numpy
            }, 
            number=5
        ) / 5
        print(f"NumPy check block rule for {batch_size} boards: {numpy_time:.6f} seconds")
        
        # Record times
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
        
        # Verify results match on a few boards
        sample_size = min(5, batch_size)
        jax_results = np.array(check_block_rule_jax_batch(boards_jax, 1))[:sample_size]
        numpy_results = check_block_rule_numpy_batch(boards_numpy, 1)[:sample_size]
        
        match_count = np.sum(jax_results == numpy_results)
        print(f"Results match for {match_count}/{sample_size} sampled boards")
        
        if match_count == sample_size:
            print("✅ All sampled results match between JAX and NumPy")
        else:
            print("❌ Some results differ between JAX and NumPy")
            print(f"JAX results: {jax_results}")
            print(f"NumPy results: {numpy_results}")
    
    return jax_times, numpy_times


def main():
    # Define batch sizes to test
    batch_sizes = [1, 32, 1024]
    
    # Run block rule benchmark
    jax_block_times, numpy_block_times = run_block_rule_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    for i, batch_size in enumerate(batch_sizes):
        speedup = numpy_block_times[i] / jax_block_times[i] if jax_block_times[i] > 0 else 0
        print(f"Block rule - Batch size: {batch_size}, JAX: {jax_block_times[i]:.6f}s, "
              f"NumPy: {numpy_block_times[i]:.6f}s, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main() 