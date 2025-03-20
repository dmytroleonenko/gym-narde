#!/usr/bin/env python3
"""
Benchmark JAX acceleration for Narde game operations.
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
    # Generate the boards with NumPy first (avoids Metal backend issues with random generation)
    np.random.seed(42)
    boards = np.random.randint(
        low=-15,
        high=15,
        size=(num_boards, 24),
        dtype=np.int32
    )
    
    if use_jax:
        # Convert to JAX array if needed
        return jnp.array(boards)
    else:
        return boards


def create_test_board_for_block_rule(use_jax=True):
    """Create a specific board state where the block rule would be relevant."""
    # Create a board where player 1 (positive values) has all pieces in opponent's home
    # and player 2 (negative values) has all pieces in player 1's home
    
    # Create with NumPy first
    board = np.zeros((24,), dtype=np.int32)
    # Player 1 has 15 pieces in opponent's home (positions 0-5)
    board[0:6] = np.array([3, 3, 3, 2, 2, 2], dtype=np.int32)
    # Player 2 has 15 pieces in player 1's home (positions 18-23)
    board[18:24] = np.array([-3, -3, -3, -2, -2, -2], dtype=np.int32)
    
    if use_jax:
        # Convert to JAX array if needed
        return jnp.array(board)
    else:
        return board


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
        
        # Verify results are correct using reference implementation
        jax_rotated = np.array(rotate_boards_jax(boards_jax))
        jax_correctly_rotated = True
        
        # Check a sample of boards to verify rotation is correct
        for i in range(min(5, batch_size)):
            original_board = np.array(boards_jax)[i]
            rotated_board = jax_rotated[i]
            # Verify using NumPy operations
            expected_rotation = -1 * np.flip(original_board)
            if not np.array_equal(rotated_board, expected_rotation):
                jax_correctly_rotated = False
                break
        
        # Same verification for NumPy
        numpy_rotated = rotate_boards_numpy(boards_numpy)
        numpy_correctly_rotated = True
        
        for i in range(min(5, batch_size)):
            original_board = boards_numpy[i]
            rotated_board = numpy_rotated[i]
            expected_rotation = -1 * np.flip(original_board)
            if not np.array_equal(rotated_board, expected_rotation):
                numpy_correctly_rotated = False
                break
        
        # Report verification results
        if jax_correctly_rotated:
            print("✅ JAX rotation is correct")
        else:
            print("❌ JAX rotation has issues")
            
        if numpy_correctly_rotated:
            print("✅ NumPy rotation is correct") 
        else:
            print("❌ NumPy rotation has issues")
    
    return jax_times, numpy_times


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
    # Define indices for both home areas
    player1_home_indices = jnp.arange(18, 24)
    player2_home_indices = jnp.arange(0, 6)
    
    # Select the appropriate home indices based on player
    home_indices = jnp.where(player == 1, player1_home_indices, player2_home_indices)
    
    # Determine opponent's sign
    opponent_sign = jnp.where(player == 1, -1, 1)
    
    # Count opponent pieces in home and total opponent pieces
    opponent_pieces_in_home = 0
    opponent_total_pieces = 0
    
    # Use a scan-like approach instead of logical operations
    def count_pieces(i, accum):
        in_home = jnp.sum(jnp.where(i == home_indices, 1, 0))
        is_opponent = jnp.sign(board[i]) == opponent_sign
        
        opponent_pieces_in_home, opponent_total_pieces = accum
        opponent_pieces_in_home += is_opponent * in_home
        opponent_total_pieces += is_opponent
        
        return (opponent_pieces_in_home, opponent_total_pieces)
    
    # Scan over all board positions
    init_accum = (0, 0)
    for i in range(24):
        init_accum = count_pieces(i, init_accum)
    
    opponent_pieces_in_home, opponent_total_pieces = init_accum
    
    # Check if all opponent pieces are in our home and there's at least one opponent piece
    return (opponent_pieces_in_home == opponent_total_pieces) & (opponent_total_pieces > 0)


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
    test_board_jax = create_test_board_for_block_rule(use_jax=True).reshape(1, 24)
    _ = check_block_rule_jax_batch(test_board_jax, 1)
    
    # Also create the numpy version for verification
    test_board_numpy = create_test_board_for_block_rule(use_jax=False).reshape(1, 24)
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


def benchmark_environment_creation(num_iterations=10):
    """
    Benchmark creating environments with JAX vs NumPy.
    """
    print("\n=== Environment Creation Benchmark ===")
    
    # Benchmark NumPy environment
    numpy_stmt = """
for _ in range(num_iterations):
    env = gym.make('Narde-v0')
    env.reset()
    env.close()
"""
    numpy_time = timeit.timeit(stmt=numpy_stmt, globals={"gym": gym, "num_iterations": num_iterations}, number=5) / 5
    print(f"Create {num_iterations} NumPy environments: {numpy_time:.6f} seconds")
    
    # Benchmark JAX environment
    jax_stmt = """
for _ in range(num_iterations):
    env = gym.make('Narde-jax-v0')
    env.reset()
    env.close()
"""
    jax_time = timeit.timeit(stmt=jax_stmt, globals={"gym": gym, "num_iterations": num_iterations}, number=5) / 5
    print(f"Create {num_iterations} JAX environments: {jax_time:.6f} seconds")


def benchmark_batch_processing(batch_sizes):
    """
    Benchmark batch processing performance with JAX vs NumPy.
    """
    print("\n=== Batch Processing Benchmark ===")
    
    jax_times = []
    numpy_times = []
    
    @jax.jit
    def process_batch_jax(batch):
        """Simple JAX batch processing function."""
        # Convert to bfloat16
        batch = batch.astype(jnp.bfloat16)
        # Normalize
        batch = batch / 15.0
        # Apply some transformations
        batch = jnp.sin(batch) + jnp.cos(batch)
        batch = jnp.exp(batch) / (1.0 + jnp.exp(batch))  # sigmoid
        return batch
    
    def process_batch_numpy(batch):
        """Simple NumPy batch processing function."""
        # Convert to float16 (closest to bfloat16)
        batch = batch.astype(np.float16)
        # Normalize
        batch = batch / 15.0
        # Apply same transformations
        batch = np.sin(batch) + np.cos(batch)
        batch = np.exp(batch) / (1.0 + np.exp(batch))  # sigmoid
        return batch
    
    # Warm up JIT compilation
    batch_jax = create_random_boards(1, use_jax=True)
    _ = process_batch_jax(batch_jax)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random data with JAX and NumPy
        batch_jax = create_random_boards(batch_size, use_jax=True)
        batch_numpy = create_random_boards(batch_size, use_jax=False)
        
        # Benchmark JAX processing
        jax_stmt = """
for _ in range(10):
    result_jax = process_batch_jax(batch_jax)
    _ = result_jax.block_until_ready()
"""
        jax_time = timeit.timeit(stmt=jax_stmt, globals={"process_batch_jax": process_batch_jax, "batch_jax": batch_jax}, number=5) / 5
        print(f"JAX process batch of {batch_size}: {jax_time:.6f} seconds")
        
        # Benchmark NumPy processing
        numpy_stmt = """
for _ in range(10):
    result_numpy = process_batch_numpy(batch_numpy)
"""
        numpy_time = timeit.timeit(stmt=numpy_stmt, globals={"process_batch_numpy": process_batch_numpy, "batch_numpy": batch_numpy}, number=5) / 5
        print(f"NumPy process batch of {batch_size}: {numpy_time:.6f} seconds")
        
        # Record times
        jax_times.append(jax_time)
        numpy_times.append(numpy_time)
    
    return jax_times, numpy_times


def plot_results(batch_sizes, jax_times, numpy_times, title, ylabel="Time (seconds)"):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 6))
    
    # Plot times
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, jax_times, 'o-', label='JAX')
    plt.plot(batch_sizes, numpy_times, 'o-', label='NumPy')
    plt.xlabel('Batch Size')
    plt.ylabel(ylabel)
    plt.title(f'{title} - Time')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    speedups = [numpy_times[i] / jax_times[i] for i in range(len(jax_times))]
    plt.plot(batch_sizes, speedups, 'o-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (NumPy time / JAX time)')
    plt.title(f'{title} - Speedup')
    plt.xscale('log', base=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.tight_layout()
    filename = f"{title.lower().replace(' ', '_')}_benchmark.png"
    plt.savefig(filename)
    print(f"Results saved to {filename}")


def main():
    # Define batch sizes to test
    small_batch_sizes = [1, 8, 32, 128]
    large_batch_sizes = [512, 2048]
    extra_large_batch_sizes = [4096, 8192, 16384]
    batch_sizes = small_batch_sizes + large_batch_sizes + extra_large_batch_sizes
    
    # Run benchmarks
    jax_rotation_times, numpy_rotation_times = run_board_rotation_benchmark(batch_sizes)
    jax_block_times, numpy_block_times = run_block_rule_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("\nBoard Rotation:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = numpy_rotation_times[i] / jax_rotation_times[i] if jax_rotation_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, JAX: {jax_rotation_times[i]:.6f}s, NumPy: {numpy_rotation_times[i]:.6f}s, Speedup: {speedup:.2f}x")
    
    print("\nBlock Rule Checking:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = numpy_block_times[i] / jax_block_times[i] if jax_block_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, JAX: {jax_block_times[i]:.6f}s, NumPy: {numpy_block_times[i]:.6f}s, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main() 