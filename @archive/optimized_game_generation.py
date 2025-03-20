#!/usr/bin/env python3
import os
import time
import random
import numpy as np
import logging
import multiprocessing as mp
from functools import lru_cache
import concurrent.futures
from tqdm import tqdm

try:
    # Try to import numba if available
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Import necessary modules
from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('OptimizedGameGeneration')

# Create a simple game history class to store game data
class GameHistory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.to_play_history = []
        self.child_visits = []
        
    def append(self, observation, action, reward, to_play=1, child_visits=None):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.to_play_history.append(to_play)
        if child_visits is not None:
            self.child_visits.append(child_visits)
            
# Optimization 1: Cache valid moves using lru_cache
@lru_cache(maxsize=1024)
def cached_get_valid_moves(board_tuple, dice_tuple):
    """Cached version of get_valid_moves to avoid recalculating for the same board and dice state."""
    # Convert tuples back to np.array for Narde
    board = np.array(board_tuple)
    
    # Create a new Narde game with this board
    game = Narde()
    game.board = board
    
    # Convert dice tuple to list to avoid NumPy array boolean check issues
    dice = list(dice_tuple)
    
    # Pass the dice explicitly to get_valid_moves
    return game.get_valid_moves(dice=dice)

# Optimization 2: If numba is available, create JIT-compiled version of key functions
if HAS_NUMBA:
    @numba.jit(nopython=True)
    def numba_rotate_board(board):
        """JIT-compiled version of board rotation."""
        rotated_board = np.zeros_like(board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                rotated_board[i, j] = -board[board.shape[0]-i-1, board.shape[1]-j-1]
        return rotated_board
else:
    def numba_rotate_board(board):
        """Fallback if numba is not available."""
        return -np.flip(board, axis=(0, 1))

def optimized_play_game(env_seed=42, max_steps=50, use_cache=True):
    """
    Play a game with optimized valid moves calculation.
    Returns time taken and number of moves.
    """
    start_time = time.time()
    env = NardeEnv()
    
    # Set seed and reset
    reset_result = env.reset(seed=env_seed)
    if isinstance(reset_result, tuple):
        observation = reset_result[0]
    else:
        observation = reset_result
        
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    # Create game history
    game_history = GameHistory()
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # Get valid moves using cache if enabled
        if use_cache:
            board_tuple = tuple(env.game.board)
            dice_tuple = tuple(env.dice)
            valid_moves = cached_get_valid_moves(board_tuple, dice_tuple)
        else:
            valid_moves = env.game.get_valid_moves()
            
        # Select random action if valid moves exist
        if len(valid_moves) > 0:
            move = random.choice(valid_moves)
            
            # Convert to NardeEnv action format
            from_pos, to_pos = move
            move_type = 1 if to_pos == 'off' else 0
            
            if move_type == 0:
                # Regular move
                move_index = from_pos * 24 + to_pos
            else:
                # Bearing off - use from_pos directly
                move_index = from_pos
                
            # Create action tuple
            action_tuple = (move_index, move_type)
            
            # Execute action
            step_result = env.step(action_tuple)
            
            # Handle step result format
            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_observation, reward, done, info = step_result
                
            # Store in game history
            game_history.append(observation, action_tuple, reward)
            
            # Update for next iteration
            observation = next_observation
        else:
            done = True
            
        step_count += 1
    
    elapsed_time = time.time() - start_time
    return elapsed_time, len(game_history.actions)

def worker_process(worker_id, num_games, seeds=None, max_moves=100):
    """Worker process for parallel game generation."""
    if seeds is None:
        seeds = list(range(worker_id * num_games, (worker_id + 1) * num_games))
    
    start_time = time.time()
    total_moves = 0
    
    for i, seed in enumerate(seeds):
        elapsed_time, num_moves = optimized_play_game(env_seed=seed, max_steps=max_moves)
        total_moves += num_moves
        
        # Log progress
        if (i + 1) % 5 == 0 or i == 0:
            completed = (i + 1) / len(seeds) * 100
            logger.debug(f"Worker {worker_id}: {completed:.1f}% complete, avg {total_moves/(i+1):.1f} moves/game")
    
    elapsed_time = time.time() - start_time
    avg_moves = total_moves / num_games if num_games > 0 else 0
    speed = num_games / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Worker {worker_id}: Generated {num_games} games in {elapsed_time:.2f}s "
                f"({speed:.2f} games/s, avg {avg_moves:.1f} moves/game)")
    
    return num_games, elapsed_time, total_moves

def generate_games_parallel(num_games, num_workers=None, max_moves=100):
    """Generate games in parallel using multiple worker processes."""
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    num_workers = min(num_workers, num_games)  # No need for more workers than games
    
    logger.info(f"Generating {num_games} games using {num_workers} workers...")
    
    # Split games among workers
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers
    
    # Create seed ranges for each worker
    seeds = list(range(num_games))
    random.shuffle(seeds)  # Shuffle to avoid potential seed-related biases
    
    start_time = time.time()
    
    # Process games in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit jobs for each worker
        futures = []
        for i in range(num_workers):
            # Distribute the remainder games
            worker_games = games_per_worker + (1 if i < remainder else 0)
            worker_seeds = seeds[i * games_per_worker + min(i, remainder):
                              (i+1) * games_per_worker + min(i+1, remainder)]
            
            # Submit the job
            futures.append(executor.submit(worker_process, i, worker_games, worker_seeds, max_moves))
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Worker process error: {e}")
    
    # Calculate statistics
    total_games = sum(num_games for num_games, _, _ in results)
    total_time = time.time() - start_time
    total_moves = sum(total_moves for _, _, total_moves in results)
    
    # Calculate overall speed
    games_per_second = total_games / total_time if total_time > 0 else 0
    avg_moves_per_game = total_moves / total_games if total_games > 0 else 0
    
    logger.info(f"Generated {total_games} games in {total_time:.2f}s "
               f"({games_per_second:.2f} games/s, avg {avg_moves_per_game:.1f} moves/game)")
    
    return games_per_second, avg_moves_per_game

def run_benchmark(num_games=25, num_workers_list=None, max_moves=50):
    """Run benchmarks with different worker counts."""
    if num_workers_list is None:
        # Test with 1, 2, 4, 8 workers (or up to CPU count)
        max_workers = min(mp.cpu_count(), 8)
        num_workers_list = [1] + [w for w in [2, 4, 8] if w <= max_workers]
    
    results = {}
    
    logger.info(f"=== BENCHMARK: Generating {num_games} games with up to {max_moves} moves each ===")
    
    for num_workers in num_workers_list:
        logger.info(f"\nTesting with {num_workers} workers:")
        start_time = time.time()
        
        game_histories = generate_games_parallel(num_games, num_workers, max_moves)
        
        elapsed = time.time() - start_time
        games_per_second = game_histories[0]
        avg_moves = game_histories[1]
        
        # Calculate statistics about the games
        total_moves = total_moves = sum(total_moves for _, _, total_moves in game_histories)
        max_game_moves = total_moves
        
        results[num_workers] = {
            'time': elapsed,
            'games_per_second': games_per_second,
            'num_games': len(game_histories),
            'total_moves': total_moves,
            'avg_moves': avg_moves,
            'max_moves': max_game_moves
        }
        
        logger.info(f"Workers: {num_workers}, Time: {elapsed:.2f}s, Speed: {games_per_second:.2f} games/s")
        logger.info(f"Game stats - Avg moves: {avg_moves:.1f}, Max moves: {max_game_moves}, Total moves: {total_moves}")
        
    logger.info("\n=== BENCHMARK RESULTS ===")
    for num_workers, result in results.items():
        logger.info(f"Workers: {num_workers}, Speed: {result['games_per_second']:.2f} games/s, " 
                   f"Avg moves: {result['avg_moves']:.1f}")
    
    # Calculate speedup relative to single worker
    if 1 in results:
        base_speed = results[1]['games_per_second']
        logger.info("\n=== SPEEDUP RELATIVE TO SINGLE WORKER ===")
        for num_workers, result in results.items():
            if num_workers > 1:
                speedup = result['games_per_second'] / base_speed
                efficiency = speedup / num_workers
                logger.info(f"Workers: {num_workers}, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}")
    
    return results

def test_cache_impact(num_games=20, max_moves=100):
    """Test the impact of caching on game generation speed."""
    logger.info(f"Testing cache impact with {num_games} games...")
    
    # Clear cache first
    cached_get_valid_moves.cache_clear()
    
    # Test without cache
    logger.info("Testing without cache...")
    start_no_cache = time.time()
    total_moves_no_cache = 0
    for i in range(num_games):
        _, num_moves = optimized_play_game(env_seed=i, max_steps=max_moves, use_cache=False)
        total_moves_no_cache += num_moves
    time_no_cache = time.time() - start_no_cache
    
    # Calculate stats
    avg_moves_no_cache = total_moves_no_cache / num_games
    speed_no_cache = num_games / time_no_cache
    logger.info(f"Without cache: {speed_no_cache:.2f} games/s with average {avg_moves_no_cache:.1f} moves/game")
    
    # Clear cache before second test
    cached_get_valid_moves.cache_clear()
    
    # Test with cache
    logger.info("Testing with cache...")
    start_with_cache = time.time()
    total_moves_with_cache = 0
    for i in range(num_games):
        _, num_moves = optimized_play_game(env_seed=i, max_steps=max_moves, use_cache=True)
        total_moves_with_cache += num_moves
    time_with_cache = time.time() - start_with_cache
    
    # Calculate stats
    avg_moves_with_cache = total_moves_with_cache / num_games
    speed_with_cache = num_games / time_with_cache
    logger.info(f"With cache: {speed_with_cache:.2f} games/s with average {avg_moves_with_cache:.1f} moves/game")
    
    # Calculate speedup
    speedup = speed_with_cache / speed_no_cache if speed_no_cache > 0 else 0
    logger.info(f"Cache speedup: {speedup:.2f}x")
    
    return speed_no_cache, speed_with_cache, speedup

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('OptimizedGameGeneration')
    
    # Run a quick test
    logger.info("Running a quick test of the optimized game generation...")
    start_time = time.time()
    elapsed_time, num_moves = optimized_play_game(env_seed=42, max_steps=50)
    logger.info(f"Test game completed in {elapsed_time:.4f}s with {num_moves} moves")
    
    # Test impact of caching
    logger.info("\nTesting impact of caching on game generation speed...")
    speed_no_cache, speed_with_cache, speedup = test_cache_impact(num_games=20, max_moves=100)
    
    # Run benchmarks for different numbers of workers
    logger.info("\nRunning parallel benchmarks...")
    
    # Define worker configurations to test
    worker_configs = [1, 2, 4, 8]
    benchmark_results = {}
    
    # Run benchmarks with small number of games
    num_benchmark_games = 25
    
    for num_workers in worker_configs:
        logger.info(f"\nBenchmarking with {num_workers} worker(s)...")
        speed, avg_moves = generate_games_parallel(num_benchmark_games, num_workers)
        benchmark_results[num_workers] = (speed, avg_moves)
    
    # Calculate theoretical best speed
    base_speed = benchmark_results[1][0] if 1 in benchmark_results else 0
    
    # Output final summary
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info(f"Cache speedup: {speedup:.2f}x")
    
    # Output best configuration
    if benchmark_results:
        logger.info("\nParallel benchmark results:")
        logger.info(f"{'Workers':<10} {'Speed (games/s)':<20} {'Avg Moves':<15} {'Speedup':<10} {'Efficiency':<10}")
        
        for workers, (speed, avg_moves) in sorted(benchmark_results.items()):
            speedup = speed / base_speed if base_speed > 0 else 0
            efficiency = speedup / workers if workers > 0 else 0
            efficiency_pct = efficiency * 100
            
            logger.info(f"{workers:<10} {speed:<20.2f} {avg_moves:<15.1f} "
                        f"{speedup:<10.2f}x {efficiency_pct:<10.0f}%")
        
        # Identify best configuration
        best_workers = max(benchmark_results.keys(), key=lambda w: benchmark_results[w][0])
        best_speed = benchmark_results[best_workers][0]
        
        logger.info(f"\nBest configuration: {best_workers} worker(s) at {best_speed:.2f} games/s")
        
        # Estimate optimal game generation rate with caching
        estimated_cached_speed = best_speed * speedup
        logger.info(f"Estimated optimal game generation rate (using {best_workers} workers): {estimated_cached_speed:.2f} games/s")
    
    logger.info("\nOptimized game generation benchmark complete") 