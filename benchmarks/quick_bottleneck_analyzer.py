#!/usr/bin/env python3
"""
Quick Bottleneck Analysis for Narde Game Generation

This script directly tests key operations in the Narde environment to identify
performance bottlenecks without using the full MCTS implementation.
"""

import os
import time
import torch
import numpy as np
import logging
from collections import defaultdict
from contextlib import contextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Quick-Bottleneck")

# Timing context manager
@contextmanager
def timeit(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} took {elapsed:.4f} seconds")
    timeit.timings[name].append(elapsed)

# Store timing data
timeit.timings = defaultdict(list)

# Import necessary modules
from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde, rotate_board

def analyze_narde_env_bottlenecks():
    """Analyze the bottlenecks in the Narde environment"""
    
    # Test environment initialization
    with timeit("NardeEnv.init"):
        env = NardeEnv()
    
    # Test reset operation
    with timeit("NardeEnv.reset"):
        obs, info = env.reset()
    
    # Get valid moves multiple times to test performance
    num_valid_move_tests = 100
    logger.info(f"Testing get_valid_moves {num_valid_move_tests} times...")
    for i in range(num_valid_move_tests):
        with timeit("Narde.get_valid_moves"):
            valid_moves = env.game.get_valid_moves()
    
    # Test step function with different moves
    max_steps = 20
    step_count = 0
    
    logger.info(f"Testing game play with up to {max_steps} steps...")
    while step_count < max_steps:
        # Get valid moves
        with timeit("Narde.get_valid_moves"):
            valid_moves = env.game.get_valid_moves()
        
        if not valid_moves:
            # Roll dice again to get valid moves
            env.game.dice = [1, 2]  # Force some dice to continue
            with timeit("Narde.get_valid_moves"):
                valid_moves = env.game.get_valid_moves()
            
            if not valid_moves:
                # If still no valid moves, rotate board and try again
                with timeit("Narde.rotate_board_for_next_player"):
                    env.game.rotate_board_for_next_player()
                continue
                
        # Select a move
        move = valid_moves[0]  # Take first valid move for simplicity
        
        # Convert valid move to action
        if move[1] == 'off':  # Bearing off
            from_pos = move[0]
            action = (from_pos * 24, 1)  # Use move type 1 for bearing off
        else:
            from_pos, to_pos = move
            action = (from_pos * 24 + to_pos, 0)  # Regular move
            
        # Execute step
        with timeit("NardeEnv.step"):
            next_obs, reward, done, truncated, info = env.step(action)
        
        step_count += 1
        
        if done:
            break
    
    logger.info(f"Completed {step_count} steps")
    
    # Test board rotation performance
    num_rotation_tests = 1000
    logger.info(f"Testing board rotation {num_rotation_tests} times...")
    board_before = env.game.board.copy()
    with timeit("Narde.rotate_board"):
        for _ in range(num_rotation_tests):
            rotated = rotate_board(board_before)
    
    # Test board checking functions
    num_game_over_tests = 1000
    logger.info(f"Testing game_over check {num_game_over_tests} times...")
    with timeit("Narde.is_game_over"):
        for _ in range(num_game_over_tests):
            is_over = env.game.is_game_over()
    
    # Analyze and print results
    logger.info("\n=== TIMING ANALYSIS ===")
    
    # Group similar operations
    categories = {
        "Initialization": ["NardeEnv.init", "NardeEnv.reset"],
        "Move Generation": ["Narde.get_valid_moves"],
        "Game Execution": ["NardeEnv.step"],
        "Board Operations": ["Narde.rotate_board", "Narde.is_game_over", "Narde.rotate_board_for_next_player"]
    }
    
    # Print results by category
    total_times = {}
    
    for category, functions in categories.items():
        logger.info(f"\n=== {category} ===")
        category_total = 0
        
        for func in functions:
            if func in timeit.timings and timeit.timings[func]:
                calls = len(timeit.timings[func])
                total_func_time = sum(timeit.timings[func])
                avg_time = total_func_time / calls
                
                logger.info(f"{func}: {avg_time:.6f}s avg ({calls} calls, {total_func_time:.4f}s total)")
                category_total += total_func_time
        
        total_times[category] = category_total
        logger.info(f"Total {category} time: {category_total:.4f}s")
    
    # Calculate overall time
    total_time = sum(total_times.values())
    logger.info(f"\nTotal analyzed time: {total_time:.4f}s")
    
    # Print percentage breakdowns
    logger.info("\n=== PERCENTAGE BREAKDOWN ===")
    for category, time in total_times.items():
        percentage = (time / total_time) * 100 if total_time > 0 else 0
        logger.info(f"{category}: {percentage:.2f}%")
    
    # Identify the bottleneck
    if total_time > 0:
        bottleneck = max(total_times.items(), key=lambda x: x[1])
        logger.info(f"\nMain bottleneck: {bottleneck[0]} operations ({bottleneck[1]:.4f}s, {(bottleneck[1]/total_time)*100:.2f}%)")
        
        # Calculate operations per second for key operations
        logger.info("\n=== OPERATIONS PER SECOND ===")
        if "Narde.get_valid_moves" in timeit.timings:
            vm_calls = len(timeit.timings["Narde.get_valid_moves"])
            vm_time = sum(timeit.timings["Narde.get_valid_moves"])
            if vm_time > 0:
                logger.info(f"get_valid_moves: {vm_calls / vm_time:.2f} ops/s")
                
        if "NardeEnv.step" in timeit.timings:
            step_calls = len(timeit.timings["NardeEnv.step"])
            step_time = sum(timeit.timings["NardeEnv.step"])
            if step_time > 0:
                logger.info(f"step: {step_calls / step_time:.2f} ops/s")
                
        if "Narde.rotate_board" in timeit.timings:
            rot_calls = len(timeit.timings["Narde.rotate_board"])
            rot_time = sum(timeit.timings["Narde.rotate_board"])
            if rot_time > 0:
                logger.info(f"rotate_board: {rot_calls / rot_time:.2f} ops/s")
        
        # Provide specific optimization suggestions
        logger.info("\n=== OPTIMIZATION SUGGESTIONS ===")
        
        if bottleneck[0] == "Move Generation":
            logger.info("Move Generation Optimizations:")
            logger.info("1. Cache valid moves when board state doesn't change")
            logger.info("2. Use NumPy vectorized operations for move generation")
            logger.info("3. Consider using Numba to JIT-compile get_valid_moves")
            logger.info("4. Pre-allocate arrays to reduce memory allocations")
            logger.info("5. Implement a more efficient algorithm for move generation")
        elif bottleneck[0] == "Game Execution":
            logger.info("Game Execution Optimizations:")
            logger.info("1. Optimize the step function to reduce overhead")
            logger.info("2. Minimize copying of game state")
            logger.info("3. Use in-place operations where possible")
            logger.info("4. Batch environment steps for better throughput")
        elif bottleneck[0] == "Board Operations":
            logger.info("Board Operations Optimizations:")
            logger.info("1. Optimize rotate_board using more efficient NumPy operations")
            logger.info("2. Use in-place operations to avoid unnecessary copies")
            logger.info("3. Cache results of commonly used operations")
        elif bottleneck[0] == "Initialization":
            logger.info("Initialization Optimizations:")
            logger.info("1. Reuse environments instead of creating new ones")
            logger.info("2. Optimize reset function to reduce overhead")
            logger.info("3. Pre-allocate arrays during initialization")
            
    # Estimate potential game generation speed
    logger.info("\n=== ESTIMATED GAME GENERATION SPEED ===")
    
    # Assume a typical game is ~50 steps
    typical_game_steps = 50
    
    # Calculate time per step
    if "NardeEnv.step" in timeit.timings and len(timeit.timings["NardeEnv.step"]) > 0:
        avg_step_time = sum(timeit.timings["NardeEnv.step"]) / len(timeit.timings["NardeEnv.step"])
        
        # Calculate time for valid move generation per step
        avg_valid_moves_time = 0
        if "Narde.get_valid_moves" in timeit.timings and len(timeit.timings["Narde.get_valid_moves"]) > 0:
            avg_valid_moves_time = sum(timeit.timings["Narde.get_valid_moves"]) / len(timeit.timings["Narde.get_valid_moves"])
        
        # Typical time per game step (including getting valid moves)
        time_per_step = avg_step_time + avg_valid_moves_time
        
        # Estimate MCTS overhead based on simulations
        # Assume each simulation is roughly 2x the cost of a step + valid moves
        mcts_simulations = 5  # Typical low value used in optimized setups
        mcts_overhead_per_step = time_per_step * 2 * mcts_simulations
        
        # Total time per step with MCTS
        total_time_per_step = time_per_step + mcts_overhead_per_step
        
        # Calculate game generation stats
        game_time = total_time_per_step * typical_game_steps
        games_per_second = 1.0 / game_time if game_time > 0 else 0
        
        logger.info(f"Estimated time per game step: {time_per_step:.6f}s")
        logger.info(f"Estimated MCTS overhead per step ({mcts_simulations} sims): {mcts_overhead_per_step:.6f}s")
        logger.info(f"Estimated total time per step with MCTS: {total_time_per_step:.6f}s")
        logger.info(f"Estimated time for a {typical_game_steps}-step game: {game_time:.4f}s")
        logger.info(f"Estimated game generation speed: {games_per_second:.2f} games/s")
        logger.info(f"With 8 parallel workers: {games_per_second * 8:.2f} games/s")
        
        # Potential optimizations impact
        logger.info("\nPotential optimization impacts:")
        
        # If we optimize get_valid_moves to be 5x faster
        optimized_valid_moves_time = avg_valid_moves_time / 5
        optimized_time_per_step = avg_step_time + optimized_valid_moves_time
        optimized_mcts_overhead = optimized_time_per_step * 2 * mcts_simulations
        optimized_total_time_per_step = optimized_time_per_step + optimized_mcts_overhead
        optimized_game_time = optimized_total_time_per_step * typical_game_steps
        optimized_games_per_second = 1.0 / optimized_game_time if optimized_game_time > 0 else 0
        
        logger.info(f"With 5x faster get_valid_moves: {optimized_games_per_second:.2f} games/s")
        logger.info(f"With 5x faster get_valid_moves + 8 workers: {optimized_games_per_second * 8:.2f} games/s")

if __name__ == "__main__":
    analyze_narde_env_bottlenecks() 