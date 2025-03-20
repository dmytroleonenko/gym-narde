#!/usr/bin/env python3
"""
Quick Bottleneck Analysis for Game Generation

This script identifies specific bottlenecks in the MuZero game generation process
with a very short run time for quick analysis.
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
logger = logging.getLogger("Bottleneck-Analysis")

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

# Define a simple GameHistory class to store game data
class GameHistory:
    def __init__(self):
        self.observations = []
        self.action_history = []
        self.rewards = []
        
    def append(self, observation, action, reward):
        self.observations.append(observation)
        self.action_history.append(action)
        self.rewards.append(reward)

# Import MuZero modules
from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS, Node
from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde

# Patch MCTS to instrument key functions
original_run = MCTS.run
original_backpropagate = MCTS.backpropagate
original_expand_node = MCTS.expand_node
original_select_child = MCTS.select_child

def instrumented_run(self, observation, valid_actions=None, add_exploration_noise=True):
    with timeit("MCTS.run"):
        return original_run(self, observation, valid_actions, add_exploration_noise)

def instrumented_backpropagate(self, search_path, value, discount):
    with timeit("MCTS.backpropagate"):
        return original_backpropagate(self, search_path, value, discount)

def instrumented_expand_node(self, node, valid_actions, policy):
    with timeit("MCTS.expand_node"):
        return original_expand_node(self, node, valid_actions, policy)

def instrumented_select_child(self, node):
    with timeit("MCTS.select_child"):
        return original_select_child(self, node)

# Patch environment functions
original_step = NardeEnv.step
original_get_valid_moves = Narde.get_valid_moves
original_decode_action = NardeEnv._decode_action

def instrumented_step(self, action):
    with timeit("NardeEnv.step"):
        return original_step(self, action)

def instrumented_get_valid_moves(self, dice=None, head_moves_count=None):
    with timeit("Narde.get_valid_moves"):
        return original_get_valid_moves(self, dice, head_moves_count)

def instrumented_decode_action(self, action):
    with timeit("NardeEnv._decode_action"):
        return original_decode_action(self, action)

def apply_instrumentation():
    """Apply instrumentation to key functions"""
    MCTS.run = instrumented_run
    MCTS.backpropagate = instrumented_backpropagate
    MCTS.expand_node = instrumented_expand_node
    MCTS.select_child = instrumented_select_child
    
    NardeEnv.step = instrumented_step
    Narde.get_valid_moves = instrumented_get_valid_moves
    NardeEnv._decode_action = instrumented_decode_action
    
    logger.info("Instrumentation applied to key functions")

def restore_original_functions():
    """Restore original functions"""
    MCTS.run = original_run
    MCTS.backpropagate = original_backpropagate
    MCTS.expand_node = original_expand_node
    MCTS.select_child = original_select_child
    
    NardeEnv.step = original_step
    Narde.get_valid_moves = original_get_valid_moves
    NardeEnv._decode_action = original_decode_action
    
    logger.info("Original functions restored")

def quick_analyze_game_generation():
    """Quick analysis of bottlenecks in game generation"""
    import torch.multiprocessing as mp
    
    # Apply instrumentation
    apply_instrumentation()
    
    try:
        # Create network with typical parameters from the codebase
        network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
        
        # Enable timing on network forward passes
        original_initial_inference = network.initial_inference
        original_recurrent_inference = network.recurrent_inference
        
        def instrumented_initial_inference(self, observation):
            with timeit("network.initial_inference"):
                return original_initial_inference(self, observation)
        
        def instrumented_recurrent_inference(self, hidden, action):
            with timeit("network.recurrent_inference"):
                return original_recurrent_inference(self, hidden, action)
        
        network.initial_inference = instrumented_initial_inference.__get__(network, MuZeroNetwork)
        network.recurrent_inference = instrumented_recurrent_inference.__get__(network, MuZeroNetwork)
        
        # Create MCTS with the same parameters used in parallel_training.py
        mcts_instance = MCTS(
            network=network,
            num_simulations=5,
            discount=0.99,
            dirichlet_alpha=0.3,
            exploration_fraction=0.25,
            action_space_size=576,
            device="cpu"
        )
        
        # Run a quick game simulation
        def quick_self_play(mcts, num_steps=5):
            env = NardeEnv()
            obs, info = env.reset()
            game_history = GameHistory()
            
            # Only play a few steps
            for _ in range(num_steps):
                # Get valid actions
                valid_actions = env.game.get_valid_moves()
                if not valid_actions:
                    break  # No valid moves left
                
                # Convert moves to indices that MCTS can use
                valid_action_indices = []
                for move in valid_actions:
                    if move[1] == 'off':  # Bearing off
                        from_pos = move[0]
                        valid_action_indices.append(from_pos * 24)  # Dummy index for bearing off
                    else:
                        from_pos, to_pos = move
                        valid_action_indices.append(from_pos * 24 + to_pos)
                
                # Convert observation to torch tensor if needed
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs)
                else:
                    obs_tensor = obs
                
                # Run MCTS - returns policy (visit counts)
                policy = mcts.run(obs_tensor, valid_action_indices)
                
                # Select action based on visit counts
                if isinstance(policy, np.ndarray):
                    # Ensure policy is valid
                    if np.sum(policy) > 0:
                        policy = policy / np.sum(policy)  # Normalize if needed
                    else:
                        policy = np.ones_like(policy) / len(policy)  # Uniform if all zeros
                    
                    selected_action_idx = np.random.choice(len(policy), p=policy)
                    action = (selected_action_idx, 0)  # Regular move by default
                    
                    # Check if it was a bearing off move
                    from_pos = selected_action_idx // 24
                    to_pos = selected_action_idx % 24
                    for move in valid_actions:
                        if move[0] == from_pos and move[1] == 'off':
                            action = (selected_action_idx, 1)  # It's a bearing off move
                            break
                else:
                    action_idx = torch.multinomial(policy, 1).item()
                    action = (action_idx, 0)  # Regular move as fallback
                
                # Execute action
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Store transition (simplified)
                game_history.append(obs, action, reward)
                
                # Next state
                obs = next_obs
                
                if done:
                    break
                    
            return game_history
            
        # Test with a single simulation count
        num_steps = 5
        logger.info(f"\n=== Quick test with {num_steps} steps ===")
        
        # Generate a partial game for analysis
        with timeit(f"quick_self_play_{num_steps}_steps"):
            game_history = quick_self_play(mcts_instance, num_steps)
        
        logger.info(f"Partial game completed in {len(game_history.action_history)} moves")
        
        # Analyze timing results
        logger.info("\n=== TIMING ANALYSIS ===")
        
        # Group functions by category
        categories = {
            "MCTS": ["MCTS.run", "MCTS.backpropagate", "MCTS.expand_node", "MCTS.select_child"],
            "Environment": ["NardeEnv.step", "Narde.get_valid_moves", "NardeEnv._decode_action"],
            "Network": ["network.initial_inference", "network.recurrent_inference"]
        }
        
        # Calculate total game time
        total_key = f"quick_self_play_{num_steps}_steps"
        total_time = timeit.timings[total_key][0] if total_key in timeit.timings else 0
        
        # Print results by category
        for category, functions in categories.items():
            logger.info(f"\n=== {category} Functions ===")
            
            for func in functions:
                if func in timeit.timings and timeit.timings[func]:
                    calls = len(timeit.timings[func])
                    total_func_time = sum(timeit.timings[func])
                    avg_time = total_func_time / calls
                    
                    logger.info(f"{func}: {avg_time:.6f}s avg ({calls} calls, {total_func_time:.4f}s total)")
                    
                    # Calculate percentage of total time
                    if total_time > 0:
                        percentage = total_func_time / total_time * 100
                        logger.info(f"  - {percentage:.2f}% of total time")
        
        # Print the top bottlenecks
        logger.info("\n=== TOP BOTTLENECKS ===")
        all_functions = []
        for funcs in categories.values():
            all_functions.extend(funcs)
        
        function_totals = {}
        for func in all_functions:
            if func in timeit.timings and timeit.timings[func]:
                function_totals[func] = sum(timeit.timings[func])
        
        # Sort by total time
        sorted_functions = sorted(function_totals.items(), key=lambda x: x[1], reverse=True)
        
        for func, func_total_time in sorted_functions[:5]:  # Top 5 bottlenecks
            calls = len(timeit.timings[func])
            avg_time = func_total_time / calls
            
            logger.info(f"{func}: {func_total_time:.4f}s total, {avg_time:.6f}s avg ({calls} calls)")
            
            # Calculate percentage of total time
            if total_time > 0:
                percentage = func_total_time / total_time * 100
                logger.info(f"  - {percentage:.2f}% of total time")
        
        # Optimization suggestions
        logger.info("\n=== OPTIMIZATION SUGGESTIONS ===")
        
        # Check which category is the bottleneck
        mcts_time = sum(function_totals.get(f, 0) for f in categories["MCTS"])
        env_time = sum(function_totals.get(f, 0) for f in categories["Environment"])
        network_time = sum(function_totals.get(f, 0) for f in categories["Network"])
        
        total_analyzed_time = mcts_time + env_time + network_time
        
        if total_analyzed_time > 0:
            logger.info(f"MCTS operations: {mcts_time:.4f}s ({mcts_time/total_analyzed_time*100:.2f}%)")
            logger.info(f"Environment operations: {env_time:.4f}s ({env_time/total_analyzed_time*100:.2f}%)")
            logger.info(f"Network operations: {network_time:.4f}s ({network_time/total_analyzed_time*100:.2f}%)")
            
            # Identify the main bottleneck
            main_bottleneck = max([
                ("MCTS", mcts_time), 
                ("Environment", env_time), 
                ("Network", network_time)
            ], key=lambda x: x[1])[0]
            
            logger.info(f"\nMain bottleneck: {main_bottleneck} operations")
            
            # Specific optimization suggestions based on bottleneck
            if main_bottleneck == "MCTS":
                logger.info("\nMCTS optimization suggestions:")
                logger.info("1. Reduce simulation count to absolute minimum (3-5)")
                logger.info("2. Implement batched MCTS to parallelize tree searches")
                logger.info("3. Optimize node selection and expansion algorithms")
                logger.info("4. Reduce memory allocations in the MCTS loop")
            elif main_bottleneck == "Environment":
                logger.info("\nEnvironment optimization suggestions:")
                logger.info("1. Cache valid moves results when board state doesn't change")
                logger.info("2. Rewrite get_valid_moves using NumPy vectorization")
                logger.info("3. Consider using Numba to compile the Narde environment")
                logger.info("4. Eliminate unnecessary copies and type conversions")
            elif main_bottleneck == "Network":
                logger.info("\nNetwork optimization suggestions:")
                logger.info("1. Reduce network size (smaller hidden dimensions)")
                logger.info("2. Implement batched inference")
                logger.info("3. Use MPS acceleration on Apple Silicon")
                logger.info("4. Consider a smaller, faster representation network")
            
            # Function-specific optimizations for top bottlenecks
            for func, _ in sorted_functions[:2]:
                if "get_valid_moves" in func:
                    logger.info("\nSpecific optimization for get_valid_moves:")
                    logger.info("- Cache valid moves when board state doesn't change")
                    logger.info("- Use NumPy vectorized operations for move generation")
                    logger.info("- Use Numba to compile this function")
                elif "MCTS.run" in func:
                    logger.info("\nSpecific optimization for MCTS.run:")
                    logger.info("- Use a more efficient tree search algorithm")
                    logger.info("- Batch node evaluations")
                    logger.info("- Reduce memory allocations in hot loops")
                elif "initial_inference" in func:
                    logger.info("\nSpecific optimization for initial_inference:")
                    logger.info("- Use a smaller network")
                    logger.info("- Implement MPS acceleration")
                    logger.info("- Pre-allocate tensors to reduce allocations")
        
    finally:
        # Restore original functions
        restore_original_functions()
        
        # Restore network functions
        if 'network' in locals():
            network.initial_inference = original_initial_inference
            network.recurrent_inference = original_recurrent_inference

if __name__ == "__main__":
    mp_ctx = torch.multiprocessing.get_context('spawn')
    p = mp_ctx.Process(target=quick_analyze_game_generation)
    p.start()
    p.join(timeout=60)  # Timeout after 60 seconds
    
    if p.is_alive():
        logger.warning("Analysis taking too long - terminating process")
        p.terminate()
        p.join() 