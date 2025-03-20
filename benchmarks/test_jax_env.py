#!/usr/bin/env python3
"""
Test script for the JAX-based Narde environment.
This validates that the JAX implementation works and compares it with the original numpy version.
"""

import gymnasium as gym
import gym_narde  # Import to register environments
import jax
import jax.numpy as jnp
import numpy as np
import time
import random
from contextlib import contextmanager

# Configure JAX hardware acceleration
if jax.default_backend() != "cpu":
    print(f"JAX is using hardware acceleration: {jax.default_backend()}")
else:
    print("JAX is using CPU. For better performance on Apple Silicon, install jax[metal]")


@contextmanager
def timer(name):
    """Simple context manager for timing code blocks."""
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.6f} seconds")


def test_environment_compatibility():
    """Test that both environments produce similar results for the same actions."""
    print("\n=== Testing Environment Compatibility ===")

    # Set the same seed for both environments
    seed = 42
    
    # Create both environments
    numpy_env = gym.make('Narde-v0')
    jax_env = gym.make('Narde-jax-v0')
    
    # Reset environments with the same seed
    numpy_obs, numpy_info = numpy_env.reset(seed=seed)
    jax_obs, jax_info = jax_env.reset(seed=seed)
    
    # Compare initial observations
    print(f"NumPy dice: {numpy_info['dice']}")
    print(f"JAX dice: {jax_info['dice']}")
    
    # Set the same dice for both environments to ensure fair comparison
    numpy_unwrapped = numpy_env.unwrapped
    jax_unwrapped = jax_env.unwrapped
    
    jax_unwrapped.dice = numpy_unwrapped.dice.copy()
    jax_unwrapped.game.dice = numpy_unwrapped.dice.copy()
    
    # Compare board states
    numpy_board = numpy_obs[:24]
    jax_board = jax_obs[:24].tolist()  # Convert JAX array to list for comparison
    
    print(f"Board states match: {np.allclose(numpy_board, jax_board)}")
    
    # Get valid moves from both environments
    numpy_valid_moves = numpy_unwrapped.game.get_valid_moves()
    jax_valid_moves = jax_unwrapped.game.get_valid_moves()
    
    print(f"Valid moves match: {numpy_valid_moves == jax_valid_moves}")
    
    # Take the same action in both environments
    if numpy_valid_moves:
        # Choose a random valid move
        move = random.choice(numpy_valid_moves)
        from_pos, to_pos = move
        
        # Encode the move for the step function
        move_index = from_pos * 24
        move_type = 0
        
        if to_pos == 'off':
            move_type = 1
        else:
            move_index += to_pos
        
        action = (move_index, move_type)
        print(f"Taking action: {action}, which corresponds to move: {move}")
        
        # Take the step
        numpy_next_obs, numpy_reward, numpy_term, numpy_trunc, numpy_info = numpy_env.step(action)
        jax_next_obs, jax_reward, jax_term, jax_trunc, jax_info = jax_env.step(action)
        
        # Compare results
        print(f"Rewards match: {numpy_reward == jax_reward}")
        print(f"Termination match: {numpy_term == jax_term}")
        print(f"Truncation match: {numpy_trunc == jax_trunc}")
        
        # Compare next board states
        numpy_next_board = numpy_next_obs[:24]
        jax_next_board = jax_next_obs[:24].tolist()
        print(f"Next board states match: {np.allclose(numpy_next_board, jax_next_board)}")
    else:
        print("No valid moves to test.")
    
    # Clean up
    numpy_env.close()
    jax_env.close()


def benchmark_environment_performance(num_episodes=10, max_steps=100):
    """Benchmark the performance of both environments."""
    print("\n=== Benchmarking Environment Performance ===")
    
    seed = 42
    
    # Benchmark NumPy version
    with timer("NumPy Environment - 10 Episodes"):
        numpy_env = gym.make('Narde-v0')
        numpy_unwrapped = numpy_env.unwrapped
        for episode in range(num_episodes):
            obs, _ = numpy_env.reset(seed=seed+episode)
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # Get valid moves
                valid_moves = numpy_unwrapped.game.get_valid_moves()
                
                # If no valid moves, end episode
                if not valid_moves:
                    break
                
                # Choose random move
                move = random.choice(valid_moves)
                from_pos, to_pos = move
                
                # Encode move
                move_index = from_pos * 24
                move_type = 0
                
                if to_pos == 'off':
                    move_type = 1
                else:
                    move_index += to_pos
                
                action = (move_index, move_type)
                
                # Take step
                obs, reward, terminated, truncated, _ = numpy_env.step(action)
                done = terminated or truncated
                steps += 1
        
        numpy_env.close()
    
    # Benchmark JAX version
    with timer("JAX Environment - 10 Episodes"):
        jax_env = gym.make('Narde-jax-v0')
        jax_unwrapped = jax_env.unwrapped
        for episode in range(num_episodes):
            obs, _ = jax_env.reset(seed=seed+episode)
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # Get valid moves
                valid_moves = jax_unwrapped.game.get_valid_moves()
                
                # If no valid moves, end episode
                if not valid_moves:
                    break
                
                # Choose random move
                move = random.choice(valid_moves)
                from_pos, to_pos = move
                
                # Encode move
                move_index = from_pos * 24
                move_type = 0
                
                if to_pos == 'off':
                    move_type = 1
                else:
                    move_index += to_pos
                
                action = (move_index, move_type)
                
                # Take step
                obs, reward, terminated, truncated, _ = jax_env.step(action)
                done = terminated or truncated
                steps += 1
        
        jax_env.close()


def benchmark_batch_states_generation(batch_size=1024):
    """Benchmark creating batches of states."""
    print("\n=== Benchmarking Batch State Generation ===")
    
    # Generate states with NumPy
    with timer(f"NumPy - Generate {batch_size} states"):
        numpy_env = gym.make('Narde-v0')
        numpy_states = []
        
        for _ in range(batch_size):
            obs, _ = numpy_env.reset(seed=random.randint(0, 10000))
            numpy_states.append(obs)
        
        numpy_states = np.array(numpy_states)
        numpy_env.close()
    
    # Generate states with JAX
    with timer(f"JAX - Generate {batch_size} states"):
        jax_env = gym.make('Narde-jax-v0')
        jax_states = []
        
        for _ in range(batch_size):
            obs, _ = jax_env.reset(seed=random.randint(0, 10000))
            jax_states.append(obs)
        
        jax_states = jnp.array(jax_states)
        jax_env.close()
    
    print(f"NumPy batch shape: {numpy_states.shape}")
    print(f"JAX batch shape: {jax_states.shape}")


if __name__ == "__main__":
    # Display JAX devices
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"Available JAX devices: {jax.devices()}")
    
    # Run tests
    test_environment_compatibility()
    benchmark_environment_performance()
    benchmark_batch_states_generation(batch_size=2048) 