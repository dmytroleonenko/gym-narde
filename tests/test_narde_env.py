#!/usr/bin/env python3
"""Test cases for the Narde environment."""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import pytest

from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde


def test_environment_initialization():
    """Test that the environment initializes correctly."""
    env = gym.make('Narde-v0')
    assert env is not None
    
    # Check initial observation space shape
    obs = env.reset()[0]
    print(f"Actual observation shape: {obs.shape}")
    assert obs.shape == (28,), f"Expected (28,), got {obs.shape}"
    
    # Check action space
    print(f"Action space: {env.action_space}")
    assert isinstance(env.action_space, gym.spaces.Tuple)
    assert len(env.action_space) == 2
    assert env.action_space[0].n == 576  # Move index space
    assert env.action_space[1].n == 2    # Move type space (regular or bear off)


def test_valid_moves_execution():
    """Test that valid moves can be executed correctly."""
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Reset environment
    obs, info = env.reset()
    
    # Check what's in the info dictionary
    print(f"Info keys: {info.keys()}")
    
    # Get dice directly from the environment
    dice = env_unwrapped.dice
    assert len(dice) > 0, "No dice available in the environment"
    print(f"Dice from environment: {dice}")
    
    # Print board state
    print(f"Initial board: {env_unwrapped.game.board}")
    
    # Get valid moves
    valid_moves = env_unwrapped.game.get_valid_moves()
    assert len(valid_moves) > 0, "No valid moves available"
    print(f"Valid moves: {valid_moves}")
    
    # Test first valid move
    first_move = valid_moves[0]
    print(f"Testing move: {first_move}")
    
    # Convert to action format
    from_pos, to_pos = first_move
    # Manually create the action index: from_pos * 24 + to_pos
    action_idx = from_pos * 24 + to_pos
    move_type = 0  # Regular move
    action = (action_idx, move_type)
    
    print(f"Converted to action: {action}")
    
    # Execute move
    next_obs, reward, done, truncated, info = env.step(action)
    
    print(f"Reward: {reward}, Done: {done}")
    print(f"Board after move: {env_unwrapped.game.board}")
    
    # The move should be valid and return non-negative reward
    assert reward >= 0, f"Valid move resulted in negative reward: {reward}"
    
    # Get new valid moves
    new_valid_moves = env_unwrapped.game.get_valid_moves()
    print(f"New valid moves: {new_valid_moves}")


def test_multiple_valid_moves():
    """Test that multiple valid moves can be executed in sequence."""
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Reset environment
    obs, info = env.reset()
    print(f"Info keys: {info.keys()}")
    
    # Execute up to 5 valid moves
    for i in range(5):
        valid_moves = env_unwrapped.game.get_valid_moves()
        if not valid_moves:
            print(f"No more valid moves after {i} steps")
            break
            
        print(f"Step {i}: {len(valid_moves)} valid moves available")
        print(f"Dice from environment: {env_unwrapped.dice}")
        print(f"Board state: {env_unwrapped.game.board}")
        
        # Select a valid move
        move = valid_moves[0]
        from_pos, to_pos = move
        # Manually create the action index: from_pos * 24 + to_pos
        action_idx = from_pos * 24 + to_pos
        action = (action_idx, 0)  # Regular move
        
        print(f"Executing action: {action} (move: {move})")
        next_obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward}, Done: {done}")
        
        if done:
            print("Episode finished")
            break
            
    assert i > 0, "Could not execute any valid moves"


if __name__ == "__main__":
    # Run tests manually
    print("Testing environment initialization...")
    test_environment_initialization()
    print("\nTesting valid moves execution...")
    test_valid_moves_execution()
    print("\nTesting multiple valid moves...")
    test_multiple_valid_moves()
    print("\nAll tests completed successfully!") 