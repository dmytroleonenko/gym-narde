#!/usr/bin/env python3
"""Test the Narde environment against the Gymnasium interface standards."""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import pytest


def test_gym_make():
    """Test that we can create the environment using gym.make."""
    env = gym.make('Narde-v0')
    assert env is not None
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'close')


def test_reset_interface():
    """Test that reset follows the Gymnasium interface."""
    env = gym.make('Narde-v0')
    
    # Reset should return an observation and info
    result = env.reset()
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    obs, info = result
    
    # Check observation
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (28,)
    
    # Check info
    assert isinstance(info, dict)


def test_step_interface():
    """Test that step follows the Gymnasium interface."""
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Reset environment
    obs, info = env.reset()
    
    # Get a valid move
    valid_moves = env_unwrapped.game.get_valid_moves()
    assert len(valid_moves) > 0
    
    # Convert to action
    from_pos, to_pos = valid_moves[0]
    action_idx = from_pos * 24 + to_pos
    action = (action_idx, 0)  # Regular move
    
    # Step should return observation, reward, terminated, truncated, info
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5
    
    next_obs, reward, terminated, truncated, info = result
    
    # Check types
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Observation should have the same shape
    assert next_obs.shape == obs.shape


def test_action_space():
    """Test that the action space follows Gymnasium standards."""
    env = gym.make('Narde-v0')
    
    # Action space should be a Tuple
    assert isinstance(env.action_space, gym.spaces.Tuple)
    
    # With two Discrete components
    assert len(env.action_space) == 2
    assert isinstance(env.action_space[0], gym.spaces.Discrete)
    assert isinstance(env.action_space[1], gym.spaces.Discrete)
    
    # With the correct sizes
    assert env.action_space[0].n == 576
    assert env.action_space[1].n == 2
    
    # Sample should produce a valid action
    action = env.action_space.sample()
    assert isinstance(action, tuple)
    assert len(action) == 2
    assert 0 <= action[0] < 576
    assert 0 <= action[1] < 2


def test_observation_space():
    """Test that the observation space follows Gymnasium standards."""
    env = gym.make('Narde-v0')
    
    # Observation space should be a Box
    assert isinstance(env.observation_space, gym.spaces.Box)
    
    # With the correct shape
    assert env.observation_space.shape == (28,)
    
    # Sample should produce a valid observation
    obs = env.observation_space.sample()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (28,)
    
    # Check that bounds are respected
    assert np.all(obs >= env.observation_space.low)
    assert np.all(obs <= env.observation_space.high)


if __name__ == "__main__":
    test_gym_make()
    test_reset_interface()
    test_step_interface()
    test_action_space()
    test_observation_space()
    print("All Gymnasium interface tests passed!") 