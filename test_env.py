import gymnasium as gym
import numpy as np
import gym_narde  # Import the gym_narde package
from gym_narde.envs.narde_env import NardeEnv  # Import the environment class directly

# Create and initialize the environment directly
env = NardeEnv()
obs, info = env.reset()

print("Initial observation:", obs)
print("Initial info:", info)

# Try to access dice from the environment
try:
    dice = env.unwrapped.game.dice
    print("Dice from game:", dice)
except (AttributeError, ValueError):
    # Default dice if we can't get it
    dice = [1, 1]
    print("Using default dice:", dice)

# Get valid moves
valid_moves = env.unwrapped.game.get_valid_moves(dice)
print("Valid moves:", valid_moves)

if valid_moves:
    # Take the first valid move
    action = valid_moves[0]
    print("Taking action:", action)
    
    # Step the environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print("Next observation:", next_obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    
    if terminated or truncated:
        print("Episode ended after one step!")
    
    # Try another episode
    print("\nStarting second episode")
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)
    
    # Try to access dice from the environment
    try:
        dice = env.unwrapped.game.dice
        print("Dice from game:", dice)
    except (AttributeError, ValueError):
        # Default dice if we can't get it
        dice = [1, 1]
        print("Using default dice:", dice)
    
    # Get valid moves
    valid_moves = env.unwrapped.game.get_valid_moves(dice)
    print("Valid moves:", valid_moves)

# Close the environment
env.close() 