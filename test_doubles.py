import gym
import numpy as np
from gym_narde.envs.narde_env import NardeEnv

# Create the environment
env = NardeEnv(debug=True)  # Enable debug mode to see more information
obs = env.reset()

# Set up a scenario with doubles
print("Testing doubles handling in Narde environment")
print("=============================================")

# Force doubles (2, 2)
env.dice = [2, 2, 2, 2]
print(f"Dice rolled: {env.dice[:2]} (doubles, so we have 4 dice: {env.dice})")

# Print initial state
print("\nInitial board state:")
print(env.game.board)
print(f"Current player: {env.current_player}")
print(f"Dice available: {env.dice}")

# Make first move
action = (23, 0)  # Move from position 23 to 21 (using die value 2)
print(f"\nMove 1: {action} (from 23 to 21)")
obs, reward, done, info = env.step(action)

# Print state after first move
print("\nBoard state after move 1:")
print(env.game.board)
print(f"Current player: {env.current_player}")
print(f"Dice available: {env.dice}")
print(f"Game over: {done}")

# Make second move
action = (21, 0)  # Move from position 21 to 19 (using die value 2)
print(f"\nMove 2: {action} (from 21 to 19)")
obs, reward, done, info = env.step(action)

# Print state after second move
print("\nBoard state after move 2:")
print(env.game.board)
print(f"Current player: {env.current_player}")
print(f"Dice available: {env.dice}")
print(f"Game over: {done}")

# Make third move
action = (19, 0)  # Move from position 19 to 17 (using die value 2)
print(f"\nMove 3: {action} (from 19 to 17)")
obs, reward, done, info = env.step(action)

# Print state after third move
print("\nBoard state after move 3:")
print(env.game.board)
print(f"Current player: {env.current_player}")
print(f"Dice available: {env.dice}")
print(f"Game over: {done}")

# Make fourth move
action = (17, 0)  # Move from position 17 to 15 (using die value 2)
print(f"\nMove 4: {action} (from 17 to 15)")
obs, reward, done, info = env.step(action)

# Print state after fourth move
print("\nBoard state after move 4:")
print(env.game.board)
print(f"Current player: {env.current_player}")
print(f"Dice available: {env.dice}")
print(f"Game over: {done}")

# Roll dice for next player
env.dice = [3, 5]
print(f"\nNext player's dice: {env.dice}")

# Make a move with the next player
action = (11, 0)  # Move from position 11 to 8 (using die value 3)
print(f"\nNext player's move: {action} (from 11 to 8)")
obs, reward, done, info = env.step(action)

# Print final state
print("\nBoard state after next player's move:")
print(env.game.board)
print(f"Current player: {env.current_player}")
print(f"Dice available: {env.dice}")
print(f"Game over: {done}")

print("\nTest completed successfully!") 