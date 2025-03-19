import gymnasium as gym
import numpy as np
import gym_narde
from gym_narde.envs.narde_env import NardeEnv

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

# Print board state
board = env.unwrapped.game.board
print("Board state:", board)

if valid_moves:
    for i, move in enumerate(valid_moves):
        print(f"\nTrying move {i+1}: {move}")
        
        # Create a copy of the environment to test each move
        import copy
        test_env = copy.deepcopy(env)
        
        # Execute move
        next_obs, reward, terminated, truncated, info = test_env.step(move)
        
        print("  Result:")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")
        
        # Check if the move was invalid
        if 'invalid_move' in info and info['invalid_move']:
            print("  ❌ Move was marked as invalid!")
            
            # Try to understand why it's invalid
            try:
                # Check if the move is in valid moves according to the game
                game_valid_moves = test_env.unwrapped.game.get_valid_moves(dice)
                if move in game_valid_moves:
                    print("  ⚠️ But the move is in the list of valid moves!")
                
                # Check the move format
                from_pos, to_pos = move
                print(f"  Move format - from: {from_pos} (type: {type(from_pos)}), to: {to_pos} (type: {type(to_pos)})")
                
                # If it's a numpy int, try with a regular int
                if isinstance(from_pos, np.integer):
                    int_move = (int(from_pos), int(to_pos) if to_pos != 'off' else 'off')
                    print(f"  Trying with regular integers: {int_move}")
                    next_obs, reward, terminated, truncated, info = test_env.step(int_move)
                    print(f"  Result with regular integers: {info}")
            except Exception as e:
                print(f"  Error analyzing invalid move: {e}")
        else:
            # Print new board state if move was valid
            print("  ✅ Move was valid!")
            board = test_env.unwrapped.game.board
            print(f"  New board state: {board}")
    
    # Execute first move on the real environment
    print("\nExecuting first move on real environment:", valid_moves[0])
    next_obs, reward, terminated, truncated, info = env.step(valid_moves[0])
    print("Next observation:", next_obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    
    if not terminated and not truncated:
        print("\nGame continues to second step")
        # Get new dice and valid moves
        try:
            dice = env.unwrapped.game.dice
            print("New dice:", dice)
        except (AttributeError, ValueError):
            dice = [1, 1]
            print("Using default dice:", dice)
        
        valid_moves = env.unwrapped.game.get_valid_moves(dice)
        print("New valid moves:", valid_moves)
        
        if valid_moves:
            # Take another move
            print("\nTaking second move:", valid_moves[0])
            next_obs, reward, terminated, truncated, info = env.step(valid_moves[0])
            print("Result after second move:")
            print("Reward:", reward)
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Info:", info)

# Close the environment
env.close() 