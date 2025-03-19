import gymnasium as gym
import numpy as np
import gym_narde
from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde

def test_valid_moves_execution():
    """
    Test that moves reported as valid by get_valid_moves() can actually be executed without being marked invalid.
    This is to verify the correctness of the environment's move validation.
    """
    # Create the environment
    env = NardeEnv()
    obs, info = env.reset()
    
    print("\n=== Testing Valid Moves Execution ===")
    
    # Get dice and valid moves
    dice = env.unwrapped.game.dice
    print(f"Dice roll: {dice}")
    
    valid_moves = env.unwrapped.game.get_valid_moves(dice)
    print(f"Valid moves returned by environment: {valid_moves}")
    
    # Print the board state
    board = env.unwrapped.game.board
    print(f"Board state:\n{board}")
    
    if not valid_moves:
        print("No valid moves returned by the environment.")
        return
    
    print("\n--- Testing each valid move ---")
    
    # Test each reported valid move
    all_moves_valid = True
    for i, move in enumerate(valid_moves):
        print(f"\nTesting move {i+1}: {move}")
        
        # Make a copy of the environment to test this move
        import copy
        test_env = copy.deepcopy(env)
        
        # Try the move
        next_obs, reward, terminated, truncated, info = test_env.step(move)
        
        # Check if the move was invalid
        if 'invalid_move' in info and info['invalid_move']:
            all_moves_valid = False
            print(f"  ❌ ERROR: Move {move} was reported as valid but was marked invalid when executed!")
            print(f"  Info: {info}")
            
            # Try to debug why it's invalid
            from_pos, to_pos = move
            
            # Check move format and type
            print(f"  Move format check:")
            print(f"  - from_pos: {from_pos} (type: {type(from_pos)})")
            print(f"  - to_pos: {to_pos} (type: {type(to_pos)})")
            
            # Try with regular int instead of numpy int
            if isinstance(from_pos, np.integer) or isinstance(to_pos, np.integer):
                int_from_pos = int(from_pos)
                int_to_pos = int(to_pos) if to_pos != 'off' else 'off'
                int_move = (int_from_pos, int_to_pos)
                
                print(f"  Trying with regular integers: {int_move}")
                next_obs, reward, terminated, truncated, info = test_env.step(int_move)
                if 'invalid_move' in info and info['invalid_move']:
                    print(f"  ❌ Still invalid with regular integers: {info}")
                else:
                    print(f"  ✓ Valid with regular integers!")
                    
            # Try to see if the move is still in the list of valid moves
            new_valid_moves = test_env.unwrapped.game.get_valid_moves(dice)
            if move in new_valid_moves:
                print(f"  ⚠️ Move is still in the valid moves list after being marked invalid!")
            
            # Check if the dice are available
            if from_pos != to_pos and to_pos != 'off':
                move_distance = abs(from_pos - to_pos)
                if move_distance in dice:
                    print(f"  ✓ Move distance {move_distance} matches a die value in {dice}")
                else:
                    print(f"  ❌ Move distance {move_distance} does not match any die value in {dice}")
            elif to_pos == 'off':
                # For bearing off
                if from_pos + 1 in dice:
                    print(f"  ✓ Bearing off position {from_pos} matches a die value {from_pos+1} in {dice}")
                else:
                    print(f"  ❌ Bearing off position {from_pos} does not match any die value in {dice}")
        else:
            print(f"  ✓ Move {move} is valid and executed successfully!")
            print(f"  New board state: {test_env.unwrapped.game.board}")
            print(f"  Reward: {reward}")
    
    # Print overall result
    if all_moves_valid:
        print("\n✅ All moves reported as valid were executed successfully!")
    else:
        print("\n❌ Some moves reported as valid were marked as invalid when executed!")

def test_game_directly():
    """
    Test the Narde game logic directly without the gymnasium wrapper.
    This can help identify if issues are in the game logic or the environment wrapper.
    """
    print("\n=== Testing Narde Game Logic Directly ===")
    
    # Create a game instance
    game = Narde()
    board = game.board
    print(f"Initial board state: {board}")
    
    # Set dice directly since there's no roll_dice method
    game.dice = [6, 2]
    dice = game.dice
    print(f"Dice values: {dice}")
    
    valid_moves = game.get_valid_moves(dice)
    print(f"Valid moves: {valid_moves}")
    
    if not valid_moves:
        print("No valid moves returned.")
        return
    
    # Test each move
    for i, move in enumerate(valid_moves):
        print(f"\nTesting direct move {i+1}: {move}")
        
        # Create a copy of the game to test this move
        import copy
        test_game = copy.deepcopy(game)
        
        # Execute the move
        try:
            original_dice = test_game.dice.copy()  # Save original dice to show what's used
            success = test_game.execute_move(move)
            print(f"  ✓ Move executed successfully: {success}")
            print(f"  Original dice: {original_dice}, Remaining dice: {test_game.dice}")
            print(f"  New board state: {test_game.board}")
        except Exception as e:
            print(f"  ❌ Error executing move: {e}")
            
            # Check move details
            from_pos, to_pos = move
            print(f"  Move details:")
            print(f"  - from_pos: {from_pos} (type: {type(from_pos)})")
            print(f"  - to_pos: {to_pos} (type: {type(to_pos)})")
            
            if isinstance(from_pos, np.integer) or isinstance(to_pos, np.integer):
                # Try with regular integers
                int_from_pos = int(from_pos)
                int_to_pos = int(to_pos) if to_pos != 'off' else 'off'
                int_move = (int_from_pos, int_to_pos)
                
                print(f"  Trying with regular integers: {int_move}")
                try:
                    success = test_game.execute_move(int_move)
                    print(f"  ✓ Move executed successfully with regular integers: {success}")
                except Exception as e:
                    print(f"  ❌ Error executing move with regular integers: {e}")

def test_move_conversion():
    """
    Test if move format conversion between numpy integers and Python integers
    affects move validation.
    """
    print("\n=== Testing Move Format Conversion ===")
    
    # Create the environment
    env = NardeEnv()
    obs, info = env.reset()
    
    # Get dice and valid moves
    dice = env.unwrapped.game.dice
    print(f"Dice roll: {dice}")
    
    valid_moves = env.unwrapped.game.get_valid_moves(dice)
    print(f"Valid moves returned by environment: {valid_moves}")
    
    if not valid_moves:
        print("No valid moves returned by the environment.")
        return
    
    # Convert moves between different integer types
    conversions = [
        ("Numpy int64 to Python int", lambda move: (int(move[0]), int(move[1]) if move[1] != 'off' else 'off')),
        ("Python int to Numpy int64", lambda move: (np.int64(move[0]), np.int64(move[1]) if move[1] != 'off' else 'off')),
        ("To list instead of tuple", lambda move: [move[0], move[1]]),
        ("Numpy array elements", lambda move: (np.array(move[0]), np.array(move[1]) if move[1] != 'off' else 'off')),
    ]
    
    for desc, converter in conversions:
        print(f"\n--- Testing {desc} ---")
        
        for i, move in enumerate(valid_moves):
            try:
                # Convert the move
                converted_move = converter(move)
                print(f"Original move: {move} (types: {type(move[0])}, {type(move[1])})")
                print(f"Converted move: {converted_move} (types: {type(converted_move[0])}, {type(converted_move[1])})")
                
                # Check if the converted move is still in valid moves
                if converted_move in valid_moves:
                    print(f"✓ Converted move is still recognized as valid")
                else:
                    print(f"❌ Converted move is no longer recognized as valid")
                
                # Try executing the converted move
                import copy
                test_env = copy.deepcopy(env)
                next_obs, reward, terminated, truncated, info = test_env.step(converted_move)
                
                if 'invalid_move' in info and info['invalid_move']:
                    print(f"❌ Converted move was marked invalid when executed: {info}")
                else:
                    print(f"✓ Converted move executed successfully!")
            
            except Exception as e:
                print(f"❌ Error during conversion or execution: {e}")
                
            # Only test the first move for each conversion type to keep output manageable
            break

def test_action_decoding():
    """
    Test the action decoding logic in NardeEnv to see if it properly 
    converts between gym action space and game moves.
    """
    print("\n=== Testing Action Decoding ===")
    
    # Create the environment
    env = NardeEnv()
    obs, info = env.reset()
    
    # Get valid moves from game
    dice = env.unwrapped.game.dice  
    valid_moves = env.unwrapped.game.get_valid_moves(dice)
    print(f"Dice roll: {dice}")
    print(f"Valid moves from game: {valid_moves}")
    
    if not valid_moves:
        print("No valid moves returned by the environment.")
        return
    
    # Test if the environment can decode these moves properly
    print("\n--- Test bidirectional move conversion ---")
    for move in valid_moves:
        from_pos, to_pos = move
        
        # Convert game move to gym action format
        # For normal moves: move_index = from_pos*24 + to_pos, move_type=0
        # For bearing off: move_index = from_pos*24 + 0, move_type=1 
        if to_pos == 'off':
            move_index = from_pos * 24
            move_type = 1
        else:
            move_index = from_pos * 24 + to_pos
            move_type = 0
            
        gym_action = (move_index, move_type)
        
        print(f"\nOriginal move: {move}")
        print(f"Converted to gym action: {gym_action}")
        
        # Decode the action back to a game move
        decoded_move = env._decode_action(gym_action)
        print(f"Decoded back to move: {decoded_move}")
        
        # Check if we got the same move back
        if decoded_move == move:
            print(f"✓ Move correctly decoded!")
        else:
            print(f"❌ Move incorrectly decoded: Expected {move}, got {decoded_move}")
            
        # Now try executing the action in the environment
        import copy
        test_env = copy.deepcopy(env)
        next_obs, reward, terminated, truncated, info = test_env.step(gym_action)
        
        if 'invalid_move' in info and info['invalid_move']:
            print(f"❌ Gym action was marked invalid when executed: {info}")
        else:
            print(f"✓ Gym action executed successfully!")
            print(f"  New board state: {test_env.unwrapped.game.board}")
    
    # Test if the environment accepts direct move format if no decoding needed
    print("\n--- Test direct move execution ---")
    for move in valid_moves[:1]:  # Just test the first move
        import copy
        test_env = copy.deepcopy(env)
        
        # First verify the move is valid
        valid_moves = test_env.unwrapped.game.get_valid_moves(test_env.dice)
        if move in valid_moves:
            print(f"Move {move} is in valid_moves from the game")
        else:
            print(f"Move {move} is NOT in valid_moves from the game")
        
        # Try executing the move directly (not decoded as a gym action)
        print(f"Executing move directly: {move}")
        next_obs, reward, terminated, truncated, info = test_env.step(move)
        
        if 'invalid_move' in info and info['invalid_move']:
            print(f"❌ Direct move was marked invalid when executed: {info}")
        else:
            print(f"✓ Direct move executed successfully!")
            print(f"  New board state: {test_env.unwrapped.game.board}")

def test_unwrapped_env():
    """
    Test the unwrapped environment directly to bypass any wrappers that might interfere.
    This helps isolate if the issue is with the environment itself or with some wrapper.
    """
    print("\n=== Testing Unwrapped Environment Directly ===")
    
    # Create environment
    env = gym.make('Narde-v0').unwrapped
    env.reset()
    
    # Debug info
    print(f"Dice: {env.dice}")
    print(f"Board: {env.game.board}")
    
    # Get valid moves
    valid_moves = env.game.get_valid_moves(env.dice)
    print(f"Valid moves: {valid_moves}")
    
    if not valid_moves:
        print("No valid moves available.")
        return
    
    # Test each move directly in the unwrapped environment
    for move in valid_moves[:1]:  # Just test the first move for brevity
        from_pos, to_pos = move
        print(f"\nTesting move {move}...")
        
        # Create gym action format for this move
        if to_pos == 'off':
            move_index = from_pos * 24
            move_type = 1  # Bearing off
        else:
            move_index = from_pos * 24 + to_pos
            move_type = 0  # Regular move
        
        gym_action = (move_index, move_type)
        print(f"Converted to gym action: {gym_action}")
        
        # Check if decode_action works correctly
        decoded_move = env._decode_action(gym_action)
        print(f"Decoded back to move: {decoded_move}")
        if decoded_move == move:
            print(f"✓ Action correctly decoded")
        else:
            print(f"❌ Action incorrectly decoded: Expected {move}, got {decoded_move}")
        
        # Now execute the move through step
        print(f"Executing step with action {gym_action}...")
        next_state, reward, terminated, truncated, info = env.step(gym_action)
        
        if 'invalid_move' in info and info['invalid_move']:
            print(f"❌ Move was marked invalid! Info: {info}")
        else:
            print(f"✓ Move executed successfully! Reward: {reward}")
            print(f"New board state: {env.game.board}")
        
        # Also try executing the move directly against the game instance
        print(f"Executing move directly with game.execute_move...")
        import copy
        game_copy = copy.deepcopy(env.game)
        original_dice = game_copy.dice.copy()
        success = game_copy.execute_move(move)
        print(f"Direct execution result: {success}")
        print(f"Original dice: {original_dice}, Remaining dice: {game_copy.dice}")
        print(f"Board after direct execution: {game_copy.board}")

if __name__ == "__main__":
    print("=== Starting Narde Environment Tests ===")
    test_valid_moves_execution()
    test_game_directly()
    test_move_conversion()
    test_action_decoding()
    test_unwrapped_env()
    print("\n=== Tests Completed ===") 