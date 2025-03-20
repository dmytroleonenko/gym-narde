#!/usr/bin/env python3
import os
import time
import random
import numpy as np
import logging

# Import necessary modules
from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Debug-Complete-Game')

def print_board_state(env):
    """Print detailed information about the current board state."""
    logger.info("=== BOARD STATE ===")
    logger.info(f"Board: {env.game.board}")
    logger.info(f"Dice: {env.dice}")
    
    # Get valid moves directly from the game
    valid_moves = env.game.get_valid_moves()
    logger.info(f"Valid moves ({len(valid_moves)}): {valid_moves[:5]}...")
    
    # Check if there are any borne off checkers
    logger.info(f"Borne off - White: {env.game.borne_off_white}, Black: {env.game.borne_off_black}")
    
    # Print observation
    obs = env._get_obs()
    logger.info(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    logger.info(f"Dice count in obs: {obs[24]}")

def test_action_conversion(from_pos, to_pos):
    """Test the conversion from Narde move to action tuple and back."""
    # Forward conversion (move to action)
    move_type = 1 if to_pos == 'off' else 0
    
    if move_type == 0:
        # Regular move
        move_index = from_pos * 24 + to_pos
    else:
        # Bearing off - use from_pos directly
        move_index = from_pos
    
    action_tuple = (move_index, move_type)
    
    # Create an environment to test the reverse conversion
    env = NardeEnv()
    decoded_move = env._decode_action(action_tuple)
    
    logger.info(f"Original move: ({from_pos}, {to_pos})")
    logger.info(f"Converted to action: {action_tuple}")
    logger.info(f"Decoded back to move: {decoded_move}")
    
    # Check if the conversion is correct
    is_correct = (decoded_move[0] == from_pos) and (decoded_move[1] == to_pos)
    logger.info(f"Conversion is correct: {is_correct}")
    
    return is_correct

def play_complete_game(env_seed=42, max_steps=100):
    """Play a complete game with detailed logging."""
    env = NardeEnv(debug=True)  # Enable debug mode in NardeEnv
    
    # Set seed and reset
    reset_result = env.reset(seed=env_seed)
    if isinstance(reset_result, tuple):
        observation = reset_result[0]
    else:
        observation = reset_result
        
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    logger.info("=== INITIAL STATE ===")
    print_board_state(env)
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        logger.info(f"\n=== STEP {step_count+1} ===")
        
        # Get valid moves directly
        valid_moves = env.game.get_valid_moves()
        
        if len(valid_moves) > 0:
            logger.info(f"Found {len(valid_moves)} valid moves")
            
            # Pick a move and convert it to action
            move = random.choice(valid_moves)
            logger.info(f"Selected move: {move}")
            
            # Convert to NardeEnv action format
            from_pos, to_pos = move
            move_type = 1 if to_pos == 'off' else 0
            
            if move_type == 0:
                # Regular move
                move_index = from_pos * 24 + to_pos
            else:
                # Bearing off - use from_pos directly
                move_index = from_pos
                
            # Create the action tuple
            action_tuple = (move_index, move_type)
            logger.info(f"Converted to action: {action_tuple}")
            
            # Execute the action
            step_result = env.step(action_tuple)
            
            # Handle step result format
            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_observation, reward, done, info = step_result
                
            logger.info(f"Step result - Reward: {reward}, Done: {done}")
            logger.info(f"Info: {info}")
            
            # Update state and print new board
            observation = next_observation
            print_board_state(env)
        else:
            logger.info("No valid moves available")
            done = True
            
        step_count += 1
    
    logger.info(f"Game completed after {step_count} steps")
    return step_count

if __name__ == "__main__":
    logger.info("Starting debug of a complete game...")
    
    # Test action conversion
    logger.info("\n=== TESTING ACTION CONVERSION ===")
    
    # Test regular move conversion
    logger.info("\nRegular move conversion test:")
    test_action_conversion(23, 17)  # Regular move from pos 23 to 17
    
    # Test bearing off move conversion
    logger.info("\nBearing off move conversion test:")
    test_action_conversion(5, 'off')  # Bearing off from pos 5
    
    # Play a complete game with detailed logging
    logger.info("\n=== PLAYING COMPLETE GAME ===")
    steps_completed = play_complete_game(env_seed=42, max_steps=20)
    
    logger.info(f"Completed {steps_completed} steps in debug game")
    logger.info("\nDebug complete!") 