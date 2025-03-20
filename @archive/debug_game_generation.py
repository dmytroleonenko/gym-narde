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
logger = logging.getLogger('Debug-GameGeneration')

def print_board_state(env):
    """Print detailed information about the current board state."""
    logger.info("Board state:")
    logger.info(f"Board: {env.game.board}")
    logger.info(f"Dice: {env.dice}")
    
    # Get valid moves directly from the game
    valid_moves = env.game.get_valid_moves()
    logger.info(f"Valid moves: {valid_moves}")
    
    # Check if there are any borne off checkers
    if hasattr(env.game, 'borne_off_white'):
        logger.info(f"Borne off - White: {env.game.borne_off_white}, Black: {env.game.borne_off_black}")

def debug_action_conversion(valid_moves):
    """Debug how we convert from Narde moves to NardeEnv action tuples."""
    logger.info("Action conversion examples:")
    
    for move in valid_moves[:3]:  # Show first 3 moves
        from_pos, to_pos = move
        move_type = 1 if to_pos == 'off' else 0
        
        if move_type == 0:
            # Regular move
            move_index = from_pos * 24 + to_pos
        else:
            # Bearing off - use from_pos directly
            move_index = from_pos
            
        action_tuple = (move_index, move_type)
        
        logger.info(f"Narde move {move} -> NardeEnv action {action_tuple}")

def debug_play_game(env_seed=42, max_steps=10):
    """Debug a game play session."""
    env = NardeEnv(debug=True)  # Enable debug mode in NardeEnv
    
    # Set seed and reset
    observation = env.reset(seed=env_seed)
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
            debug_action_conversion(valid_moves)
            
            # Select a move
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
            next_observation, reward, terminated, truncated, info = env.step(action_tuple)
            logger.info(f"Step result - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            logger.info(f"Info: {info}")
            
            # Update state
            observation = next_observation
            done = terminated or truncated
            
            # Print new state
            logger.info("Board state after move:")
            print_board_state(env)
        else:
            logger.info("No valid moves available")
            done = True
            
        step_count += 1
    
    logger.info(f"Game completed after {step_count} steps")
    return step_count

def debug_game_history():
    """Debug storing game history."""
    env = NardeEnv()
    observation = env.reset(seed=42)
    
    # Check observation format
    logger.info(f"Observation shape: {observation.shape}")
    logger.info(f"Observation sample: {observation}")
    
    # Get a valid action
    valid_moves = env.game.get_valid_moves()
    if valid_moves:
        move = valid_moves[0]
        
        # Convert to action
        from_pos, to_pos = move
        move_type = 1 if to_pos == 'off' else 0
        
        if move_type == 0:
            move_index = from_pos * 24 + to_pos
        else:
            move_index = from_pos
            
        action_tuple = (move_index, move_type)
        
        # Execute and see result
        next_obs, reward, terminated, truncated, info = env.step(action_tuple)
        
        logger.info(f"Action: {action_tuple}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Next observation: {next_obs}")
    else:
        logger.info("No valid moves found")

if __name__ == "__main__":
    logger.info("Starting debug of game generation...")
    
    # Debug a full game play with detailed logging
    steps_completed = debug_play_game(env_seed=42, max_steps=10)
    
    logger.info(f"Completed {steps_completed} steps in debug game")
    
    # Debug how game history is stored
    logger.info("\nTesting game history storage...")
    debug_game_history()
    
    logger.info("\nDebug completed!") 