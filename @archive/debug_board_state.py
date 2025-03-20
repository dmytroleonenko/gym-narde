#!/usr/bin/env python3
"""
Debug Board State and Game Mechanics

This script inspects the internal state of the Narde game to understand how
the board evolves, why valid moves might be unavailable, and how to properly
complete a game through bearing off.
"""

import os
import time
import random
import numpy as np
import logging

from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Debug-Board-State')

def print_board(board, dice=None):
    """Print the Narde board in a more readable format."""
    logger.info("Board state:")
    
    # Check if board is not in numpy array format
    if not isinstance(board, np.ndarray):
        logger.warning(f"Board is not a numpy array: {type(board)}")
        logger.info(f"Board value: {board}")
        return
    
    # Print board shape for debugging
    logger.info(f"Board shape: {board.shape}")
    
    # Adapt to the board format
    if board.ndim == 1:  # Linear board
        logger.info("Linear board format:")
        for i in range(0, len(board), 8):  # Assuming 8 positions per row
            row_str = ""
            for j in range(min(8, len(board) - i)):
                cell = board[i + j]
                if cell > 0:
                    row_str += f"+{cell:2d} "
                elif cell < 0:
                    row_str += f"{cell:3d} "
                else:
                    row_str += "  0 "
            logger.info(f"Row {i//8}: {row_str}")
    elif board.ndim == 2:  # 2D board
        for i in range(board.shape[0]):  # Rows
            row_str = ""
            for j in range(board.shape[1]):  # Columns
                cell = board[i, j]
                if cell > 0:
                    row_str += f"+{cell:2d} "
                elif cell < 0:
                    row_str += f"{cell:3d} "
                else:
                    row_str += "  0 "
            logger.info(f"Row {i}: {row_str}")
    else:
        logger.warning(f"Unexpected board dimension: {board.ndim}")
        logger.info(f"Raw board data: {board}")
        
    if dice:
        logger.info(f"Dice: {dice}")

def print_player_positions(board):
    """Print positions of pieces for each player."""
    if not isinstance(board, np.ndarray):
        logger.warning("Board is not a numpy array, can't print positions")
        return
        
    white_positions = []
    black_positions = []
    
    # Flatten the board to linear positions if it's not already flat
    flat_board = board.flatten() if board.ndim > 1 else board
    
    for pos, count in enumerate(flat_board):
        if count > 0:  # White pieces
            white_positions.append((pos, count))
        elif count < 0:  # Black pieces
            black_positions.append((pos, abs(count)))
    
    logger.info(f"White pieces: {white_positions}")
    logger.info(f"Black pieces: {black_positions}")
    logger.info(f"Total white: {sum(count for _, count in white_positions)}")
    logger.info(f"Total black: {sum(count for _, count in black_positions)}")

def explore_valid_moves(env, debug_level=0):
    """Explore what valid moves are possible with the current board and dice."""
    try:
        valid_moves = env.game.get_valid_moves()
        
        logger.info(f"Found {len(valid_moves)} valid moves")
        if len(valid_moves) == 0:
            logger.info("No valid moves available")
            
            # Check if dice is the issue by trying all possible dice combinations
            if debug_level >= 1:
                logger.info("Testing all possible dice combinations...")
                for d1 in range(1, 7):
                    for d2 in range(1, 7):
                        test_dice = [d1, d2]
                        env.dice = test_dice
                        test_moves = env.game.get_valid_moves()
                        if test_moves:
                            logger.info(f"Found {len(test_moves)} valid moves with dice {test_dice}")
                            logger.info(f"Example moves: {test_moves[:5]}")
                            return
                logger.warning("No valid moves available with any dice combination!")
        else:
            logger.info(f"Valid moves: {valid_moves[:10]}")
            
            # Convert a few moves to action tuples to verify conversion
            if debug_level >= 1 and valid_moves:
                logger.info("Converting moves to action tuples:")
                for i, move in enumerate(valid_moves[:3]):
                    from_pos, to_pos = move
                    move_type = 1 if to_pos == 'off' else 0
                    
                    if move_type == 0:
                        # Regular move
                        move_index = from_pos * 24 + to_pos
                    else:
                        # Bearing off - use from_pos directly
                        move_index = from_pos
                        
                    action_tuple = (move_index, move_type)
                    logger.info(f"Move {i}: {move} -> Action: {action_tuple}")
    except Exception as e:
        logger.error(f"Error exploring valid moves: {e}")
        import traceback
        logger.error(traceback.format_exc())

def examine_bearing_off_rules(env):
    """Examine the rules for bearing off in the Narde environment."""
    logger.info("Examining bearing off rules...")
    
    # Check if the game has bearing off counters
    if hasattr(env.game, 'borne_off_white'):
        logger.info(f"Current borne off - White: {env.game.borne_off_white}, Black: {env.game.borne_off_black}")
    else:
        logger.warning("Game does not have borne_off_white/black attributes")
    
    # Check the implementation of is_game_over
    if hasattr(env.game, 'is_game_over'):
        logger.info("Inspecting is_game_over implementation...")
        try:
            game_over = env.game.is_game_over()
            logger.info(f"is_game_over() returned: {game_over}")
            
            # Look at the code for is_game_over method
            import inspect
            if inspect.ismethod(env.game.is_game_over):
                try:
                    logger.info(f"is_game_over method source:")
                    logger.info(inspect.getsource(env.game.is_game_over.__func__))
                except Exception as e:
                    logger.error(f"Couldn't get source: {e}")
        except Exception as e:
            logger.error(f"Error calling is_game_over: {e}")
    
    # Check if the step method properly handles game completion
    logger.info("Checking NardeEnv.step handling of game completion...")
    if hasattr(env, 'step'):
        import inspect
        if inspect.ismethod(env.step):
            try:
                step_source = inspect.getsource(env.step)
                done_check_lines = [line for line in step_source.split('\n') if 'done' in line]
                logger.info(f"Step method done checks: {done_check_lines}")
            except Exception as e:
                logger.error(f"Couldn't get step source: {e}")

def debug_game_step_by_step(env_seed=42, max_steps=20):
    """Play a game step by step with detailed debugging at each step."""
    env = NardeEnv()
    reset_result = env.reset(seed=env_seed)
    
    if isinstance(reset_result, tuple):
        observation = reset_result[0]
    else:
        observation = reset_result
    
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    logger.info(f"Starting debug game with seed {env_seed}")
    logger.info("=== INITIAL STATE ===")
    
    # Debug environment properties
    logger.info(f"Environment properties:")
    for attr in dir(env):
        if not attr.startswith('_'):
            try:
                value = getattr(env, attr)
                if not callable(value):
                    logger.info(f"env.{attr} = {value}")
            except Exception as e:
                logger.debug(f"Error getting attribute {attr}: {e}")
    
    # Debug game properties
    logger.info(f"Game properties:")
    for attr in dir(env.game):
        if not attr.startswith('_'):
            try:
                value = getattr(env.game, attr)
                if not callable(value):
                    logger.info(f"env.game.{attr} = {value}")
            except Exception as e:
                logger.debug(f"Error getting attribute {attr}: {e}")
                
    print_board(env.game.board, env.dice)
    print_player_positions(env.game.board)
    
    # Try to determine whose turn it is
    if hasattr(env.game, 'current_player'):
        logger.info(f"Current player: {env.game.current_player}")
    else:
        # Infer from the observation or board state
        player_indicators = observation[-3:] if len(observation.shape) == 1 else observation[-3:]  # Last elements often indicate player
        logger.info(f"Player indicators in observation: {player_indicators}")
    
    step_count = 0
    done = False
    
    while not done and step_count < max_steps:
        logger.info(f"\n=== STEP {step_count + 1} ===")
        
        # Show board and dice
        print_board(env.game.board, env.dice)
        
        # Examine valid moves
        explore_valid_moves(env, debug_level=1)
        
        # Check bearing off status
        if hasattr(env.game, 'borne_off_white'):
            logger.info(f"Bearing off - White: {env.game.borne_off_white}, Black: {env.game.borne_off_black}")
        
        # Try to make a move
        valid_moves = env.game.get_valid_moves()
        
        if valid_moves:
            # Select a move
            move = random.choice(valid_moves)
            logger.info(f"Selected move: {move}")
            
            # Convert to action tuple
            from_pos, to_pos = move
            move_type = 1 if to_pos == 'off' else 0
            
            if move_type == 0:
                # Regular move
                move_index = from_pos * 24 + to_pos
            else:
                # Bearing off
                move_index = from_pos
                
            action_tuple = (move_index, move_type)
            logger.info(f"Action tuple: {action_tuple}")
            
            # Execute step
            try:
                step_result = env.step(action_tuple)
                
                # Handle result
                if len(step_result) == 5:
                    next_observation, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_observation, reward, done, info = step_result
                    
                logger.info(f"Step result - Reward: {reward}, Done: {done}")
                if info:
                    logger.info(f"Info: {info}")
                    
                observation = next_observation
            except Exception as e:
                logger.error(f"Error executing step: {e}")
                import traceback
                logger.error(traceback.format_exc())
                break
        else:
            logger.warning("No valid moves available")
            
            # Roll new dice and continue
            env.dice = [random.randint(1, 6), random.randint(1, 6)]
            logger.info(f"Rolled new dice: {env.dice}")
            
            # Check if there are still no valid moves with new dice
            try:
                new_valid_moves = env.game.get_valid_moves()
                if not new_valid_moves:
                    logger.warning("Still no valid moves after rolling new dice")
                    
                    # Check if we need to handle player turn
                    if hasattr(env.game, 'rotate_board_for_next_player'):
                        logger.info("Rotating board for next player")
                        env.game.rotate_board_for_next_player()
                        logger.info("Board after rotation:")
                        print_board(env.game.board, env.dice)
            except Exception as e:
                logger.error(f"Error getting valid moves after reroll: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        step_count += 1
        
        # Check for game over
        if hasattr(env.game, 'is_game_over'):
            try:
                game_over = env.game.is_game_over()
                logger.info(f"is_game_over() after step: {game_over}")
            except Exception as e:
                logger.error(f"Error checking game over: {e}")
    
    # Final state
    logger.info("\n=== FINAL STATE ===")
    print_board(env.game.board, env.dice)
    print_player_positions(env.game.board)
    
    # Check bearing off status
    if hasattr(env.game, 'borne_off_white'):
        logger.info(f"Final bearing off - White: {env.game.borne_off_white}, Black: {env.game.borne_off_black}")
    
    logger.info(f"Game completed after {step_count} steps, done={done}")
    
    # Check is_game_over one last time
    if hasattr(env.game, 'is_game_over'):
        try:
            final_game_over = env.game.is_game_over()
            logger.info(f"Final is_game_over(): {final_game_over}")
        except Exception as e:
            logger.error(f"Error checking final game over: {e}")

if __name__ == "__main__":
    logger.info("Starting board state and game mechanics debugging")
    
    try:
        # First, debug a step-by-step game
        debug_game_step_by_step(env_seed=42, max_steps=20)
        
        # Then, examine the bearing off rules in detail
        env = NardeEnv()
        env.reset()
        examine_bearing_off_rules(env)
        
        logger.info("Debugging complete!")
    except Exception as e:
        logger.error(f"Error during debugging: {e}")
        import traceback
        logger.error(traceback.format_exc()) 