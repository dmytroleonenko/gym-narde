#!/usr/bin/env python3
import os
import time
import random
import numpy as np
import logging
from functools import lru_cache

# Import necessary modules
from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Debug-Cache-Issue')

# Simple game history class to store game data
class GameHistory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.to_play_history = []
        
    def append(self, observation, action, reward, to_play=1):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.to_play_history.append(to_play)

# Test the direct vs. cached valid moves function
@lru_cache(maxsize=1024)
def cached_get_valid_moves(board_tuple):
    """Cached version of get_valid_moves to avoid recalculating for the same board state."""
    # Convert tuple back to np.array for Narde
    board = np.array(board_tuple)
    game = Narde()
    game.board = board
    return game.get_valid_moves()

def test_valid_moves_equality(env):
    """Test if direct and cached valid moves return the same results."""
    # Get valid moves directly
    direct_moves = env.game.get_valid_moves()
    
    # Get valid moves via cache
    board_tuple = tuple(env.game.board)
    cached_moves = cached_get_valid_moves(board_tuple)
    
    # Compare length
    logger.info(f"Direct moves count: {len(direct_moves)}")
    logger.info(f"Cached moves count: {len(cached_moves)}")
    
    # Compare a few examples
    if len(direct_moves) > 0 and len(cached_moves) > 0:
        logger.info(f"Direct move example: {direct_moves[0]}")
        logger.info(f"Cached move example: {cached_moves[0]}")
    
    # Check if they are equal
    moves_match = set(str(m) for m in direct_moves) == set(str(m) for m in cached_moves)
    logger.info(f"Moves match: {moves_match}")
    
    return moves_match

def play_game_with_mode(env_seed=42, max_moves=50, use_cache=False):
    """Play a game with either direct or cached valid moves."""
    env = NardeEnv()
    
    # Set seed and reset
    reset_result = env.reset(seed=env_seed)
    if isinstance(reset_result, tuple):
        observation = reset_result[0]
    else:
        observation = reset_result
        
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    # Create game history
    game_history = GameHistory()
    done = False
    step_count = 0
    
    # Log info about initial state
    mode = "cached" if use_cache else "direct"
    logger.info(f"Starting game with {mode} mode, seed={env_seed}")
    
    while not done and step_count < max_moves:
        # Get valid moves - either from cache or directly
        if use_cache:
            board_tuple = tuple(env.game.board)
            valid_moves = cached_get_valid_moves(board_tuple)
        else:
            valid_moves = env.game.get_valid_moves()
        
        # Log state at first step and every 10 steps
        if step_count == 0 or step_count % 10 == 0:
            logger.info(f"Step {step_count}: Valid moves count: {len(valid_moves)}")
            if len(valid_moves) > 0:
                logger.info(f"First valid move: {valid_moves[0]}")
                logger.info(f"Board state: {env.game.board}")
                logger.info(f"Dice: {env.dice}")
        
        # Select random action if valid moves exist
        if len(valid_moves) > 0:
            move = random.choice(valid_moves)
            
            # Convert to NardeEnv action format
            from_pos, to_pos = move
            move_type = 1 if to_pos == 'off' else 0
            
            if move_type == 0:
                # Regular move
                move_index = from_pos * 24 + to_pos
            else:
                # Bearing off - use from_pos directly
                move_index = from_pos
                
            # Create action tuple
            action_tuple = (move_index, move_type)
            
            # Execute action
            step_result = env.step(action_tuple)
            
            # Handle step result format
            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_observation, reward, done, info = step_result
                
            # Store in game history
            game_history.append(observation, action_tuple, reward)
            
            # Update for next iteration
            observation = next_observation
        else:
            logger.info(f"Step {step_count}: No valid moves available!")
            done = True
            
        step_count += 1
    
    logger.info(f"Game completed after {step_count} steps")
    logger.info(f"Moves in game history: {len(game_history.actions)}")
    
    return game_history

def investigate_dice_issue():
    """Investigate if the dice are properly stored/transferred in the cached version."""
    env = NardeEnv()
    env.reset(seed=42)
    
    # Log initial dice and board
    logger.info(f"Initial dice: {env.dice}")
    logger.info(f"Initial board: {env.game.board}")
    
    # Create a Narde game with the same board
    game = Narde()
    game.board = env.game.board.copy()
    
    # Check if the Narde game has dice
    logger.info(f"Does game have dice attribute: {hasattr(game, 'dice')}")
    
    # Get valid moves directly
    direct_moves = env.game.get_valid_moves()
    logger.info(f"Direct moves from env.game: {len(direct_moves)}")
    
    # Get moves from our copied game
    copied_moves = game.get_valid_moves()
    logger.info(f"Moves from copied game: {len(copied_moves)}")
    
    # Quick comparison
    if len(direct_moves) > 0 and len(copied_moves) > 0:
        logger.info(f"Direct move example: {direct_moves[0]}")
        logger.info(f"Copied move example: {copied_moves[0]}")
    
    # Check if dice need to be passed to get_valid_moves
    if hasattr(game, 'get_valid_moves') and callable(getattr(game, 'get_valid_moves')):
        # Check if get_valid_moves accepts dice parameter
        import inspect
        sig = inspect.signature(game.get_valid_moves)
        logger.info(f"get_valid_moves signature: {sig}")
        
        # Try passing dice explicitly
        if 'dice' in sig.parameters:
            with_dice_moves = game.get_valid_moves(env.dice)
            logger.info(f"Moves with explicit dice: {len(with_dice_moves)}")

if __name__ == "__main__":
    logger.info("Investigating cache issue in game generation...")
    
    # Test if cached and direct valid moves return the same results
    logger.info("\n=== TESTING VALID MOVES EQUALITY ===")
    env = NardeEnv()
    env.reset(seed=42)
    test_valid_moves_equality(env)
    
    # Investigate dice issue
    logger.info("\n=== INVESTIGATING DICE ISSUE ===")
    investigate_dice_issue()
    
    # Play a game with direct valid moves
    logger.info("\n=== PLAYING GAME WITH DIRECT VALID MOVES ===")
    direct_history = play_game_with_mode(env_seed=42, max_moves=50, use_cache=False)
    
    # Clear cache
    cached_get_valid_moves.cache_clear()
    
    # Play a game with cached valid moves
    logger.info("\n=== PLAYING GAME WITH CACHED VALID MOVES ===")
    cached_history = play_game_with_mode(env_seed=42, max_moves=50, use_cache=True)
    
    # Compare results
    logger.info("\n=== COMPARISON RESULTS ===")
    logger.info(f"Direct moves: {len(direct_history.actions)}")
    logger.info(f"Cached moves: {len(cached_history.actions)}")
    
    # Output final result of investigation
    logger.info("\nInvestigation complete!") 