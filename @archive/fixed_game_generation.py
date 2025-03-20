#!/usr/bin/env python3
"""
Fixed Game Generation for Narde

This script provides a properly fixed implementation for generating games in the Narde
environment, ensuring that:
1. Games properly complete through bearing off of all checkers
2. The script handles situations when there are no valid moves
3. Games don't get stuck in endless loops
"""

import os
import time
import random
import logging
import argparse
import numpy as np
from functools import lru_cache

from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Fixed-GameGeneration')

class GameHistory:
    """Store game history including observations, actions, and rewards."""
    
    def __init__(self):
        """Initialize empty game history."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.moves = []  # Store actual moves (from_pos, to_pos)
        self.bearing_off_count = 0
        
    def append(self, observation, action, reward, move=None):
        """Add a step to the game history."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.moves.append(move)
        if move and move[1] == 'off':
            self.bearing_off_count += 1

@lru_cache(maxsize=1024)
def cached_get_valid_moves(board_tuple, dice_tuple):
    """
    Cached version of get_valid_moves to avoid recalculating for the same board and dice state.
    
    Args:
        board_tuple (tuple): Board state converted to tuple for caching
        dice_tuple (tuple): Dice values converted to tuple for caching
        
    Returns:
        list: Valid moves for the given board and dice state
    """
    # Convert tuples back to numpy arrays
    board = np.array(board_tuple)
    
    # Create a new Narde game with this board
    game = Narde()
    game.board = board
    
    # Convert dice tuple to list
    dice = list(dice_tuple)
    
    # Pass the dice explicitly to get_valid_moves
    return game.get_valid_moves(dice=dice)

def play_game(env_seed=42, max_steps=1000, use_cache=True, max_no_move_steps=50):
    """
    Play a complete game of Narde until proper completion (all checkers borne off).
    
    Args:
        env_seed (int): Random seed for environment initialization
        max_steps (int): Maximum number of steps before terminating the game
        use_cache (bool): Whether to use cached valid moves computation
        max_no_move_steps (int): Maximum consecutive steps without valid moves before ending
        
    Returns:
        tuple: (elapsed_time, game_history)
    """
    start_time = time.time()
    env = NardeEnv()
    
    # Set seed and reset
    reset_result = env.reset(seed=env_seed)
    if isinstance(reset_result, tuple):
        observation = reset_result[0]
    else:
        observation = reset_result
        
    # Set random seeds for consistency
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    # Create game history
    game_history = GameHistory()
    done = False
    step_count = 0
    no_move_counter = 0
    
    while not done and step_count < max_steps:
        # Get valid moves using cache if enabled
        if use_cache:
            board_tuple = tuple(env.game.board.flatten())
            dice_tuple = tuple(env.dice)
            valid_moves = cached_get_valid_moves(board_tuple, dice_tuple)
        else:
            valid_moves = env.game.get_valid_moves()
            
        # Check for proper game completion
        if hasattr(env.game, 'borne_off_white') and env.game.borne_off_white >= 15:
            logger.info(f"Game completed after {step_count} steps - White won by bearing off all checkers")
            done = True
            break
            
        if hasattr(env.game, 'borne_off_black') and env.game.borne_off_black >= 15:
            logger.info(f"Game completed after {step_count} steps - Black won by bearing off all checkers")
            done = True
            break
            
        # Select random action if valid moves exist
        if valid_moves:
            no_move_counter = 0  # Reset the counter since we have valid moves
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
            game_history.append(observation, action_tuple, reward, move)
            
            # Update for next iteration
            observation = next_observation
        else:
            # No valid moves available
            no_move_counter += 1
            
            if no_move_counter >= max_no_move_steps:
                logger.warning(f"Game ended due to {max_no_move_steps} consecutive steps with no valid moves")
                done = True
                break
                
            # Try with new dice
            env.dice = [random.randint(1, 6), random.randint(1, 6)]
            
            # Log dice reroll
            if step_count % 50 == 0 or no_move_counter % 10 == 0:
                logger.debug(f"Step {step_count}: No valid moves, rerolling dice to {env.dice}")
            
        step_count += 1
    
    # Calculate game stats
    elapsed_time = time.time() - start_time
    
    # Log completion status
    if step_count >= max_steps:
        logger.warning(f"Game terminated after reaching max steps ({max_steps})")
    
    # Log bearing off stats
    if hasattr(env.game, 'borne_off_white') and hasattr(env.game, 'borne_off_black'):
        logger.info(f"Final state - White borne off: {env.game.borne_off_white}, Black borne off: {env.game.borne_off_black}")
    
    return elapsed_time, game_history

def analyze_games(num_games=5, max_steps=1000, use_cache=True):
    """
    Play and analyze multiple games to get statistics on game length and completion.
    
    Args:
        num_games (int): Number of games to play
        max_steps (int): Maximum steps per game
        use_cache (bool): Whether to use caching for valid moves
        
    Returns:
        dict: Statistics about the games
    """
    logger.info(f"Analyzing {num_games} games (max_steps={max_steps}, use_cache={use_cache})...")
    
    start_time = time.time()
    games_data = []
    
    for i in range(num_games):
        logger.info(f"Starting game {i+1}/{num_games}")
        game_time, game_history = play_game(
            env_seed=i+42, 
            max_steps=max_steps, 
            use_cache=use_cache
        )
        
        # Store game data
        games_data.append({
            'game_id': i,
            'num_moves': len(game_history.moves),
            'bearing_off_moves': game_history.bearing_off_count,
            'time': game_time
        })
        
        logger.info(f"Game {i+1} completed in {game_time:.2f}s with {len(game_history.moves)} moves "
                   f"({game_history.bearing_off_count} bearing off)")
    
    # Calculate overall statistics
    total_time = time.time() - start_time
    total_moves = sum(g['num_moves'] for g in games_data)
    total_bearing_off = sum(g['bearing_off_moves'] for g in games_data)
    
    avg_moves = total_moves / num_games if num_games > 0 else 0
    avg_bearing_off = total_bearing_off / num_games if num_games > 0 else 0
    games_per_second = num_games / total_time if total_time > 0 else 0
    
    # Print summary
    logger.info("\n=== ANALYSIS RESULTS ===")
    logger.info(f"Total games: {num_games}")
    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Average moves per game: {avg_moves:.1f}")
    logger.info(f"Average bearing off moves per game: {avg_bearing_off:.1f}")
    logger.info(f"Games per second: {games_per_second:.2f}")
    
    # Calculate cache impact if multiple games were played
    if num_games > 1 and use_cache:
        cache_info = cached_get_valid_moves.cache_info()
        hit_ratio = cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
        logger.info(f"Cache statistics - Hits: {cache_info.hits}, Misses: {cache_info.misses}, Hit ratio: {hit_ratio:.2f}")
    
    return {
        'num_games': num_games,
        'total_time': total_time,
        'avg_moves': avg_moves,
        'avg_bearing_off': avg_bearing_off,
        'games_per_second': games_per_second,
        'games_data': games_data
    }

def test_cache_impact(num_games=5, max_steps=1000):
    """
    Compare performance with and without caching.
    
    Args:
        num_games (int): Number of games to play for each mode
        max_steps (int): Maximum steps per game
        
    Returns:
        tuple: (no_cache_stats, with_cache_stats, speedup)
    """
    logger.info("Testing cache impact...")
    
    # Clear cache
    cached_get_valid_moves.cache_clear()
    
    # Test without cache
    logger.info("\nRunning without cache:")
    no_cache_stats = analyze_games(num_games, max_steps, use_cache=False)
    
    # Clear cache again
    cached_get_valid_moves.cache_clear()
    
    # Test with cache
    logger.info("\nRunning with cache:")
    with_cache_stats = analyze_games(num_games, max_steps, use_cache=True)
    
    # Calculate speedup
    speedup = with_cache_stats['games_per_second'] / no_cache_stats['games_per_second'] if no_cache_stats['games_per_second'] > 0 else 0
    logger.info(f"\nCache speedup: {speedup:.2f}x")
    
    return no_cache_stats, with_cache_stats, speedup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Game Generation for Narde")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per game")
    parser.add_argument("--test-cache", action="store_true", help="Test impact of caching")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test_cache:
        test_cache_impact(args.games, args.max_steps)
    else:
        analyze_games(args.games, args.max_steps) 