#!/usr/bin/env python3
"""
Narde Game Completion Verification

This script verifies that Narde games are played to completion with proper bearing off
of all checkers, and reports detailed statistics on game length.
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Narde-Verification')

def play_full_game(env_seed=42, max_steps=1000):
    """
    Play a complete game of Narde until proper completion (all checkers borne off).
    Includes detailed tracking of bearing off progress.
    
    Args:
        env_seed: Seed for the environment
        max_steps: Safety limit for maximum steps (should be high enough for a full game)
        
    Returns:
        dict: Statistics about the game
    """
    env = NardeEnv()
    
    # Set seed and reset
    reset_result = env.reset(seed=env_seed)
    if isinstance(reset_result, tuple):
        observation = reset_result[0]
    else:
        observation = reset_result
        
    random.seed(env_seed)
    np.random.seed(env_seed)
    
    # Track game progress
    step_count = 0
    bearing_off_count = 0
    regular_move_count = 0
    no_valid_moves_count = 0
    bearing_off_progress = []  # Track bearing off progress
    
    # Start game
    logger.info(f"Starting game with seed {env_seed}")
    
    done = False
    true_completion = False
    last_valid_moves = []
    
    while not done and step_count < max_steps:
        # Get valid moves
        valid_moves = env.game.get_valid_moves()
        last_valid_moves = valid_moves
        
        # Log bearing off status every 50 steps
        if step_count % 50 == 0:
            logger.info(f"Step {step_count}: White borne off: {env.game.borne_off_white}, "
                        f"Black borne off: {env.game.borne_off_black}")
        
        if len(valid_moves) > 0:
            move = random.choice(valid_moves)
            
            # Convert to NardeEnv action format
            from_pos, to_pos = move
            move_type = 1 if to_pos == 'off' else 0
            
            if move_type == 0:
                # Regular move
                move_index = from_pos * 24 + to_pos
                regular_move_count += 1
            else:
                # Bearing off
                move_index = from_pos
                bearing_off_count += 1
                
                # Record bearing off progress
                bearing_off_progress.append({
                    'step': step_count,
                    'white': env.game.borne_off_white,
                    'black': env.game.borne_off_black,
                    'move': move
                })
                
            # Create action tuple and execute
            action_tuple = (move_index, move_type)
            step_result = env.step(action_tuple)
            
            # Handle step result format
            if len(step_result) == 5:
                next_observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_observation, reward, done, info = step_result
            
            # Update observation
            observation = next_observation
            
            # Check if game should be considered truly complete
            if env.game.borne_off_white >= 15 or env.game.borne_off_black >= 15:
                true_completion = True
                logger.info(f"Game properly completed at step {step_count} through bearing off!")
                logger.info(f"Final state - White: {env.game.borne_off_white}, Black: {env.game.borne_off_black}")
                break
        else:
            no_valid_moves_count += 1
            logger.info(f"Step {step_count}: No valid moves available")
            
            # Check if we should end the game or roll again
            if hasattr(env, 'dice'):
                # Force new dice and try again
                env.dice = [random.randint(1, 6), random.randint(1, 6)]
                logger.info(f"Rolling new dice: {env.dice}")
            else:
                # No dice attribute, have to end
                done = True
        
        step_count += 1
    
    # Final state
    logger.info(f"Game ended after {step_count} steps")
    logger.info(f"White borne off: {env.game.borne_off_white}, Black borne off: {env.game.borne_off_black}")
    logger.info(f"Regular moves: {regular_move_count}, Bearing off moves: {bearing_off_count}")
    logger.info(f"True completion through bearing off: {true_completion}")
    
    if done and not true_completion:
        logger.warning("Game ended without proper completion!")
        if step_count >= max_steps:
            logger.warning(f"Reached maximum steps limit ({max_steps})")
        elif no_valid_moves_count > 0:
            logger.warning(f"Ended due to no valid moves ({no_valid_moves_count} occurrences)")
        
        # Log final valid moves
        logger.info(f"Last valid moves: {last_valid_moves}")
        
        # Log board state
        logger.info(f"Final board state: {env.game.board}")
    
    # Return stats
    return {
        'steps': step_count,
        'regular_moves': regular_move_count,
        'bearing_off_moves': bearing_off_count,
        'no_valid_moves_count': no_valid_moves_count,
        'white_borne_off': env.game.borne_off_white,
        'black_borne_off': env.game.borne_off_black,
        'true_completion': true_completion,
        'bearing_off_progress': bearing_off_progress
    }

def analyze_game_lengths(num_games=10, max_steps=1000):
    """Analyze the length and outcome of multiple Narde games."""
    
    logger.info(f"Analyzing {num_games} complete Narde games...")
    
    stats = []
    start_time = time.time()
    
    for i in range(num_games):
        logger.info(f"\n=== GAME {i+1}/{num_games} ===")
        game_stats = play_full_game(env_seed=i+42, max_steps=max_steps)
        stats.append(game_stats)
    
    total_time = time.time() - start_time
    
    # Analyze results
    completed_games = [s for s in stats if s['true_completion']]
    incomplete_games = [s for s in stats if not s['true_completion']]
    
    # Calculate stats for completed games
    if completed_games:
        avg_steps = sum(s['steps'] for s in completed_games) / len(completed_games)
        avg_regular_moves = sum(s['regular_moves'] for s in completed_games) / len(completed_games)
        avg_bearing_off = sum(s['bearing_off_moves'] for s in completed_games) / len(completed_games)
    else:
        avg_steps = 0
        avg_regular_moves = 0
        avg_bearing_off = 0
    
    # Print summary
    logger.info("\n=== ANALYSIS RESULTS ===")
    logger.info(f"Total games: {num_games}")
    logger.info(f"Completed games: {len(completed_games)} ({len(completed_games)/num_games*100:.1f}%)")
    logger.info(f"Incomplete games: {len(incomplete_games)} ({len(incomplete_games)/num_games*100:.1f}%)")
    
    if completed_games:
        logger.info("\nCompleted game statistics:")
        logger.info(f"Average steps: {avg_steps:.1f}")
        logger.info(f"Average regular moves: {avg_regular_moves:.1f}")
        logger.info(f"Average bearing off moves: {avg_bearing_off:.1f}")
        logger.info(f"Total average moves: {avg_regular_moves + avg_bearing_off:.1f}")
        
        # Step distribution
        steps_list = [s['steps'] for s in completed_games]
        min_steps = min(steps_list)
        max_steps = max(steps_list)
        logger.info(f"Step range: {min_steps} to {max_steps}")
    
    if incomplete_games:
        logger.info("\nIncomplete game analysis:")
        avg_incomplete_steps = sum(s['steps'] for s in incomplete_games) / len(incomplete_games)
        avg_incomplete_borne_white = sum(s['white_borne_off'] for s in incomplete_games) / len(incomplete_games)
        avg_incomplete_borne_black = sum(s['black_borne_off'] for s in incomplete_games) / len(incomplete_games)
        
        logger.info(f"Average steps before termination: {avg_incomplete_steps:.1f}")
        logger.info(f"Average white borne off: {avg_incomplete_borne_white:.1f}/15")
        logger.info(f"Average black borne off: {avg_incomplete_borne_black:.1f}/15")
    
    # Performance analysis
    games_per_second = num_games / total_time
    logger.info(f"\nPerformance: {games_per_second:.2f} games/s")
    
    return stats

if __name__ == "__main__":
    # Run analysis
    analyze_game_lengths(num_games=5, max_steps=1000) 