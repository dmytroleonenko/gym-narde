#!/usr/bin/env python3
"""
MuZero-based Game Generation for Narde

This script generates complete Narde games using MuZero for move selection:
1. The environment handles dice rolling, move execution, and board presentation
2. MuZero network is used to select valid moves from those provided by the environment
3. Games run in parallel with performance timing
4. Optimizations are applied to reduce move selection time
5. Batched MCTS inference for GPU acceleration
"""

import traceback
import time
import os
import concurrent.futures
import logging
import argparse
import sys
import pickle
import random
from copy import deepcopy

from functools import lru_cache
from collections import Counter

from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv
import numpy as np

# For MuZero
import torch
from muzero.mcts import MCTS
from muzero.mcts_batched import BatchedMCTS
from muzero.models import MuZeroNetwork

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('muzero_games.log')
    ]
)

# Configure logger
logger = logging.getLogger("MuZero-GameGeneration")

class GameHistory:
    """Simple container for game history data"""
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.done = False
        
    def add_step(self, observation, action, reward):
        """Add a step to the game history"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def __getitem__(self, key):
        return getattr(self, key)

def print_board_state(board, dice=None, borne_off_white=0, borne_off_black=0, debug=False):
    """Print the Narde board in a readable format for debugging."""
    if not isinstance(board, np.ndarray):
        logger.warning(f"Board is not a numpy array: {type(board)}")
        return
    
    # Format board display
    if board.ndim == 1:  # Linear board
        board_str = "Linear board format:\n"
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
            board_str += f"Row {i//8}: {row_str}\n"
    else:
        board_str = f"Unexpected board dimension: {board.ndim}\n"
        board_str += f"Raw board data: {board}\n"
        
    if dice:
        board_str += f"Dice: {dice}\n"
        
    # Add bearing off counts
    board_str += f"Borne off - White: {borne_off_white}/15, Black: {borne_off_black}/15\n"
    
    # Count pieces for each player
    white_count = np.sum(board > 0)
    black_count = np.sum(board < 0)
    board_str += f"Piece count - White: {white_count}, Black: {abs(black_count)}\n"
    
    if debug:
        logger.debug(board_str)
    else:
        logger.info(board_str)
    
    return board_str

def load_muzero_model(model_path, config=None):
    """
    Load a MuZero model from the given path.
    
    Args:
        model_path: Path to the model file
        config: Configuration dictionary (optional)
        
    Returns:
        tuple: (network, config) - The loaded network and updated config
    """
    if not config:
        config = {}
        
    # Default values
    config.setdefault('device', 'cpu')
    
    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location=config['device'])
        
        # Try to infer dimensions from the checkpoint
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Detect input dimension from the first representation network layer
        input_dim = None
        action_dim = None
        hidden_dim = 256  # Increased to match the saved model dimension
        
        # Look for typical layers that might indicate dimensions
        for key, value in state_dict.items():
            if 'representation_network.0.weight' in key:
                input_dim = value.shape[1]
                logger.info(f"Detected input dimension: {input_dim}")
                
            if 'prediction_network.policy_logits.weight' in key:
                action_dim = value.shape[0]
                logger.info(f"Detected action dimension: {action_dim}")
                
            # Try to detect hidden dimension
            if 'representation_network.fc.6.weight' in key:
                hidden_dim = value.shape[0]
                logger.info(f"Detected hidden dimension: {hidden_dim}")
            
        if input_dim is None:
            input_dim = config.get('input_dim', 28)  # Default for Narde
            logger.warning(f"Could not detect input dimension, using default: {input_dim}")
            
        if action_dim is None:
            action_dim = config.get('action_dim', 576)  # Default for Narde (24x24)
            logger.warning(f"Could not detect action dimension, using default: {action_dim}")
            
        # Update config with detected dimensions
        config['input_dim'] = input_dim
        config['action_dim'] = action_dim
        config['hidden_dim'] = hidden_dim
        
        logger.info(f"Creating MuZero network with input_dim={input_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        
        # Create network with the appropriate dimensions
        network = MuZeroNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # Load weights
        try:
            # Use non-strict loading to handle any minor mismatches
            missing, unexpected = network.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys in state_dict: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in state_dict: {unexpected}")
                
            logger.info(f"Successfully loaded model weights from {model_path} (non-strict)")
                
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            logger.error(traceback.format_exc())
            return None, config
            
        # Set network to evaluation mode
        network.eval()
        
        return network, config
        
    except Exception as e:
        logger.error(f"Error loading MuZero model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, config

def select_move_with_muzero(env, network, valid_moves, dice, config, logger):
    """
    Select a move using the MuZero network.
    
    Args:
        env: The game environment
        network: The MuZero network
        valid_moves: List of valid moves
        dice: The current dice values
        config: Configuration dictionary
        logger: Logger object
        
    Returns:
        The selected move and its action index
    """
    # If there's no network provided, fall back to random selection
    if network is None or not valid_moves:
        logger.warning("No MuZero network provided or no valid moves, selecting randomly")
        if not valid_moves:
            return None, None
        return random.choice(valid_moves), None
    
    try:
        # Get the current observation from the environment
        obs = env._get_obs()
        
        # Map valid moves to action indices
        valid_action_indices = []
        valid_moves_mapping = {}
        
        for move in valid_moves:
            from_pos, to_pos = move
            if to_pos == 'off':
                # Bearing off move
                move_type = 1
                action_idx = from_pos * 24
            else:
                # Regular move
                move_type = 0
                action_idx = from_pos * 24 + to_pos
                
            valid_action_indices.append(action_idx)
            valid_moves_mapping[action_idx] = move
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).to(config.get('device', 'cpu'))
        
        # Run MCTS
        mcts = MCTS(
            network=network,
            num_simulations=config.get('num_simulations', 50),
            discount=config.get('discount', 0.99),
            dirichlet_alpha=config.get('dirichlet_alpha', 0.3),
            exploration_fraction=config.get('exploration_fraction', 0.25),
            action_space_size=config.get('action_dim', 576),
            device=config.get('device', 'cpu')
        )
        
        # Fix tensor shape issue - convert valid_action_indices to proper tensor format
        valid_action_indices_tensor = torch.tensor(valid_action_indices, dtype=torch.long, 
                                                   device=config.get('device', 'cpu'))
        
        try:
            # Use the fixed tensor format for valid actions
            root = mcts.run(obs_tensor, valid_action_indices_tensor)
            
            # Get visit counts or policy
            if hasattr(root, 'children'):
                # Extract visit counts from tree
                visit_counts = torch.zeros(config.get('action_dim', 576), 
                                          device=config.get('device', 'cpu'))
                
                for action, child in root.children.items():
                    visit_counts[action] = child.visit_count
            else:
                # If root is a policy vector, use it directly
                visit_counts = root
                
            # Select action based on visit counts (temperature-based)
            temperature = config.get('temperature', 1.0)
            
            if temperature == 0:
                # Deterministic selection
                if isinstance(visit_counts, torch.Tensor):
                    action_index = torch.argmax(visit_counts).item()
                else:
                    action_index = np.argmax(visit_counts)
            else:
                # Stochastic selection based on visit count distribution
                if isinstance(visit_counts, torch.Tensor):
                    visit_counts = visit_counts ** (1 / temperature)
                    visit_counts = visit_counts / visit_counts.sum()
                    action_probs = visit_counts.cpu().numpy()
                else:
                    visit_counts = visit_counts ** (1 / temperature)
                    visit_counts = visit_counts / visit_counts.sum()
                    action_probs = visit_counts
                
                # Only consider valid actions
                valid_action_probs = np.zeros_like(action_probs)
                valid_action_probs[valid_action_indices] = action_probs[valid_action_indices]
                valid_action_probs = valid_action_probs / valid_action_probs.sum()
                
                # Sample from the distribution
                action_index = np.random.choice(len(valid_action_probs), p=valid_action_probs)
            
            # Map back to move
            selected_move = valid_moves_mapping.get(action_index)
            
            if selected_move is None:
                logger.warning(f"Selected action {action_index} not in valid moves mapping, selecting randomly")
                selected_move = random.choice(valid_moves)
                
            logger.debug(f"MCTS selected move: {selected_move} (action: {action_index})")
            return selected_move, action_index
            
        except Exception as e:
            logger.error(f"Error in MCTS search: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("MCTS selection failed, falling back to random selection")
            return random.choice(valid_moves), None
            
    except Exception as e:
        logger.error(f"Error in MuZero move selection: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("MuZero selection failed, falling back to random selection")
        return random.choice(valid_moves), None

def play_game_with_muzero(env_seed=42, max_steps=300, model_path=None, config=None, temperature=1.0, debug=False):
    """
    Play a complete game using MuZero for move selection if a model is provided.
    
    Args:
        env_seed: Seed for the environment
        max_steps: Maximum steps to play
        model_path: Path to the MuZero model (if None, will use random selection)
        config: Configuration for MuZero
        temperature: Temperature for action selection
        debug: Whether to log debug information
        
    Returns:
        tuple: Game statistics
    """
    # Configure logging
    global logger
    
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        
    # Set random seeds
    np.random.seed(env_seed)
    random.seed(env_seed)
        
    # Load model if path provided
    network = None
    if model_path and not config:
        config = {}  # Initialize empty config
        
    if model_path:
        try:
            network, config = load_muzero_model(model_path, config)
            if network is None:
                logger.info("Using random move selection")
        except Exception as e:
            logger.error(f"Error loading model, using random selection: {str(e)}")
            
    # Create and seed environment
    env = NardeEnv()
    
    # Play the game
    result = play_complete_game(
        env=env,
        max_steps=max_steps,
        debug=debug,
        use_muzero=(network is not None),
        network=network,
        config=config,
        logger=logger
    )
    
    return result

def game_worker(worker_id, model_path, config, num_games, max_steps, result_queue, debug=False):
    """
    Worker process to generate games in parallel
    
    Args:
        worker_id (int): ID of the worker
        model_path (str): Path to MuZero model
        config (MuZeroConfig): MuZero configuration
        num_games (int): Number of games to generate
        max_steps (int): Maximum steps per game
        result_queue (Queue): Queue to store results
        debug (bool): Whether to print detailed debug information
    """
    logger.info(f"Worker {worker_id} started, generating {num_games} games")
    
    worker_results = []
    
    for i in range(num_games):
        seed = worker_id * 1000 + i  # Ensure unique seeds across workers
        try:
            elapsed_time, game_history = play_game_with_muzero(
                env_seed=seed,
                max_steps=max_steps,
                model_path=model_path,
                config=config,
                debug=debug
            )
            
            # Store game data
            game_data = {
                'game_id': i,
                'worker_id': worker_id,
                'num_moves': len(game_history.moves),
                'no_move_steps': game_history.no_move_steps,
                'bearing_off_moves': game_history.bearing_off_moves,
                'white_borne_off': game_history.white_borne_off,
                'black_borne_off': game_history.black_borne_off,
                'time': elapsed_time,
                'end_reason': game_history.end_reason
            }
            
            worker_results.append(game_data)
            
            logger.info(f"Worker {worker_id} - Game {i+1}/{num_games} completed in {elapsed_time:.2f}s with {len(game_history.moves)} moves")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} - Error in game {i+1}: {e}")
    
    # Put results in queue
    result_queue.put(worker_results)
    logger.info(f"Worker {worker_id} finished")

def generate_games_parallel(num_games=10, max_steps=300, model_path=None, config=None, num_workers=None, debug=False):
    """
    Generate games in parallel using multiple workers
    
    Args:
        num_games (int): Total number of games to generate
        max_steps (int): Maximum steps per game
        model_path (str): Path to MuZero model
        config (MuZeroConfig): MuZero configuration
        num_workers (int): Number of worker processes (default: CPU count)
        debug (bool): Whether to print detailed debug information
        
    Returns:
        dict: Statistics about the games
    """
    start_time = time.time()
    
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Ensure we don't create more workers than games
    num_workers = min(num_workers, num_games)
    
    # Calculate games per worker
    games_per_worker = [num_games // num_workers] * num_workers
    # Distribute remainder
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1
    
    logger.info(f"Generating {num_games} games using {num_workers} workers")
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start workers
    processes = []
    for worker_id in range(num_workers):
        process = mp.Process(
            target=game_worker,
            args=(
                worker_id,
                model_path,
                config,
                games_per_worker[worker_id],
                max_steps,
                result_queue,
                debug
            )
        )
        processes.append(process)
        process.start()
    
    # Collect results
    all_results = []
    for _ in range(num_workers):
        worker_results = result_queue.get()
        all_results.extend(worker_results)
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    # Calculate overall statistics
    total_time = time.time() - start_time
    total_moves = sum(g['num_moves'] for g in all_results)
    total_bearing_off = sum(g['bearing_off_moves'] for g in all_results)
    total_no_move_steps = sum(g['no_move_steps'] for g in all_results)
    
    avg_moves = total_moves / num_games if num_games > 0 else 0
    avg_bearing_off = total_bearing_off / num_games if num_games > 0 else 0
    avg_no_move_steps = total_no_move_steps / num_games if num_games > 0 else 0
    games_per_second = num_games / total_time if total_time > 0 else 0
    
    # Count end reasons
    end_reasons = {}
    for game in all_results:
        end_reason = game['end_reason'] or "unknown"
        end_reasons[end_reason] = end_reasons.get(end_reason, 0) + 1
    
    # Print summary
    logger.info("\n=== GAME GENERATION RESULTS ===")
    logger.info(f"Total games: {num_games}")
    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Average moves per game: {avg_moves:.1f}")
    logger.info(f"Average bearing off moves per game: {avg_bearing_off:.1f}")
    logger.info(f"Average no-move steps per game: {avg_no_move_steps:.1f}")
    logger.info(f"Games per second: {games_per_second:.2f}")
    logger.info(f"End reasons: {end_reasons}")
    
    return {
        'num_games': num_games,
        'total_time': total_time,
        'avg_moves': avg_moves,
        'avg_bearing_off': avg_bearing_off,
        'avg_no_move_steps': avg_no_move_steps,
        'games_per_second': games_per_second,
        'end_reasons': end_reasons,
        'games_data': all_results
    }

def get_muzero_config(model_path):
    """
    Create a MuZero configuration based on model path
    
    Args:
        model_path (str): Path to MuZero model
        
    Returns:
        MuZeroConfig: MuZero configuration
    """
    config = MuZeroConfig()
    config.action_space_size = 24 * 24 + 24  # Regular moves + bearing off moves
    config.observation_shape = (24,)  # Linear board representation
    config.max_moves = 300
    config.discount = 0.99
    
    # MCTS parameters - using modest values for efficiency
    config.num_simulations = 50
    config.root_dirichlet_alpha = 0.3
    config.root_exploration_fraction = 0.25
    
    return config

# Create a simple config class since there isn't a MuZeroConfig in the codebase
class MuZeroConfig:
    """Configuration for MuZero"""
    def __init__(self):
        self.action_space_size = 24 * 24 + 24  # Regular moves + bearing off moves
        self.observation_shape = (24,)  # Linear board representation
        self.max_moves = 300
        self.discount = 0.99
        
        # MCTS parameters
        self.num_simulations = 50
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

def play_complete_game(env, max_steps=500, debug=True, use_muzero=False, network=None, config=None, logger=None):
    """
    Play a complete game of Narde, potentially using MuZero for move selection.
    
    This function plays a game from start to finish, selecting moves either randomly
    or using a trained MuZero network. It returns information about the game including
    the moves made, bearing off progress, and more.
    
    Args:
        env: The game environment.
        max_steps: Maximum number of steps to play.
        debug: Whether to print debug information.
        use_muzero: Whether to use MuZero for move selection.
        network: The MuZero network (if use_muzero is True).
        config: Configuration for MuZero (if use_muzero is True).
        logger: Logger to use for logging.
        
    Returns:
        tuple: (steps, moves_made, bearing_off_moves, bearing_off_progress, white_borne_off, black_borne_off, history, end_reason)
    """
    if logger is None:
        logger = logging.getLogger("MuZero-GameGeneration")
        
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        
    # Initialize counters
    steps = 0
    moves_made = 0
    no_move_counter = 0
    bearing_off_moves = 0
    consecutive_skips = 0
    
    # Reset environment
    obs = env.reset()
    
    # Create game history to track states
    game_history = GameHistory()
    
    # Track bearing off progress
    bearing_off_progress = {
        "white": 0,
        "black": 0,
    }
    
    # Track total bearing off
    total_bearing_off = {"white": 0, "black": 0}
    
    # Game end reason
    end_reason = "completed"
    
    # Initialize last 50 steps bearing off tracking
    last_bearing_off_check = {"steps": 0, "white": 0, "black": 0}
    
    # Track stagnation
    stagnation_warning_count = 0
    
    while steps < max_steps:
        # Get current state
        observation = env._get_obs()
        
        # Get valid moves from the environment
        valid_moves = env.game.get_valid_moves()
        
        # Log game state and valid moves for debugging
        if debug:
            logger.debug(f"Step {steps} - Found {len(valid_moves)} valid moves with dice {env.dice}")
            if valid_moves and len(valid_moves) < 10:  # Only show if not too many
                logger.debug(f"Valid moves: {valid_moves}")
            
        # If no valid moves, the environment will handle rotation automatically
        if not valid_moves:
            no_move_counter += 1
            consecutive_skips += 1
            game_history.no_move_steps += 1
            
            # Print board state when both players skip moves consecutively
            if consecutive_skips >= 2:
                logger.warning(f"Both players skipped moves consecutively. Consecutive skips: {consecutive_skips}")
                logger.warning(f"Current board state: {env.game.board}")
                logger.warning(f"Current dice: {env.dice}")
                
                # Print more detailed board information
                board_str = print_board_state(
                    board=env.game.board,
                    dice=env.dice,
                    borne_off_white=env.game.borne_off_white,
                    borne_off_black=env.game.borne_off_black,
                    debug=False
                )
                logger.warning(f"Detailed board state:\n{board_str}")
            else:
                # Log more information about why there are no valid moves
                logger.warning(f"Step {steps} - No valid moves with dice {env.dice}")
                
                # Log pieces that can be moved (non-zero positions for current player)
                pieces = [i for i, val in enumerate(env.game.board) if val > 0]
                logger.warning(f"Step {steps} - Pieces at positions: {pieces}")
                
                # Check if there are any dice values that match the moves that would be needed
                for piece_pos in pieces:
                    for die in env.dice:
                        target_pos = piece_pos - die  # Try moving by the die value
                        if 0 <= target_pos < 24:  # Check if the target position is on the board
                            # Check if there is a blocking piece
                            if env.game.board[target_pos] < 0:  # Opponent's piece
                                logger.warning(f"Step {steps} - Move from {piece_pos} to {target_pos} blocked by opponent's piece")
            
            if debug:
                logger.debug(f"Step {steps} - No valid moves with dice {env.dice}, skipping turn")
                
            # Pass a dummy action to let the environment handle turn skipping
            dummy_action = (0, 0)  # This will be ignored by the environment
            _, _, done, info = env.step(dummy_action)
            
            steps += 1
            continue
        
        # Reset consecutive skips counter when valid moves are available
        consecutive_skips = 0
        
        # Select move using MuZero or prioritization
        if use_muzero and network and config:
            move, action_index = select_move_with_muzero(env, network, valid_moves, env.dice, config, logger)
            if move is None:
                # No valid moves or MuZero failed
                end_reason = "no_valid_moves"
                break
        else:
            # Prioritize moves to encourage progression
            move = prioritize_moves(env, valid_moves, game_history)
            
        if debug:
            logger.debug(f"Step {steps} - Selected move: {move}")
            
        # Determine if it's a bearing off move
        is_bearing_off = len(move) == 2 and move[1] == 'off'
        
        # Convert move to action format expected by environment
        from_pos, to_pos = move
        if to_pos == 'off':
            # Bearing off move
            move_type = 1
            move_index = from_pos * 24
        else:
            # Regular move
            move_type = 0
            move_index = from_pos * 24 + to_pos
            
        action = (move_index, move_type)
        
        # Execute the move
        _, reward, done, info = env.step(action)
        
        # Record state to detect repetition
        current_board = env.game.board
        game_history.add_step(current_board, action, reward)
        
        # Track bearing off progress
        if is_bearing_off:
            bearing_off_moves += 1
            current_player = "white"  # Always white from perspective
            bearing_off_progress[current_player] += 1
            total_bearing_off[current_player] += 1
            
        # If we're in the later part of the game (after step 50), check bearing off progress
        if steps > 50 and steps % 50 == 0:
            # Check progress in the last 50 steps
            white_progress = total_bearing_off["white"] - last_bearing_off_check["white"]
            black_progress = total_bearing_off["black"] - last_bearing_off_check["black"]
            steps_since_last_check = steps - last_bearing_off_check["steps"]
            
            # Update the checkpoint
            last_bearing_off_check = {
                "steps": steps,
                "white": total_bearing_off["white"],
                "black": total_bearing_off["black"]
            }
            
            # If no bearing off in last 50 steps, issue warning
            if white_progress == 0 and black_progress == 0 and steps_since_last_check >= 50:
                stagnation_warning_count += 1
                logger.warning(f"Game stagnating at steps {steps-50} to {steps} - no bearing off occurring")
                
                # If stagnating for too long, end the game
                if stagnation_warning_count >= 3:
                    logger.warning(f"Game terminated due to insufficient bearing off progress")
                    end_reason = "insufficient_bearing_off"
                    break
        
        # Log state after step
        if debug:
            logger.debug(f"Step {steps} - Reward: {reward}, Done: {done}")
            if info:
                logger.debug(f"Info: {info}")
            logger.debug(f"=== AFTER STEP {steps} ===")
            logger.debug(f"Board: {env.game.board}")
            logger.debug(f"Dice: {env.dice}")
            logger.debug(f"Borne off - White: {env.game.borne_off_white}/15, Black: {env.game.borne_off_black}/15")
            logger.debug("")
            
        steps += 1
        moves_made += 1
        
        # Check if the game is done
        if done:
            end_reason = "completed"
            break
            
    # If we've reached the maximum steps, log it
    if steps >= max_steps:
        logger.warning(f"Game terminated after reaching max steps ({max_steps})")
        end_reason = "max_steps_reached"
        
    # Log final game state
    logger.info(f"=== FINAL GAME STATE ===")
    logger.info(f"Board: {env.game.board}")
    logger.info(f"Dice: {env.dice}")
    logger.info(f"Borne off - White: {env.game.borne_off_white}/15, Black: {env.game.borne_off_black}/15")
    logger.info("")
    
    # Return game statistics
    return (
        steps,
        moves_made,
        bearing_off_moves,
        bearing_off_progress,
        env.game.borne_off_white,
        env.game.borne_off_black,
        game_history,
        end_reason
    )

def run_game(agent1, agent2, config, game_id=0, seed=None, debug=False):
    """
    Run a single game between two agents.
    
    Args:
        agent1: First agent
        agent2: Second agent
        config: Configuration dictionary
        game_id: ID for the game
        seed: Random seed
        debug: Whether to enable debug logging
        
    Returns:
        Game result dictionary
    """
    try:
        # Set up logging
        logger = logging.getLogger(f"Game-{game_id}")
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Create environment
        env = NardeEnv(debug=debug)
        
        # Initialize game state
        obs = env.reset()
        done = False
        
        # Clear any initial dice rolls that happened during initialization
        # This ensures we start with fresh dice on the first turn
        logger.debug("Initializing environment and clearing any initial dice")
        
        game_history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dices': [],
            'valid_moves': []
        }
        
        # Store initial state
        observation = env._get_obs()
        game_history['states'].append(observation)
        
        num_moves = 0
        start_time = time.time()
        max_moves = config.get('max_moves', 500) if isinstance(config, dict) else 500  # Maximum number of moves allowed
        
        # Track consecutive skips
        consecutive_skips = 0
        
        # Game loop
        current_agent = agent1
        agent_map = {0: agent1, 1: agent2}
        current_player = 0  # Start with player 0 (white)
        
        while not done and num_moves < max_moves:
            # Get fresh dice roll for each turn by using the environment's dice rolling mechanism
            if hasattr(env, 'roll_dice'):
                env.roll_dice()
            else:
                # Manual dice rolling if the method doesn't exist
                env.dice = [random.randint(1, 6) for _ in range(2)]
                
                # Make sure the Narde game object has the same dice
                if hasattr(env, 'game') and hasattr(env.game, 'dice'):
                    env.game.dice = env.dice.copy()
            
            dice = env.dice.copy() if hasattr(env, 'dice') else [1, 1]  # Fallback if dice not found
            logger.debug(f"Turn {num_moves} - New dice roll: {dice}")
            
            # Verify dice are synchronized between env and game objects
            if hasattr(env, 'game') and hasattr(env.game, 'dice'):
                if not np.array_equal(env.dice, env.game.dice):
                    logger.warning(f"Dice mismatch! env.dice={env.dice}, env.game.dice={env.game.dice}")
                    # Synchronize the dice
                    env.game.dice = env.dice.copy()
                    
            game_history['dices'].append(dice)
            
            # Process both dice sequentially to complete one turn
            remaining_dice = dice.copy()
            
            while remaining_dice and not done:
                # Get valid moves for the current dice state
                valid_moves = env.game.get_valid_moves()
                
                logger.debug(f"Remaining dice: {remaining_dice}, Valid moves: {valid_moves}")
                
                if not valid_moves:
                    # No valid moves, skip remaining dice
                    logger.debug(f"No valid moves for remaining dice {remaining_dice}, skipping")
                    action = None
                    observation, reward, done, truncated, info = env.step(action)
                    logger.debug(f"Step result - done: {done}, reward: {reward}, info: {info}")
                    
                    # Exit the dice processing loop since we've skipped the turn
                    break
                else:
                    # Select move based on current agent
                    if current_agent == "random":
                        selected_move = random.choice(valid_moves)
                        action_idx = None
                        logger.debug(f"Random agent selected move: {selected_move}")
                    elif current_agent == "muzero":
                        # Using MuZero for move selection
                        selected_move, action_idx = select_move_with_muzero(env, config.get('network'), valid_moves, remaining_dice, config, logger)
                        logger.debug(f"MuZero agent selected move: {selected_move}")
                    else:
                        raise ValueError(f"Unknown agent type: {current_agent}")
                    
                    if selected_move is None:
                        # No move selected, skip turn
                        logger.debug("No move selected, skipping turn")
                        action = None
                    else:
                        from_pos, to_pos = selected_move
                        if to_pos == 'off':
                            # Bearing off move
                            action = (from_pos * 24, 1)  # Use tuple format with move_type=1
                            logger.debug(f"Bearing off from position {from_pos}")
                        else:
                            # Regular move
                            action = from_pos * 24 + to_pos
                            logger.debug(f"Regular move from {from_pos} to {to_pos}")
                    
                    logger.debug(f"Selected move: {selected_move}, Action: {action}")
                    
                    # Verify the move is valid before executing
                    if selected_move not in valid_moves:
                        logger.warning(f"Selected move {selected_move} not in valid moves: {valid_moves}")
                        # Choose a random valid move instead
                        if valid_moves:
                            selected_move = random.choice(valid_moves)
                            from_pos, to_pos = selected_move
                            if to_pos == 'off':
                                # Bearing off move
                                action = (from_pos * 24, 1)
                            else:
                                # Regular move
                                action = from_pos * 24 + to_pos
                            logger.warning(f"Replaced with random valid move: {selected_move}, Action: {action}")
                        else:
                            # This shouldn't happen, but just in case
                            action = None
                            logger.warning("No valid moves available, skipping turn instead")
                    
                    # Execute move
                    observation, reward, done, truncated, info = env.step(action)
                    logger.debug(f"Step result - done: {done}, reward: {reward}, info: {info}")
                    
                    # Store move
                    game_history['actions'].append(action)
                    game_history['rewards'].append(reward)
                    game_history['states'].append(observation)
                    
                    num_moves += 1
                    
                    # Update remaining dice based on the info returned from the environment
                    if 'dice' in info:
                        remaining_dice = info['dice'] if isinstance(info['dice'], list) else [info['dice']]
                    else:
                        # If info doesn't contain dice, assume all dice are used
                        remaining_dice = []
                    
                    # Check if game is over
                    if done:
                        break
            
            # Check if game over or max moves reached
            if done:
                logger.debug(f"Game over! Reward: {reward}, Info: {info}")
                if 'winner' in info:
                    logger.info(f"Winner: {'White' if info['winner'] == 0 else 'Black'}")
                elif 'won' in info and info['won']:
                    logger.info(f"Current player won!")
                break
            
            # Check if max moves reached
            if num_moves >= max_moves:
                logger.warning(f"Game terminated after reaching max moves ({max_moves})")
                break
            
            # Switch to other agent only after all dice are used or turn is skipped
            current_player = 1 - current_player
            current_agent = agent_map[current_player]
            
            logger.debug(f"Current player: {current_player}, Current agent: {current_agent}")
        
        # Game finished
        end_time = time.time()
        
        # Get winner - if game ended due to max moves, consider it a draw
        if num_moves >= max_moves:
            logger.warning(f"Game reached maximum moves ({max_moves}), ending as a draw")
            winner = -1  # Draw
        else:
            # Get winner from info
            if 'winner' in info:
                winner = info['winner']
            elif reward > 0:
                # If reward is positive, current player won
                winner = current_player
            elif 'won' in info and info['won']:
                winner = current_player
            else:
                # Default to draw
                winner = -1
        
        # Count pieces borne off
        white_borne_off = observation[25]
        black_borne_off = observation[26]
        
        # Record game result
        result = {
            'game_id': game_id,
            'seed': seed,
            'num_moves': num_moves,
            'winner': winner,
            'white_borne_off': white_borne_off,
            'black_borne_off': black_borne_off,
            'elapsed_time': end_time - start_time,
            'game_history': game_history
        }
        
        logger.info(f"Game {game_id} finished: Winner={winner}, Moves={num_moves}, White borne off={white_borne_off}, Black borne off={black_borne_off}, Time={end_time-start_time:.2f}s")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in game {game_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'game_id': game_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

class BatchedGameSimulator:
    """
    Simulates multiple games in parallel, using batched MCTS inference on GPU.
    Handles dynamic batch sizing and ensures inference runs even when batch isn't full.
    """
    def __init__(self, num_games, model_path, batch_size=32, max_games=100, device="cuda", num_simulations=5, debug=False):
        self.num_games = num_games
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_games = max_games
        self.device = device
        self.debug = debug
        self.logger = self._setup_logger()
        self.num_simulations = num_simulations
        self.max_moves_per_game = 500  # Limit games to 500 moves

        # Initialize active games
        self.env_list = []
        self.game_histories = []
        self.completed_games = []  # Add this attribute to store completed games
        self.consecutive_skips = {}  # Track consecutive skips for each game
        
        # Initialize the model
        self.model = self._load_model()
        
        # Initialize batched MCTS
        self.mcts = BatchedMCTS(self.model, device=self.device, num_simulations=self.num_simulations)
        
        self.logger.info(f"Initialized BatchedGameSimulator with device={self.device}, max_batch_size={self.batch_size}")
        
        # Initialize games
        self.logger.info(f"Initializing {self.num_games} active games...")
        self._initialize_games()

    def _setup_logger(self):
        logger = logging.getLogger("BatchedGameSimulator")
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger

    def _load_model(self):
        network, _ = load_muzero_model(self.model_path)
        network.to(self.device)
        return network

    def _initialize_games(self):
        for i in range(self.num_games):
            env = NardeEnv(debug=self.debug)  # Pass debug flag to environment
            obs, _ = env.reset()
            self.env_list.append(env)
            self.game_histories.append(GameHistory())
            self.consecutive_skips[i] = 0  # Initialize consecutive skips counter
            self.completed_games.append({
                "env": env,
                "observation": obs,
                "game_history": self.game_histories[-1],
                "done": False,
                "game_id": i,
                "move_count": 0  # Track number of moves for each game
            })

    def run(self):
        """Run multiple games in parallel with batched MCTS inference"""
        start_time = time.time()
        active_games = []
        finished_games = []
        
        # Convert env_list to the active_games format expected by _process_game_batch
        for i, env in enumerate(self.env_list):
            observation = env.reset()[0]  # Get the initial observation
            active_games.append({
                "env": env,
                "observation": observation,
                "game_history": self.game_histories[i],
                "done": False,
                "game_id": i,
                "move_count": 0
            })
        
        self.logger.info(f"Starting simulation with {len(active_games)} active games")
        
        step_count = 0
        while active_games:
            step_count += 1
            
            # Process games in batches to avoid memory issues
            batch_size = min(self.batch_size, len(active_games))
            
            # Process active games
            self._process_game_batch(active_games, finished_games, step_count)
            
            # Remove finished games from active_games
            active_games = [game for game in active_games if not game["done"]]
            
            # Stop if we've reached the game limit
            if len(finished_games) >= self.num_games:
                break
        
        # Calculate statistics
        duration = time.time() - start_time
        self.logger.info(f"Completed {len(finished_games)} games in {duration:.2f} seconds ({duration/max(1, len(finished_games)):.2f} seconds/game)")
        
        # Save game data
        timestamp = int(time.time())
        os.makedirs("muzero_training/games", exist_ok=True)
        filename = f"muzero_training/games/muzero_games_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.game_histories, f)
        self.logger.info(f"Game data saved to {filename}")
        
        return self.game_histories
    
    def _process_game_batch(self, active_games, finished_games, step_count):
        """Process a batch of games, running MCTS for games that need moves."""
        
        # Count active games with and without moves
        games_with_moves = []
        games_without_moves = []
        
        for i, game in enumerate(active_games[:]):
            if not game["done"]:
                env = game["env"]
                valid_moves = env.game.get_valid_moves()
                
                if len(valid_moves) > 0:
                    games_with_moves.append(game)
                    # Reset skip counter when the game has valid moves
                    self.consecutive_skips[game["game_id"]] = 0
                else:
                    games_without_moves.append(game)
                    # For games with no valid moves, log basic dice and board info
                    self.logger.warning(f"Game {game['game_id']} has no valid moves - Dice: {env.dice}")
                    
                    # Log the board state
                    board_state = env.game.board
                    self.logger.warning(f"Game {game['game_id']} - Board state: {board_state}")
                    
                    # Log pieces that can be moved (non-zero positions for current player)
                    pieces = [i for i, val in enumerate(board_state) if val > 0]
                    self.logger.warning(f"Game {game['game_id']} - Pieces at positions: {pieces}")
                    
                    # Check bearing off status
                    white_borne_off = env.game.borne_off_white if hasattr(env.game, 'borne_off_white') else 0
                    black_borne_off = env.game.borne_off_black if hasattr(env.game, 'borne_off_black') else 0
                    self.logger.warning(f"Game {game['game_id']} - Bearing off status - White: {white_borne_off}/15, Black: {black_borne_off}/15")
                    
                    # We trust the environment's determination that there are no valid moves
                    # No need to run redundant diagnostics

        # Log the counts
        self.logger.info(f"Step {step_count}: Active games: {len(active_games)}, With moves: {len(games_with_moves)}, Without moves: {len(games_without_moves)}, Finished: {len(finished_games)}/{self.num_games}")
        
        # Track dice distribution for games without moves if there's a significant number
        if len(games_without_moves) > len(active_games) * 0.05:  # More than 5% of active games have no moves
            dice_values = []
            for game in games_without_moves:
                dice_values.extend(game["env"].dice)
            dice_counts = Counter(dice_values)
            self.logger.warning(f"Step {step_count}: Games without moves - Dice distribution: {dict(dice_counts)}")
        
        # Detect stalled situation - active games but none categorized
        if len(active_games) > 0 and len(games_with_moves) == 0 and len(games_without_moves) == 0:
            self.logger.warning(f"Step {step_count}: Detected stalled situation - {len(active_games)} active games but none categorized correctly")
            # Force all active games to be treated as games without moves so they can progress
            for game in active_games[:]:
                if not game["done"]:
                    games_without_moves.append(game)
                    self.logger.warning(f"Game {game['game_id']} forced to no-move state to break stall")
        
        # Process games with valid moves using batched MCTS
        if games_with_moves:
            self.logger.info(f"Step {step_count}: Running MCTS with batch size {len(games_with_moves)}")
            self._process_games_with_moves(games_with_moves, finished_games, active_games)
        
        # Process games without valid moves (roll dice or end game)
        for game in games_without_moves:
            env = game["env"]
            game_id = game["game_id"]
            
            # Increment consecutive skips counter
            self.consecutive_skips[game_id] += 1
            
            # Print board state and dice roll when both players skip a move
            if self.consecutive_skips[game_id] >= 2:
                board = env.game.board
                dice = env.dice
                self.logger.warning(f"Game {game_id} - Both players skipped moves. Consecutive skips: {self.consecutive_skips[game_id]}")
                self.logger.warning(f"Game {game_id} - Current board state: {board}")
                self.logger.warning(f"Game {game_id} - Current dice: {dice}")
                
                # Print more detailed information using the print_board_state function if available
                try:
                    board_str = print_board_state(
                        board=board, 
                        dice=dice, 
                        borne_off_white=env.game.borne_off_white,
                        borne_off_black=env.game.borne_off_black,
                        debug=False
                    )
                    self.logger.warning(f"Game {game_id} - Detailed board state:\n{board_str}")
                except Exception as e:
                    self.logger.error(f"Error printing detailed board state: {str(e)}")
                    
                # If we have too many consecutive skips, force end the game
                if self.consecutive_skips[game_id] >= 10:
                    self.logger.warning(f"Game {game_id} - Ending due to too many consecutive skips ({self.consecutive_skips[game_id]})")
                    game["done"] = True
                    finished_games.append(game)
                    active_games.remove(game)
                    continue
            
            # Take a "no-move" action to advance to next player
            observation, reward, done, truncated, info = env.step(None)
            game["observation"] = observation
            
            # Store in game history
            game["game_history"].add_step(observation, None, reward)
            
            # Increment move count and check if we've reached the limit
            game["move_count"] += 1
            if game["move_count"] >= self.max_moves_per_game:
                self.logger.info(f"Game {game_id} reached move limit of {self.max_moves_per_game}, ending game")
                game["done"] = True
                finished_games.append(game)
                active_games.remove(game)
                continue
            
            if done:
                # Get more detailed information about the game state
                white_borne_off = env.game.borne_off_white if hasattr(env.game, 'borne_off_white') else 0
                black_borne_off = env.game.borne_off_black if hasattr(env.game, 'borne_off_black') else 0
                
                # Only mark game as truly completed if pieces have been borne off
                # or if we've reached a substantial number of moves
                if white_borne_off > 0 or black_borne_off > 0 or game["move_count"] > 10:
                    self.logger.info(f"Game {game_id} completed after {len(game['game_history'].actions)} moves. Winner: {info.get('winner', 'Unknown')}. Bearing off - White: {white_borne_off}/15, Black: {black_borne_off}/15")
                    game["done"] = True
                    finished_games.append(game)
                    active_games.remove(game)
                else:
                    # If game is marked as done but no pieces borne off and very few moves,
                    # this likely indicates an early termination bug - log and continue the game
                    self.logger.warning(f"Game {game_id} marked as done after only {len(game['game_history'].actions)} moves with no pieces borne off. This is likely incorrect. Continuing game.")
                    # Keep the game going by setting done to False
                    done = False
                    game["done"] = False

    def _process_games_with_moves(self, games, finished_games=None, active_games=None):
        """Use batched MCTS to get actions for games with valid moves.
        
        Args:
            games: List of games with valid moves
            finished_games: List to store finished games (optional)
            active_games: List of active games to remove finished games from (optional)
        """
        
        batch_size = len(games)
        observations = []
        valid_actions_list = []
        raw_valid_moves_list = []  # Store the raw valid moves for debugging
        
        # Collect observations and valid moves
        for game in games:
            env = game["env"]
            observation = game["observation"]
            valid_moves = env.game.get_valid_moves()
            
            # Debug log raw valid moves
            self.logger.debug(f"Game {game['game_id']} - Valid moves: {valid_moves}")
            raw_valid_moves_list.append(valid_moves)
            
            # Convert to action indices
            valid_actions = []
            for from_pos, to_pos in valid_moves:
                if to_pos == 'off':
                    # Bearing off
                    action_idx = from_pos * 24
                    valid_actions.append(action_idx)
                else:
                    # Regular move
                    action_idx = from_pos * 24 + to_pos
                    valid_actions.append(action_idx)
            
            self.logger.debug(f"Game {game['game_id']} - Valid action indices: {valid_actions}")
            
            observations.append(observation)
            valid_actions_list.append(valid_actions)
        
        # Convert observations to numpy arrays first if they're not already
        for i in range(len(observations)):
            if not isinstance(observations[i], np.ndarray):
                observations[i] = np.array(observations[i], dtype=np.float32)
        
        # Convert to tensor format
        batch_observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        
        # Run batched MCTS
        with torch.no_grad():
            batch_actions = []
            for i in range(batch_size):
                # For each game, run MCTS and select actions
                policy = self.mcts.run(batch_observations[i], valid_actions_list[i])
                action = np.random.choice(len(policy), p=policy)
                batch_actions.append(action)
                self.logger.debug(f"Game {games[i]['game_id']} - Selected action: {action}, Policy shape: {policy.shape}")
                
                # Debug - check if selected action is in valid actions
                if action not in valid_actions_list[i]:
                    self.logger.error(f"Game {games[i]['game_id']} - CRITICAL ERROR: Selected action {action} not in valid actions {valid_actions_list[i]}")
                    # Log more details about the policy
                    top_actions = np.argsort(policy)[-5:][::-1]  # Top 5 actions by probability
                    self.logger.error(f"Game {games[i]['game_id']} - Top 5 actions in policy: {top_actions} with probs: {policy[top_actions]}")
        
        # Take actions in environments
        for i, game in enumerate(games):
            env = game["env"]
            action = batch_actions[i]
            
            # Store action in game history
            game_history = game["game_history"]
            game_history.add_step(game["observation"], action, None)
            
            # Log the action being applied
            self.logger.debug(f"Game {game['game_id']} - Taking action: {action}")
            from_pos = action // 24
            to_pos = action % 24
            self.logger.debug(f"Game {game['game_id']} - Decoded as move: from={from_pos}, to={to_pos}")
            
            # Take action in environment
            observation, reward, done, truncated, info = env.step(action)
            
            # Log whether the move was invalid
            if info.get("invalid_move", False):
                self.logger.error(f"Game {game['game_id']} - Action {action} was INVALID. Valid moves were: {raw_valid_moves_list[i]}")
            
            # Handle invalid moves by selecting a valid one instead
            if info.get("invalid_move", False):
                self.logger.warning(f"Game {game['game_id']} attempted invalid move {action}. Selecting a valid move instead.")
                # Get valid moves
                valid_moves = env.game.get_valid_moves()
                if valid_moves:
                    # Select a random valid move
                    valid_move = random.choice(valid_moves)
                    # Encode the valid move to an action
                    from_pos, to_pos = valid_move
                    move_type = 1 if to_pos == 'off' else 0
                    if move_type == 0:
                        # Regular move
                        new_action = from_pos * 24 + to_pos
                    else:
                        # Bearing off - arbitrary to_pos that will be ignored
                        new_action = from_pos * 24
                    
                    # Update the game history with the new action
                    game_history.actions[-1] = new_action
                    
                    # Take the corrected action
                    observation, reward, done, truncated, info = env.step((new_action, move_type))
                else:
                    # No valid moves, skip turn
                    self.logger.warning(f"Game {game['game_id']} has no valid moves. Skipping turn.")
                    # Skip by passing None
                    observation, reward, done, truncated, info = env.step(None)
            
            game["observation"] = observation
            
            # Store reward
            game_history.rewards[-1] = reward
            
            # Increment move count and check if we've reached the limit
            game["move_count"] += 1
            if game["move_count"] >= self.max_moves_per_game:
                self.logger.info(f"Game {game['game_id']} reached move limit of {self.max_moves_per_game}, ending game")
                game["done"] = True
                if finished_games is not None and active_games is not None:
                    finished_games.append(game)
                    active_games.remove(game)
                continue
            
            # Check if game is done
            if done:
                # Get more detailed information about the game state
                white_borne_off = env.game.borne_off_white if hasattr(env.game, 'borne_off_white') else 0
                black_borne_off = env.game.borne_off_black if hasattr(env.game, 'borne_off_black') else 0
                
                # Only mark game as truly completed if pieces have been borne off
                # or if we've reached a substantial number of moves
                if white_borne_off > 0 or black_borne_off > 0 or game["move_count"] > 10:
                    self.logger.info(f"Game {game['game_id']} completed after {len(game_history.actions)} moves. Winner: {info.get('winner', 'Unknown')}. Bearing off - White: {white_borne_off}/15, Black: {black_borne_off}/15")
                    game["done"] = True
                    finished_games.append(game)
                    active_games.remove(game)
                else:
                    # If game is marked as done but no pieces borne off and very few moves,
                    # this likely indicates an early termination bug - log and continue the game
                    self.logger.warning(f"Game {game['game_id']} marked as done after only {len(game_history.actions)} moves with no pieces borne off. This is likely incorrect. Continuing game.")
                    # Keep the game going by setting done to False
                    done = False
                    game["done"] = False

    def extract_game_histories(self):
        """Extract game history data in the format expected by training."""
        game_history_list = []
        for i, history in enumerate(self.game_histories):
            # Only add complete histories that have at least one action
            # And exclude very short games (less than 10 moves) as they were likely incorrectly marked as done
            if len(history.observations) > 0 and len(history.actions) > 0 and len(history.actions) >= 10:
                # Convert to the format that ReplayBuffer.save_game expects:
                # List of (observation, action, reward, policy) tuples
                formatted_history = []
                for j in range(len(history.actions)):
                    if j < len(history.observations) and history.actions[j] is not None:
                        # Create a placeholder uniform policy vector of the right size
                        policy = np.zeros(576, dtype=np.float32)  # Use the action space size
                        # Set a uniform policy (1/num_actions)
                        policy.fill(1.0 / 576)
                        
                        # Match the format expected by ReplayBuffer.save_game
                        formatted_history.append((
                            history.observations[j],  # observation
                            history.actions[j],       # action
                            history.rewards[j] if j < len(history.rewards) else 0,  # reward
                            policy                    # uniform policy 
                        ))
                
                # Only add games with valid moves
                if len(formatted_history) > 0:
                    game_history_list.append(formatted_history)
        
        self.logger.info(f"Extracted {len(game_history_list)} valid game histories with {sum(len(g) for g in game_history_list)} total moves")
        return game_history_list

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate self-play games using MuZero.')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--model', type=str, default=None, help='Path to MuZero model file')
    parser.add_argument('--random', action='store_true', help='Use random policy instead of MuZero')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per game')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--parallel', action='store_true', help='Run games in parallel')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--output', type=str, default='games', help='Output directory for game data')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Default to INFO level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('muzero_games.log')
        ]
    )
    
    logger = logging.getLogger('MuZero-Games')
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Make sure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Load MuZero model if specified
    network = None
    config = {'max_moves': args.max_steps}  # Pass max_steps to config
    
    if args.model and not args.random:
        try:
            network, config = load_muzero_model(args.model, config)
            config['network'] = network
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            if args.debug:
                logger.error(traceback.format_exc())
            
            if not args.random:
                logger.warning("Falling back to random policy")
                args.random = True
    else:
        # Ensure max_steps is in config if we didn't load a model
        config['max_moves'] = args.max_steps
    
    # Determine agent types
    agent1 = "random" if args.random else "muzero"
    agent2 = "random" if args.random else "muzero"  # Use MuZero for both agents when --random is not provided
    
    logger.info(f"Starting {args.games} games with Agent 1: {agent1}, Agent 2: {agent2}")
    
    # Run games
    start_time = time.time()
    results = []
    
    if args.parallel and args.games > 1:
        # Run games in parallel
        logger.info(f"Running games in parallel with {args.workers or 'auto'} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            
            for game_idx in range(args.games):
                seed = game_idx  # Use game index as seed for reproducibility
                future = executor.submit(
                    run_game, 
                    agent1, 
                    agent2, 
                    config,
                    game_idx,
                    seed,
                    args.debug
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in game: {str(e)}")
                    if args.debug:
                        logger.error(traceback.format_exc())
    else:
        # Run games sequentially
        logger.info("Running games sequentially")
        
        for game_idx in range(args.games):
            seed = game_idx  # Use game index as seed for reproducibility
            result = run_game(
                agent1, 
                agent2, 
                config,
                game_idx,
                seed,
                args.debug
            )
            
            if result:
                results.append(result)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Analyze results
    logger.info(f"Completed {len(results)} games out of {args.games} in {elapsed_time:.2f} seconds")
    
    if not results:
        logger.warning("No games were completed successfully")
        sys.exit(1)
    
    # Save game data
    timestamp = int(time.time())
    output_file = os.path.join(args.output, f"muzero_games_{timestamp}.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Game data saved to {output_file}")
    
    # Print summary statistics
    wins = {-1: 0, 0: 0, 1: 0}  # -1: draw, 0: white, 1: black
    total_moves = 0
    total_time = 0
    
    for result in results:
        if 'winner' in result:
            wins[result['winner']] += 1
        if 'num_moves' in result:
            total_moves += result['num_moves']
        if 'elapsed_time' in result:
            total_time += result['elapsed_time']
    
    logger.info(f"Game statistics:")
    logger.info(f"  Total games: {len(results)}")
    logger.info(f"  White wins: {wins[0]} ({wins[0]/len(results)*100:.1f}%)")
    logger.info(f"  Black wins: {wins[1]} ({wins[1]/len(results)*100:.1f}%)")
    logger.info(f"  Draws: {wins[-1]} ({wins[-1]/len(results)*100:.1f}%)")
    logger.info(f"  Average moves per game: {total_moves/len(results):.1f}")
    logger.info(f"  Average time per game: {total_time/len(results):.2f} seconds") 