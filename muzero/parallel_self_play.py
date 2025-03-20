"""
Parallel Self-Play Module for MuZero

This module implements parallel game generation using ProcessPoolExecutor.
It automatically detects available CPU cores and distributes game generation
across multiple processes for efficient training data generation.
"""

import os
import time
import pickle
import logging
import multiprocessing
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from muzero.mcts import MCTS
from muzero.mcts_batched import BatchedMCTS
from muzero.vectorized_env import VectorizedNardeEnv
from muzero.training_optimized import optimized_self_play

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MuZero-Parallel")


def get_optimal_worker_count(requested_workers: Optional[int] = None) -> int:
    """
    Determine the optimal number of worker processes based on system resources.
    
    Args:
        requested_workers: Number of workers explicitly requested, if any
        
    Returns:
        The optimal number of worker processes
    """
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()
    
    # If requested_workers is specified and valid, use it
    if requested_workers is not None and requested_workers > 0:
        return min(requested_workers, cpu_count)
    
    # Otherwise, use all available CPUs but leave one for the main process
    return max(1, cpu_count - 1)


def save_game_history(game_history: List[Tuple], save_dir: str, game_id: int) -> str:
    """
    Save a game history to disk.
    
    Args:
        game_history: List of (observation, action, reward, policy) tuples
        save_dir: Directory to save the game
        game_id: Unique ID for this game
        
    Returns:
        Path to the saved game file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = int(time.time())
    filename = f"game_{game_id}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    # Save the game
    with open(filepath, 'wb') as f:
        pickle.dump(game_history, f)
        
    return filepath


def load_game_history(filepath: str) -> List[Tuple]:
    """
    Load a game history from disk.
    
    Args:
        filepath: Path to the saved game file
        
    Returns:
        Game history as a list of tuples
    """
    with open(filepath, 'rb') as f:
        game_history = pickle.load(f)
        
    return game_history


def play_game_worker(game_id: int, network_weights: Dict, num_simulations: int,
                     temperature: float, save_dir: Optional[str] = None) -> Union[List[Tuple], str]:
    """
    Worker function to play a complete game.
    This function runs in a separate process.
    
    Args:
        game_id: Unique ID for this game
        network_weights: Serialized network weights
        num_simulations: Number of MCTS simulations per move
        temperature: Temperature for action selection
        save_dir: Directory to save the game, or None if not saving
        
    Returns:
        Either the game history (list of tuples) or the filepath if save_dir is provided
    """
    try:
        from gym_narde.envs import NardeEnv
        import logging
        
        # Configure worker-specific logging
        logger = logging.getLogger("MuZero-Worker")
        
        # Create device - each worker should use CPU to avoid GPU contention
        device = "cpu"
        
        # Recreate the network
        from muzero.models import MuZeroNetwork
        input_dim = 28  # Actual observation space size for Narde
        action_dim = 576  # 24x24 possible moves
        
        # Detect hidden_dim from weights - first look for representation_network.fc.6.bias
        hidden_dim = None
        for key in ['representation_network.fc.6.bias', 'representation_network.fc.7.bias']:
            if key in network_weights:
                tensor_shape = network_weights[key].shape
                hidden_dim = tensor_shape[0]
                logger.info(f"Game worker {game_id}: Detected hidden_dim={hidden_dim} from {key} with shape {tensor_shape}")
                break
                
        if hidden_dim is None:
            # Fallback to default used in MuZeroNetwork
            hidden_dim = 256
            logger.info(f"Game worker {game_id}: Using default hidden_dim={hidden_dim}")
        
        logger.info(f"Game worker {game_id}: Creating network with hidden_dim={hidden_dim}")
        network = MuZeroNetwork(input_dim, action_dim, hidden_dim=hidden_dim)
        
        # Load the weights
        logger.info(f"Game worker {game_id}: Loading weights")
        network.load_state_dict(network_weights)
        network = network.to(device)
        
        # Create environment and MCTS
        env = NardeEnv()
        mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            action_space_size=action_dim,
            device=device
        )
        
        # Run a game
        from muzero.training import self_play_game
        game_history = self_play_game(
            env=env,
            network=network,
            mcts=mcts,
            num_simulations=num_simulations,
            temperature=temperature,
            device=device
        )
        
        # Save the game if requested
        if save_dir:
            filepath = save_game_history(game_history, save_dir, game_id)
            return filepath
        else:
            return game_history
            
    except Exception as e:
        logger.error(f"Error in game worker {game_id}: {str(e)}")
        raise


def generate_games_parallel(
    network,
    num_games: int,
    num_simulations: int = 50,
    temperature: float = 1.0,
    save_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    temperature_drop: Optional[int] = None
) -> List[Union[List[Tuple], str]]:
    """
    Generate games in parallel using multiple processes.
    
    Args:
        network: The MuZero network
        num_games: Number of games to generate
        num_simulations: Number of MCTS simulations per move
        temperature: Temperature for action selection
        save_dir: Directory to save games, or None to return in memory
        num_workers: Number of worker processes (default: auto-detect)
        temperature_drop: Move number after which temperature is dropped to 0
        
    Returns:
        List of game histories or filepaths depending on save_dir
    """
    # Determine worker count
    worker_count = get_optimal_worker_count(num_workers)
    logger.info(f"Generating {num_games} games using {worker_count} workers")
    
    # Prepare network weights for passing to workers
    # Must move to CPU first for pickling
    cpu_network = network.to("cpu")
    network_weights = cpu_network.state_dict()
    
    # Check network dimensions
    hidden_dim = None
    if hasattr(cpu_network, 'hidden_dim'):
        hidden_dim = cpu_network.hidden_dim
        logger.info(f"Network has hidden_dim attribute: {hidden_dim}")
    else:
        # Default used in MuZeroNetwork
        hidden_dim = 256
        logger.info(f"No hidden_dim attribute found, using default: {hidden_dim}")
    
    # Check for key tensor shapes
    example_keys = [
        'representation_network.fc.6.bias',
        'representation_network.fc.7.bias',
        'dynamics_network.fc.6.bias',
        'prediction_network.fc.0.weight'
    ]
    
    for key in example_keys:
        if key in network_weights:
            shape = network_weights[key].shape
            logger.info(f"Network weight '{key}' has shape: {shape}")
    
    logger.info(f"Using network with hidden_dim={hidden_dim}")
    
    # Move network back to original device
    if hasattr(network, 'device') and network.device != "cpu":
        network = network.to(network.device)
    
    # Start parallel game generation
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        # Submit all game tasks
        futures = [
            executor.submit(
                play_game_worker,
                game_id=i,
                network_weights=network_weights,
                num_simulations=num_simulations,
                temperature=temperature,
                save_dir=save_dir
            )
            for i in range(num_games)
        ]
        
        # Process results as they complete
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0 or (i + 1) == num_games:
                    elapsed = time.time() - start_time
                    games_per_second = (i + 1) / elapsed
                    logger.info(f"Completed {i+1}/{num_games} games " 
                               f"({games_per_second:.2f} games/s)")
                    
            except Exception as e:
                logger.error(f"Game {i} failed: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"Generated {len(results)} games in {total_time:.2f}s " 
               f"({len(results) / total_time:.2f} games/s)")
    
    return results


def load_games_from_directory(save_dir: str, limit: Optional[int] = None) -> List[List[Tuple]]:
    """
    Load all game histories from a directory.
    
    Args:
        save_dir: Directory containing saved games
        limit: Maximum number of games to load (None for all)
        
    Returns:
        List of game histories
    """
    game_files = list(Path(save_dir).glob("game_*.pkl"))
    
    # Sort by creation time to get most recent games
    game_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Limit if specified
    if limit is not None:
        game_files = game_files[:limit]
    
    # Load games
    logger.info(f"Loading {len(game_files)} games from {save_dir}")
    games = []
    
    for file_path in game_files:
        try:
            game = load_game_history(str(file_path))
            games.append(game)
        except Exception as e:
            logger.error(f"Error loading game {file_path}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(games)} games")
    return games


def get_game_stats(filepath: str) -> Dict[str, Any]:
    """
    Extract basic statistics from a saved game.
    
    Args:
        filepath: Path to saved game
        
    Returns:
        Dictionary with game statistics
    """
    game_history = load_game_history(filepath)
    
    # Basic stats
    stats = {
        "moves": len(game_history),
        "final_reward": game_history[-1][2] if game_history else 0,
    }
    
    return stats


def self_play_demo(num_games: int = 1, num_simulations: int = 50, save_dir: Optional[str] = None):
    """
    Generate games using self-play and print statistics.
    For demonstration purposes.
    """
    # Create environment and network
    network = MuZeroNetwork(input_dim=28, action_dim=576)
    
    # Generate games
    game_paths = generate_games_parallel(
        network=network,
        num_games=num_games,
        num_simulations=num_simulations,
        temperature=1.0,
        save_dir=save_dir
    )


if __name__ == "__main__":
    # Example usage
    from muzero.models import MuZeroNetwork
    
    # Create a network
    network = MuZeroNetwork(input_dim=24, action_dim=576)
    
    # Generate games
    save_dir = "games"
    num_games = 4
    num_workers = 2
    
    game_paths = generate_games_parallel(
        network=network,
        num_games=num_games,
        num_simulations=20,
        save_dir=save_dir,
        num_workers=num_workers
    )
    
    print(f"Generated and saved {len(game_paths)} games")
    
    # Load games
    games = load_games_from_directory(save_dir)
    print(f"Loaded {len(games)} games") 