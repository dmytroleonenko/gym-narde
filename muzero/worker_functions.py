#!/usr/bin/env python3
"""
Worker functions for parallel MuZero training

This module contains the worker functions that are used for parallel processing
in the MuZero training pipeline. These are defined at the module level to ensure
they can be pickled for multiprocessing.
"""

import os
import torch
import logging

# Configure logging
logger = logging.getLogger("MuZero-Pipeline")

def optimized_self_play_worker(worker_id, network_weights, num_games, save_dir, 
                               num_simulations, temperature, temperature_drop,
                               mcts_batch_size, env_batch_size, hidden_dim, target_device):
    """
    Worker function for optimized self-play in parallel.
    
    This function is defined at the module level (not inside a class) to ensure
    it can be pickled for multiprocessing.
    
    Args:
        worker_id: ID of this worker
        network_weights: State dict of the network to use
        num_games: Number of games to generate
        save_dir: Directory to save games to
        num_simulations: Number of MCTS simulations per move
        temperature: Temperature for action selection
        temperature_drop: Move number to drop temperature to 0
        mcts_batch_size: Batch size for MCTS simulations
        env_batch_size: Batch size for vectorized environments  
        hidden_dim: Hidden dimension for the network
        target_device: Device to use (cuda, mps, cpu, or auto)
    
    Returns:
        List of paths to saved game files
    """
    worker_process_id = os.getpid()
    logger.info(f"Worker {worker_id} started in process {worker_process_id}")
    
    try:
        # Import necessary modules inside worker to avoid multiprocessing issues
        from muzero.models import MuZeroNetwork
        from muzero.training_optimized import optimized_self_play
        from muzero.parallel_self_play import save_game_history
        import torch
        
        # Determine device to use in this worker
        device = target_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Worker {worker_id} detected CUDA device")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info(f"Worker {worker_id} detected MPS (Apple Silicon)")
            else:
                device = "cpu"
                logger.info(f"Worker {worker_id} defaulting to CPU (no GPU detected)")
        
        # When using MPS, make sure it's initialized properly
        if device == "mps":
            # Force MPS initialization explicitly before any tensor operations
            logger.info(f"Worker {worker_id} explicitly initializing MPS device")
            _ = torch.zeros(1, device="mps")
            torch.mps.synchronize()  # Wait for initialization to complete
        
        logger.info(f"Worker {worker_id} using device: {device}")
        
        # Create a new network instance in this process
        network = MuZeroNetwork(
            input_dim=28, 
            action_dim=576, 
            hidden_dim=hidden_dim
        )
        # Load weights first on CPU (avoids MPS initialization issues)
        network.load_state_dict(network_weights)
        # Then move to target device
        network = network.to(device)
        logger.info(f"Worker {worker_id} created network and moved to {device}")
        
        # Worker specific directory
        worker_dir = os.path.join(save_dir, f"worker_{worker_id}")
        os.makedirs(worker_dir, exist_ok=True)
        
        logger.info(f"Worker {worker_id} running optimized_self_play for {num_games} games with device={device}")
        
        # Run optimized self-play
        batch_game_histories = optimized_self_play(
            network=network,
            num_games=num_games,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
            env_batch_size=min(num_games, env_batch_size),
            temperature=temperature,
            temperature_drop=temperature_drop,
            device=device
        )
        
        # Save games and return paths
        game_paths = []
        for i, game_history in enumerate(batch_game_histories):
            filepath = save_game_history(game_history, worker_dir, i)
            game_paths.append(filepath)
        
        logger.info(f"Worker {worker_id} completed {len(game_paths)} games")
        return game_paths
    except Exception as e:
        logger.error(f"Worker {worker_id} encountered error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [] 