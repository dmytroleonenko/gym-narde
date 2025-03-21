#!/usr/bin/env python3
"""
Parallel Training Pipeline for MuZero

This module implements a full training pipeline for MuZero that combines:
1. Parallel game generation using multiple CPU cores
2. Efficient batched learning using GPU acceleration
3. Automated pipeline management for continuous training

The pipeline alternates between generating games and training the model in cycles.
"""

import os
import time
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor

# Import local modules
from models import MuZeroNetwork
from replay import ReplayBuffer
from parallel_self_play import load_games_from_directory, save_game_history
from training_optimized import optimized_self_play
from worker_functions import optimized_self_play_worker
from config import MuZeroConfig
from mcts import MCTS, Node
from mcts_batched import BatchedMCTS, BatchedNode
from game_simulator import GameSimulator
from tqdm import tqdm

# Configure logging
log_level_name = os.environ.get('LOGLEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific logger levels
worker_logger = logging.getLogger("MuZero-Worker")
worker_logger.setLevel(max(log_level, logging.INFO))  # Ensure worker logs are at least INFO


class TrainingPipeline:
    """
    End-to-end training pipeline for MuZero that combines parallel game
    generation with efficient batched learning.
    """
    def __init__(
        self,
        base_dir: str = "muzero_training",
        input_dim: int = 28,
        action_dim: int = 576,
        hidden_dim: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        replay_buffer_size: int = 100000,
        games_per_iteration: int = 2000,
        num_simulations: int = 50,
        num_epochs: int = 10,
        temperature: float = 1.0,
        temperature_drop: Optional[int] = None,
        device: str = "auto",
        num_workers: Optional[int] = None,
        mcts_batch_size: int = 8,
        env_batch_size: int = 16,
        use_optimized_self_play: bool = False,
        use_batched_game_generation: bool = False,
        training_iterations: int = 10,
        save_checkpoint_every: int = 1,
        debug: bool = False
    ):
        """
        Initialize the training pipeline.
        
        Args:
            base_dir: Directory for saving checkpoints and games
            input_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            lr: Learning rate
            weight_decay: Weight decay coefficient
            batch_size: Batch size for training
            replay_buffer_size: Maximum replay buffer size
            games_per_iteration: Number of games to generate per iteration
            num_simulations: Number of MCTS simulations per move
            num_epochs: Number of training epochs per iteration
            temperature: Temperature for action selection
            temperature_drop: Move number to drop temperature to 0
            device: Device to use (cpu, cuda, mps, or auto)
            num_workers: Number of worker processes
            mcts_batch_size: Batch size for batched MCTS when using optimized self-play
            env_batch_size: Batch size for vectorized environment when using optimized self-play
            use_optimized_self_play: Whether to use the optimized self-play implementation
            use_batched_game_generation: Whether to use batched game generation
            training_iterations: Number of training iterations to run
            save_checkpoint_every: Save checkpoint every N iterations
            debug: Whether to enable debug logging
        """
        # Create a namespace to store all arguments for easy access
        self.args = argparse.Namespace()
        self.args.base_dir = base_dir
        self.args.input_dim = input_dim
        self.args.action_dim = action_dim
        self.args.hidden_dim = hidden_dim
        self.args.lr = lr
        self.args.weight_decay = weight_decay
        self.args.batch_size = batch_size
        self.args.replay_buffer_size = replay_buffer_size
        self.args.games_per_iteration = games_per_iteration
        self.args.num_simulations = num_simulations
        self.args.num_epochs = num_epochs
        self.args.temperature = temperature
        self.args.temperature_drop = temperature_drop
        self.args.device = device
        self.args.num_workers = get_optimal_worker_count(num_workers)
        self.args.mcts_batch_size = mcts_batch_size
        self.args.env_batch_size = env_batch_size
        self.args.use_optimized_self_play = use_optimized_self_play
        self.args.use_batched_game_generation = use_batched_game_generation
        self.args.training_iterations = training_iterations
        self.args.save_checkpoint_every = save_checkpoint_every
        self.args.debug = debug
        
        # Also set individual attributes for backward compatibility
        self.base_dir = base_dir
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.games_per_iteration = games_per_iteration
        self.num_simulations = num_simulations
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.temperature_drop = temperature_drop
        self.device = device
        self.num_workers = get_optimal_worker_count(num_workers)
        self.mcts_batch_size = mcts_batch_size
        self.env_batch_size = env_batch_size
        self.use_optimized_self_play = use_optimized_self_play
        self.training_iterations = training_iterations
        self.save_checkpoint_every = save_checkpoint_every
        self.debug = debug
        
        # Set up directories
        self.games_dir = os.path.join(base_dir, "games")
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for training")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon) for training")
            else:
                self.device = "cpu"
                logger.info("Using CPU for training (no GPU available)")
        else:
            self.device = device
        
        # Create network
        self.network = MuZeroNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        self.network = self.network.to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Set up metrics tracking
        self.metrics = {
            "iterations": [],
            "game_generation_times": [],
            "training_times": [],
            "value_losses": [],
            "policy_losses": [],
            "reward_losses": []
        }
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        logger.info(f"Initialized training pipeline with {self.num_workers} workers")
        logger.info(f"Will generate {games_per_iteration} games per iteration")
        logger.info(f"Training with batch size {batch_size} for {num_epochs} epochs per iteration")

    def _get_game_iteration_dir(self, iteration: int) -> str:
        """Get directory for games from a specific iteration."""
        return os.path.join(self.games_dir, f"iteration_{iteration}")
    
    def _get_checkpoint_path(self, iteration: int) -> str:
        """Get path for checkpoint at a specific iteration."""
        return os.path.join(self.models_dir, f"muzero_iteration_{iteration}.pt")
    
    def save_checkpoint(self, iteration: int) -> str:
        """Save the current network and optimizer state."""
        checkpoint_path = self._get_checkpoint_path(iteration)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'metrics': self.metrics,
            'hidden_dim': self.hidden_dim,
            'input_dim': self.input_dim,
            'action_dim': self.action_dim
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, iteration: Optional[int] = None) -> int:
        """
        Load the latest checkpoint or a specific iteration.
        
        Returns:
            The iteration number of the loaded checkpoint
        """
        if iteration is not None:
            checkpoint_path = self._get_checkpoint_path(iteration)
        else:
            # Find the latest checkpoint
            checkpoint_files = sorted(list(Path(self.models_dir).glob("muzero_iteration_*.pt")))
            if not checkpoint_files:
                logger.info("No checkpoints found, starting from scratch")
                return 0
            
            checkpoint_path = checkpoint_files[-1]
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint to CPU first to examine state dict
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint['network_state_dict']
        
        # Check for explicitly saved dimensions first (preferred method)
        if 'hidden_dim' in checkpoint:
            checkpoint_hidden_dim = checkpoint['hidden_dim']
            checkpoint_input_dim = checkpoint.get('input_dim', self.input_dim)
            checkpoint_action_dim = checkpoint.get('action_dim', self.action_dim)
            
            # If dimensions don't match, recreate the network with correct dimensions
            if (checkpoint_hidden_dim != self.hidden_dim or 
                checkpoint_input_dim != self.input_dim or 
                checkpoint_action_dim != self.action_dim):
                logger.info(f"Checkpoint has different dimensions. Using: " +
                           f"hidden_dim={checkpoint_hidden_dim}, " +
                           f"input_dim={checkpoint_input_dim}, " +
                           f"action_dim={checkpoint_action_dim}")
                
                self.hidden_dim = checkpoint_hidden_dim
                self.input_dim = checkpoint_input_dim
                self.action_dim = checkpoint_action_dim
                
                self.network = MuZeroNetwork(
                    input_dim=self.input_dim,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim
                )
        # Fall back to old method of detecting dimensions from weights
        elif 'representation_network.fc.6.bias' in state_dict:
            checkpoint_hidden_dim = state_dict['representation_network.fc.6.bias'].shape[0]
            
            # If dimensions don't match, recreate the network with correct dimensions
            if checkpoint_hidden_dim != self.hidden_dim:
                logger.info(f"Detected different hidden_dim in checkpoint: {checkpoint_hidden_dim}, " +
                           f"current model: {self.hidden_dim}. Recreating network...")
                self.hidden_dim = checkpoint_hidden_dim
                self.network = MuZeroNetwork(
                    input_dim=self.input_dim,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim
                )
        
        # Now load the state dict to the device
        self.network = self.network.to(self.device)
        self.network.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move optimizer params to device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        loaded_iteration = checkpoint['iteration']
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        logger.info(f"Loaded checkpoint from iteration {loaded_iteration} with hidden_dim={self.hidden_dim}")
        return loaded_iteration
    
    def generate_games(self, iteration: int, games_per_iteration: int) -> float:
        """
        Generate games in parallel and store them to disk.
        
        Args:
            iteration: Current training iteration
            games_per_iteration: Number of games to generate per iteration
            
        Returns:
            Time taken to generate games
        """
        iteration_games_dir = self._get_game_iteration_dir(iteration)
        os.makedirs(iteration_games_dir, exist_ok=True)
        
        logger.info(f"Generating {games_per_iteration} games for iteration {iteration}")
        logger.info(f"Network dimensions: input_dim={self.input_dim}, action_dim={self.action_dim}, hidden_dim={self.hidden_dim}")
        
        # Performance tips
        logger.info("Performance tips:")
        logger.info(" - Reduce num_simulations to speed up game generation")
        logger.info(" - Increase num_workers to utilize more CPU cores")
        logger.info(" - Set mcts_batch_size higher for better GPU utilization")
        logger.info(" - For GPU acceleration, use optimized_self_play with CUDA instead of parallel_self_play")
        
        start_time = time.time()
        
        # If this is just the first iteration with no trained model yet,
        # we can use a smaller number of simulations to speed up initial game generation
        if iteration == 1 and not hasattr(self, '_checkpoint_loaded'):
            original_simulations = self.num_simulations
            reduced_simulations = min(20, original_simulations)  # Use at most 20 simulations for first iteration
            logger.info(f"First iteration: Reducing simulations from {original_simulations} to {reduced_simulations} for faster bootstrap")
            self.num_simulations = reduced_simulations
        
        # Move network to appropriate device
        device = "cpu"  # Default for parallel game generation
        
        # Check if optimized_self_play should be used (if we have GPU and the option is enabled)
        use_optimized = False
        if self.use_optimized_self_play:
            if torch.cuda.is_available():
                device = "cuda"
                use_optimized = True
                logger.info("Using optimized_self_play with CUDA acceleration")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Check if MPS is truly working by creating a small tensor
                try:
                    # Create a small tensor on MPS to verify it works
                    test_tensor = torch.ones(1, device="mps")
                    # Try a simple operation to confirm MPS backend is operational
                    _ = test_tensor + test_tensor  
                    torch.mps.synchronize()
                    device = "mps"
                    use_optimized = True
                    logger.info("Using optimized_self_play with MPS (Apple Silicon) acceleration - verified working")
                except Exception as e:
                    logger.warning(f"MPS appears available but failed initialization test: {str(e)}")
                    logger.warning("Falling back to CPU implementation")
                    device = "cpu"
            else:
                logger.warning("use_optimized_self_play is enabled but no GPU device available, falling back to parallel CPU version")
        
        # Move the network to the appropriate device
        network_for_games = self.network.to(device)
        
        if use_optimized:
            # Use optimized self-play with GPU acceleration (CUDA or MPS)
            gpu_type = "CUDA" if torch.cuda.is_available() else "MPS"
            logger.info(f"Using optimized_self_play with {gpu_type} acceleration")
            
            # Calculate batch configuration
            games_per_worker = games_per_iteration // self.num_workers
            remainder = games_per_iteration % self.num_workers
            
            # Use ProcessPoolExecutor to parallelize the generation tasks
            network_for_games = self.network.to('cpu')  # Move to CPU for pickling
            network_weights = network_for_games.state_dict()
            
            logger.info(f"Using optimized self-play with {self.num_workers} workers and {gpu_type} acceleration")
            logger.info(f"Each worker will generate ~{games_per_worker} games")
            
            import concurrent.futures
            import multiprocessing
            from functools import partial
            
            # Distribute games across workers
            start_time = time.time()
            game_paths = []
            
            # Make sure we're using spawn method for starting processes 
            # This is required for CUDA and recommended for MPS
            ctx = multiprocessing.get_context('spawn')
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers, 
                mp_context=ctx
            ) as executor:
                # Create tasks for each worker
                futures = []
                for worker_id in range(self.num_workers):
                    # Distribute games evenly, with remainder going to early workers
                    worker_games = games_per_worker + (1 if worker_id < remainder else 0)
                    if worker_games > 0:
                        future = executor.submit(
                            optimized_self_play_worker,
                            worker_id,
                            network_weights,
                            worker_games,
                            iteration_games_dir,
                            self.num_simulations,
                            self.temperature,
                            self.temperature_drop,
                            self.mcts_batch_size,
                            self.env_batch_size,
                            self.hidden_dim,
                            self.device  # Pass the device configured in the pipeline
                        )
                        futures.append(future)
                        logger.info(f"Submitted worker {worker_id} to generate {worker_games} games")
                
                # Process results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        worker_paths = future.result()
                        game_paths.extend(worker_paths)
                        completed += len(worker_paths)
                        
                        # Log progress
                        elapsed = time.time() - start_time
                        games_per_second = completed / elapsed if elapsed > 0 else 0
                        logger.info(f"Completed {completed}/{games_per_iteration} games " 
                                   f"({games_per_second:.2f} games/s)")
                    except Exception as e:
                        logger.error(f"Error processing worker result: {str(e)}")
                               
            # Move network back to training device
            self.network = self.network.to(self.device)
        else:
            # Generate games using standard parallel implementation
            logger.info("Using standard parallel implementation for game generation")
            game_paths = generate_games_parallel(
                network=network_for_games,
                num_games=games_per_iteration,
                num_simulations=self.num_simulations,
                temperature=self.temperature,
                save_dir=iteration_games_dir,
                num_workers=self.num_workers,
                temperature_drop=self.temperature_drop
            )
        
        # Restore original num_simulations if it was reduced
        if iteration == 1 and not hasattr(self, '_checkpoint_loaded'):
            self.num_simulations = original_simulations
            logger.info(f"Restored num_simulations to {original_simulations}")
        
        # Move network back to training device
        self.network = self.network.to(self.device)
        
        generation_time = time.time() - start_time
        games_per_second = games_per_iteration / generation_time
        
        logger.info(f"Generated {len(game_paths)} games in {generation_time:.2f}s "
                    f"({games_per_second:.2f} games/s)")
        
        # Update metrics
        self.metrics["game_generation_times"].append(generation_time)
        
        return generation_time
    
    def train_on_generated_games(self, iteration: int) -> Dict[str, float]:
        """
        Train the network on generated games.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Training metrics
        """
        iteration_games_dir = self._get_game_iteration_dir(iteration)
        
        # Load games
        logger.info(f"Loading games from {iteration_games_dir}")
        games = load_games_from_directory(iteration_games_dir)
        logger.info(f"Loaded {len(games)} games")
        
        # Add games to replay buffer
        for game in games:
            self.replay_buffer.save_game(game)
        
        logger.info(f"Replay buffer now contains {len(self.replay_buffer.buffer)} games")
        
        # Train the network
        logger.info(f"Training for {self.num_epochs} epochs with batch size {self.batch_size}")
        start_time = time.time()
        
        value_losses = []
        policy_losses = []
        reward_losses = []
        
        for epoch in range(self.num_epochs):
            # Use optimized training epoch
            metrics = optimized_training_epoch(
                network=self.network,
                optimizer=self.optimizer,
                replay_buffer=self.replay_buffer,
                batch_size=self.batch_size,
                device=self.device
            )
            
            value_losses.append(metrics["value_loss"])
            policy_losses.append(metrics["policy_loss"])
            reward_losses.append(metrics["reward_loss"])
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.num_epochs:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                           f"Value loss: {metrics['value_loss']:.4f}, "
                           f"Policy loss: {metrics['policy_loss']:.4f}, "
                           f"Reward loss: {metrics['reward_loss']:.4f}")
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Update metrics
        self.metrics["iterations"].append(iteration)
        self.metrics["training_times"].append(training_time)
        self.metrics["value_losses"].append(np.mean(value_losses))
        self.metrics["policy_losses"].append(np.mean(policy_losses))
        self.metrics["reward_losses"].append(np.mean(reward_losses))
        
        return {
            "training_time": training_time,
            "value_loss": np.mean(value_losses),
            "policy_loss": np.mean(policy_losses),
            "reward_loss": np.mean(reward_losses)
        }
    
    def run_training_pipeline(self, start_iteration: int = 0) -> None:
        """Run the full training pipeline for the specified number of iterations."""
        logger.info(f"Starting training pipeline from iteration {start_iteration}")
        logger.info(f"Will run for {self.training_iterations} iterations")
        
        for iteration in range(start_iteration, start_iteration + self.training_iterations):
            logger.info(f"=== Iteration {iteration} ===")
            
            # Game generation step update (skip if requested)
            if not self.args.skip_game_generation:
                if self.args.use_batched_game_generation:
                    try:
                        logger.info("Using batched game generation with GPU accelerated MCTS.")
                        
                        # Create a model for batched MCTS
                        logger.info("Creating initial model for batched game generation")
                        model_path = os.path.join(self.models_dir, "muzero_model_temp.pth")
                        torch.save(self.network.state_dict(), model_path)
                        logger.info(f"Using model from {model_path} for batched game generation")

                        # Initialize BatchedGameSimulator
                        from final_complete_game_generation import BatchedGameSimulator
                        simulator = BatchedGameSimulator(
                            num_games=self.args.games_per_iteration,
                            model_path=model_path,
                            batch_size=self.args.mcts_batch_size,
                            max_games=self.args.games_per_iteration,
                            device=self.device,
                            debug=self.args.debug
                        )
                        
                        # Run simulation
                        start_time = time.time()
                        simulator.run()
                        
                        # Get game histories and add to replay buffer
                        game_histories = simulator.extract_game_histories()
                        self.add_games_to_replay_buffer(game_histories)
                        generation_time = time.time() - start_time
                        logger.info(f"Generated {len(game_histories)} games in {generation_time:.2f}s with batched MCTS")
                        
                    except Exception as e:
                        logger.error(f"Error in batched game generation: {str(e)}")
                        logger.error(traceback.format_exc())
                        logger.warning("Falling back to standard game generation...")
                        self.network.to("cpu")
                        start_time = time.time()
                        self.generate_games(iteration, self.args.games_per_iteration)
                        generation_time = time.time() - start_time
                else:
                    start_time = time.time()
                    self.generate_games(iteration, self.args.games_per_iteration)
                    generation_time = time.time() - start_time
            else:
                logger.info("Skipping game generation as requested")
                generation_time = 0
                
                # If existing_games_dir is provided, load games from there
                if self.args.existing_games_dir and os.path.exists(self.args.existing_games_dir):
                    logger.info(f"Loading existing games from {self.args.existing_games_dir}")
                    try:
                        game_files = []
                        # Collect all .pkl files in the directory
                        for file in os.listdir(self.args.existing_games_dir):
                            if file.endswith('.pkl'):
                                game_files.append(os.path.join(self.args.existing_games_dir, file))
                                
                        if not game_files:
                            logger.warning(f"No .pkl files found in {self.args.existing_games_dir}")
                        else:
                            logger.info(f"Found {len(game_files)} game files")
                            for game_file in game_files:
                                games = load_games_from_directory(game_file)
                                logger.info(f"Loaded {len(games)} games from {game_file}")
                                self.add_games_to_replay_buffer(games)
                    except Exception as e:
                        logger.error(f"Error loading existing games: {str(e)}")
                        logger.error(traceback.format_exc())
            
            # Train on generated games
            training_metrics = self.train_on_generated_games(iteration)
            
            # Log combined metrics
            total_time = generation_time + training_metrics["training_time"]
            logger.info(f"Iteration {iteration} completed in {total_time:.2f}s")
            if not self.args.skip_game_generation:
                logger.info(f"  Game generation: {generation_time:.2f}s")
            logger.info(f"  Training: {training_metrics['training_time']:.2f}s")
            logger.info(f"  Value loss: {training_metrics['value_loss']:.4f}")
            logger.info(f"  Policy loss: {training_metrics['policy_loss']:.4f}")
            logger.info(f"  Reward loss: {training_metrics['reward_loss']:.4f}")
            
            # Save checkpoint if needed
            if (iteration + 1) % self.save_checkpoint_every == 0 or (iteration + 1) == self.training_iterations:
                self.save_checkpoint(iteration)
        
        logger.info("Training pipeline completed")

    def add_games_to_replay_buffer(self, game_histories):
        """Add game histories to the replay buffer."""
        logger.info(f"Adding {len(game_histories)} games to replay buffer")
        for game_history in game_histories:
            try:
                # Handle different formats of game history
                if isinstance(game_history, list) and len(game_history) > 0 and isinstance(game_history[0], tuple):
                    # If it's already in the format expected by ReplayBuffer.save_game() (list of tuples)
                    self.replay_buffer.save_game(game_history)
                elif isinstance(game_history, dict) and 'observations' in game_history:
                    # If it's a dictionary with observations, actions, rewards
                    observations = game_history.get('observations', [])
                    actions = game_history.get('actions', [])
                    rewards = game_history.get('rewards', [])
                    
                    # Create a placeholder uniform policy
                    uniform_policy = np.zeros(self.action_dim, dtype=np.float32)
                    uniform_policy.fill(1.0 / self.action_dim)
                    
                    # Create the game history tuples format
                    game_tuples = []
                    for i in range(min(len(observations), len(actions))):
                        reward = rewards[i] if i < len(rewards) else 0
                        game_tuples.append((observations[i], actions[i], reward, uniform_policy))
                    
                    # Save the game
                    if game_tuples:
                        self.replay_buffer.save_game(game_tuples)
                else:
                    logger.warning(f"Unsupported game history format: {type(game_history)}")
            except Exception as e:
                logger.error(f"Error adding game to replay buffer: {str(e)}")
                logger.error(traceback.format_exc())


def parse_arguments():
    """Parse command line arguments for the MuZero training pipeline."""
    parser = argparse.ArgumentParser(description="MuZero training pipeline")
    parser.add_argument("--base_dir", type=str, default="muzero_training",
                        help="Directory for saving games and models")
    parser.add_argument("--input_dim", type=int, default=64,
                        help="Dimension of input observation")
    parser.add_argument("--action_dim", type=int, default=4162,
                        help="Dimension of action space")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension of MuZero network")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of residual blocks in MuZero network")
    parser.add_argument("--num_channels", type=int, default=256,
                        help="Number of channels in MuZero network")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--num_simulations", type=int, default=80,
                        help="Number of MCTS simulations")
    parser.add_argument("--mcts_batch_size", type=int, default=32,
                        help="Batch size for MCTS inference")
    parser.add_argument("--discount", type=float, default=0.997,
                        help="Discount factor for rewards")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3,
                        help="Alpha parameter for Dirichlet noise")
    parser.add_argument("--exploration_fraction", type=float, default=0.25,
                        help="Fraction of noise to add to root node")
    parser.add_argument("--games_per_iteration", type=int, default=2000,
                        help="Number of games to generate per iteration")
    parser.add_argument("--training_iterations", type=int, default=10,
                        help="Number of training iterations to run")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs per iteration")
    parser.add_argument("--replay_buffer_size", type=int, default=100000,
                        help="Maximum size of replay buffer")
    parser.add_argument("--num_unroll_steps", type=int, default=5,
                        help="Number of unroll steps during training")
    parser.add_argument("--td_steps", type=int, default=10,
                        help="Number of steps for bootstrapping targets")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--save_checkpoint_every", type=int, default=1,
                        help="How often to save checkpoints (in iterations)")
    parser.add_argument("--load_iteration", type=int, default=None,
                        help="Iteration to load model from (None for no loading)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_batched_game_generation", action="store_true",
                        help="Use batched game generation with GPU-accelerated MCTS")
    parser.add_argument("--use_value_prefix", action="store_true",
                        help="Use value prefix in MuZero network")
    parser.add_argument("--use_symmetry", action="store_true",
                        help="Use board symmetry for data augmentation")
    parser.add_argument("--skip_game_generation", action="store_true",
                        help="Skip game generation and only perform training with existing games")
    parser.add_argument("--existing_games_dir", type=str, default=None,
                        help="Directory containing existing games to use for training (only with --skip_game_generation)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize the pipeline
    pipeline = TrainingPipeline(
        base_dir=args.base_dir,
        input_dim=args.input_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        device=args.device,
        temperature=args.temperature,
        temperature_drop=args.temperature_drop,
        mcts_batch_size=args.mcts_batch_size,
        env_batch_size=args.env_batch_size,
        use_optimized_self_play=args.use_optimized_self_play,
        use_batched_game_generation=args.use_batched_game_generation,
        training_iterations=args.training_iterations,
        save_checkpoint_every=args.save_checkpoint_every,
        debug=args.debug
    )
    
    # Load checkpoint if specified
    start_iteration = 0
    if args.load_iteration is not None:
        start_iteration = pipeline.load_checkpoint(args.load_iteration) + 1
    elif os.path.exists(args.base_dir) and os.path.isdir(args.base_dir):
        # Try to load the latest checkpoint
        try:
            start_iteration = pipeline.load_checkpoint() + 1
        except Exception as e:
            logger.warning(f"Failed to load latest checkpoint: {str(e)}")
            logger.warning("Starting from scratch")
    
    # Run the pipeline
    pipeline.run_training_pipeline(start_iteration=start_iteration) 