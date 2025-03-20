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

# Import local modules
from muzero.models import MuZeroNetwork
from muzero.replay import ReplayBuffer
from muzero.training_optimized import optimized_training_epoch, train_muzero
from muzero.parallel_self_play import (
    generate_games_parallel,
    load_games_from_directory,
    get_optimal_worker_count
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MuZero-Pipeline")


class TrainingPipeline:
    """
    End-to-end training pipeline for MuZero that combines parallel game
    generation with efficient batched learning.
    """
    def __init__(
        self,
        base_dir: str = "muzero_training",
        input_dim: int = 24,
        action_dim: int = 576,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        games_per_iteration: int = 2000,
        num_simulations: int = 50,
        num_epochs: int = 10,
        num_workers: Optional[int] = None,
        device: Optional[str] = None,
        temperature: float = 1.0,
        temperature_drop: Optional[int] = None,
        mcts_batch_size: int = 16,
        training_iterations: int = 10,
        save_checkpoint_every: int = 1
    ):
        """
        Initialize the training pipeline.
        
        Args:
            base_dir: Base directory for saving games and models
            input_dim: Input dimension for the MuZero network
            action_dim: Action dimension for the MuZero network
            hidden_dim: Hidden dimension for the MuZero network
            latent_dim: Latent dimension for the MuZero network
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay
            games_per_iteration: Number of games to generate per iteration
            num_simulations: Number of MCTS simulations per move
            num_epochs: Number of training epochs per iteration
            num_workers: Number of worker processes (default: auto-detect)
            device: Device to use (default: auto-detect)
            temperature: Temperature for action selection
            temperature_drop: Move number after which temperature is dropped to 0
            mcts_batch_size: Batch size for MCTS simulations
            training_iterations: Number of training iterations
            save_checkpoint_every: Save checkpoint every N iterations
        """
        self.base_dir = base_dir
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.games_per_iteration = games_per_iteration
        self.num_simulations = num_simulations
        self.num_epochs = num_epochs
        self.num_workers = get_optimal_worker_count(num_workers)
        self.temperature = temperature
        self.temperature_drop = temperature_drop
        self.mcts_batch_size = mcts_batch_size
        self.training_iterations = training_iterations
        self.save_checkpoint_every = save_checkpoint_every
        
        # Set up directories
        self.games_dir = os.path.join(base_dir, "games")
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Set device
        if device is None:
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
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        
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
            'metrics': self.metrics
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
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_iteration = checkpoint['iteration']
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        logger.info(f"Loaded checkpoint from iteration {loaded_iteration}")
        return loaded_iteration
    
    def generate_games(self, iteration: int) -> float:
        """
        Generate games in parallel and store them to disk.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Time taken to generate games
        """
        iteration_games_dir = self._get_game_iteration_dir(iteration)
        os.makedirs(iteration_games_dir, exist_ok=True)
        
        logger.info(f"Generating {self.games_per_iteration} games for iteration {iteration}")
        start_time = time.time()
        
        # Move network to CPU and get state dict
        cpu_network = self.network.to("cpu")
        
        # Generate games
        game_paths = generate_games_parallel(
            network=cpu_network,
            num_games=self.games_per_iteration,
            num_simulations=self.num_simulations,
            temperature=self.temperature,
            save_dir=iteration_games_dir,
            num_workers=self.num_workers,
            temperature_drop=self.temperature_drop
        )
        
        # Move network back to device
        self.network = self.network.to(self.device)
        
        generation_time = time.time() - start_time
        games_per_second = self.games_per_iteration / generation_time
        
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
            
            # Generate games
            generation_time = self.generate_games(iteration)
            
            # Train on generated games
            training_metrics = self.train_on_generated_games(iteration)
            
            # Log combined metrics
            total_time = generation_time + training_metrics["training_time"]
            logger.info(f"Iteration {iteration} completed in {total_time:.2f}s")
            logger.info(f"  Game generation: {generation_time:.2f}s")
            logger.info(f"  Training: {training_metrics['training_time']:.2f}s")
            logger.info(f"  Value loss: {training_metrics['value_loss']:.4f}")
            logger.info(f"  Policy loss: {training_metrics['policy_loss']:.4f}")
            logger.info(f"  Reward loss: {training_metrics['reward_loss']:.4f}")
            
            # Save checkpoint if needed
            if (iteration + 1) % self.save_checkpoint_every == 0 or (iteration + 1) == self.training_iterations:
                self.save_checkpoint(iteration)
        
        logger.info("Training pipeline completed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MuZero Parallel Training Pipeline")
    
    # Training parameters
    parser.add_argument("--base_dir", type=str, default="muzero_training", 
                        help="Base directory for saving games and models")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--games_per_iteration", type=int, default=2000, 
                        help="Number of games to generate per iteration")
    parser.add_argument("--num_simulations", type=int, default=50, 
                        help="Number of MCTS simulations per move")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs per iteration")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of worker processes (default: auto-detect)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Temperature for action selection")
    parser.add_argument("--temperature_drop", type=int, default=None, 
                        help="Move number after which temperature is dropped to 0")
    parser.add_argument("--mcts_batch_size", type=int, default=16, 
                        help="Batch size for MCTS simulations")
    parser.add_argument("--training_iterations", type=int, default=10, 
                        help="Number of training iterations")
    parser.add_argument("--save_checkpoint_every", type=int, default=1, 
                        help="Save checkpoint every N iterations")
    parser.add_argument("--load_iteration", type=int, default=None, 
                        help="Load checkpoint from specific iteration (default: latest)")
    
    # Network parameters
    parser.add_argument("--hidden_dim", type=int, default=128, 
                        help="Hidden dimension for the MuZero network")
    # Note: latent_dim is kept for backwards compatibility but not used in current MuZeroNetwork implementation
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize the pipeline
    pipeline = TrainingPipeline(
        base_dir=args.base_dir,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
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
        training_iterations=args.training_iterations,
        save_checkpoint_every=args.save_checkpoint_every
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