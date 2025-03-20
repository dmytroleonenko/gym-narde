#!/usr/bin/env python3
"""
MuZero Model Evaluation Script

This script evaluates a trained MuZero model by:
1. Playing games against a baseline agent or self-play
2. Computing performance metrics (win rate, game length, etc.)
3. Comparing metrics across different model iterations
4. Generating performance visualizations

Usage:
    python -m muzero.evaluate_model --base_dir muzero_training --iteration latest --num_games 100
"""

import os
import json
import time
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Import local modules
from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS
from gym_narde.envs import NardeEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MuZero-Evaluate")


class ModelEvaluator:
    """
    Evaluates MuZero models and tracks performance across iterations.
    """
    def __init__(
        self,
        base_dir: str = "muzero_training",
        input_dim: int = 24,
        action_dim: int = 576,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        device: Optional[str] = None
    ):
        """
        Initialize the model evaluator.
        
        Args:
            base_dir: Base directory for model checkpoints and evaluation results
            input_dim: Input dimension for the MuZero network
            action_dim: Action dimension for the MuZero network 
            hidden_dim: Hidden dimension for the MuZero network
            latent_dim: Latent dimension for the MuZero network
            device: Device to use for evaluation (default: auto-detect)
        """
        self.base_dir = base_dir
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Set up directories
        self.models_dir = os.path.join(base_dir, "models")
        self.eval_dir = os.path.join(base_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for evaluation")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon) for evaluation")
            else:
                self.device = "cpu"
                logger.info("Using CPU for evaluation (no GPU available)")
        else:
            self.device = device
        
        # Baseline model (for comparison)
        self.baseline_network = None
    
    def _get_checkpoint_path(self, iteration: Union[int, str]) -> str:
        """
        Get path for checkpoint at a specific iteration.
        
        Args:
            iteration: Iteration number or 'latest'
            
        Returns:
            Path to the checkpoint file
        """
        if iteration == "latest":
            # Find the latest checkpoint
            checkpoint_files = sorted(list(Path(self.models_dir).glob("muzero_iteration_*.pt")))
            if not checkpoint_files:
                raise ValueError(f"No checkpoints found in {self.models_dir}")
            
            return str(checkpoint_files[-1])
        else:
            # Specific iteration
            return os.path.join(self.models_dir, f"muzero_iteration_{iteration}.pt")
    
    def _get_iteration_from_path(self, checkpoint_path: str) -> int:
        """Extract iteration number from checkpoint path."""
        filename = os.path.basename(checkpoint_path)
        return int(filename.split("_")[2].split(".")[0])
    
    def load_model(self, iteration: Union[int, str] = "latest") -> Tuple[MuZeroNetwork, int]:
        """
        Load a model from checkpoint.
        
        Args:
            iteration: Iteration number or 'latest'
            
        Returns:
            Loaded network and iteration number
        """
        checkpoint_path = self._get_checkpoint_path(iteration)
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Create network
        network = MuZeroNetwork(
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        network.load_state_dict(checkpoint['network_state_dict'])
        network = network.to(self.device)
        
        # Get iteration number
        if iteration == "latest":
            loaded_iteration = self._get_iteration_from_path(checkpoint_path)
        else:
            loaded_iteration = iteration
        
        logger.info(f"Loaded model from iteration {loaded_iteration}")
        return network, loaded_iteration
    
    def load_baseline_model(self, iteration: Optional[Union[int, str]] = None) -> None:
        """
        Load a baseline model for comparison.
        If iteration is None, uses a random policy.
        
        Args:
            iteration: Iteration number or 'latest' or None for random
        """
        if iteration is None:
            logger.info("Using random policy as baseline")
            self.baseline_network = None
        else:
            self.baseline_network, baseline_iteration = self.load_model(iteration)
            logger.info(f"Using model from iteration {baseline_iteration} as baseline")
    
    def _play_evaluation_game(
        self,
        network: MuZeroNetwork,
        baseline_network: Optional[MuZeroNetwork],
        num_simulations: int = 50,
        temperature: float = 0.0  # Use temperature 0 for deterministic evaluation
    ) -> Tuple[int, int]:
        """
        Play a single evaluation game between the model and baseline.
        
        Args:
            network: MuZero network to evaluate
            baseline_network: Baseline network (or None for random policy)
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection
            
        Returns:
            Tuple of (result, game_length) where:
                - result is 1 (model win), 0 (draw), -1 (model loss)
                - game_length is the number of moves in the game
        """
        # Initialize environment
        env = NardeEnv()
        current_obs, info = env.reset()
        
        # Initialize MCTS for both players
        model_mcts = MCTS(
            network=network, 
            num_simulations=num_simulations,
            action_space_size=self.action_dim,
            device=self.device
        )
        
        if baseline_network is not None:
            baseline_mcts = MCTS(
                network=baseline_network,
                num_simulations=num_simulations,
                action_space_size=self.action_dim,
                device=self.device
            )
        
        # Play the game
        done = False
        game_length = 0
        model_player = 0  # Start as player 0
        
        # Get access to the unwrapped environment for easier state manipulation
        unwrapped_env = env.unwrapped
        
        while not done:
            # Determine which player's turn it is
            is_model_turn = (unwrapped_env.game.current_player_idx == model_player)
            
            # Get valid actions
            from muzero.training import get_valid_action_indices
            valid_actions = get_valid_action_indices(env)
            
            if not valid_actions:
                # No valid moves available, skip turn
                next_obs, reward, done, truncated, info = env.step((0, 0))  # Dummy action
                if done:
                    break
                continue
            
            # Select action
            if is_model_turn:
                # Model's turn
                policy = model_mcts.run(
                    observation=current_obs,
                    valid_actions=valid_actions,
                    add_exploration_noise=False
                )
                
                # Deterministic selection for evaluation
                action_idx = np.argmax(policy)
            else:
                # Baseline's turn
                if baseline_network is None:
                    # Random policy
                    action_idx = np.random.choice(valid_actions)
                else:
                    # Baseline MCTS
                    policy = baseline_mcts.run(
                        observation=current_obs,
                        valid_actions=valid_actions,
                        add_exploration_noise=False
                    )
                    action_idx = np.argmax(policy)
            
            # Convert action index to action tuple
            move_type = 0  # Default to regular move
            from_pos = action_idx // 24
            
            for move in unwrapped_env.game.get_valid_moves():
                if move[0] == from_pos and move[1] == 'off':
                    move_type = 1  # It's a bear-off
                    break
                    
            action = (action_idx, move_type)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            current_obs = next_obs
            game_length += 1
            
            if truncated:
                done = True
        
        # Determine game result
        if unwrapped_env.game.winner == model_player:
            result = 1  # Model won
        elif unwrapped_env.game.winner is None:
            result = 0  # Draw
        else:
            result = -1  # Model lost
        
        return result, game_length
    
    def evaluate_model(
        self,
        iteration: Union[int, str] = "latest",
        baseline_iteration: Optional[Union[int, str]] = None,
        num_games: int = 100,
        num_simulations: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate a model by playing against a baseline model.
        
        Args:
            iteration: Iteration to evaluate or 'latest'
            baseline_iteration: Baseline iteration to compare against, or None for random
            num_games: Number of games to play
            num_simulations: Number of MCTS simulations per move
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load models
        network, loaded_iteration = self.load_model(iteration)
        self.load_baseline_model(baseline_iteration)
        
        # Create evaluation directory
        eval_dir = os.path.join(self.eval_dir, f"iteration_{loaded_iteration}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Initialize metrics
        wins = 0
        draws = 0
        losses = 0
        game_lengths = []
        
        logger.info(f"Evaluating model from iteration {loaded_iteration}")
        logger.info(f"Playing {num_games} games with {num_simulations} simulations per move")
        
        start_time = time.time()
        
        # Play evaluation games
        for game_id in range(num_games):
            result, game_length = self._play_evaluation_game(
                network=network,
                baseline_network=self.baseline_network,
                num_simulations=num_simulations
            )
            
            # Update metrics
            if result > 0:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
                
            game_lengths.append(game_length)
            
            # Log progress
            if (game_id + 1) % 10 == 0 or (game_id + 1) == num_games:
                elapsed = time.time() - start_time
                games_per_second = (game_id + 1) / elapsed
                win_rate = (wins / (game_id + 1)) * 100
                logger.info(f"Completed {game_id+1}/{num_games} games " 
                           f"({games_per_second:.2f} games/s), "
                           f"Win rate: {win_rate:.2f}%")
        
        # Calculate metrics
        win_rate = (wins / num_games) * 100
        draw_rate = (draws / num_games) * 100
        loss_rate = (losses / num_games) * 100
        avg_game_length = sum(game_lengths) / len(game_lengths)
        
        total_time = time.time() - start_time
        games_per_second = num_games / total_time
        
        # Create results dictionary
        results = {
            "iteration": loaded_iteration,
            "baseline_iteration": baseline_iteration,
            "num_games": num_games,
            "num_simulations": num_simulations,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "avg_game_length": avg_game_length,
            "total_time": total_time,
            "games_per_second": games_per_second,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log results
        logger.info(f"Evaluation results (iteration {loaded_iteration}):")
        logger.info(f"  Win rate: {win_rate:.2f}%")
        logger.info(f"  Draw rate: {draw_rate:.2f}%")
        logger.info(f"  Loss rate: {loss_rate:.2f}%")
        logger.info(f"  Average game length: {avg_game_length:.2f} moves")
        logger.info(f"  Evaluation speed: {games_per_second:.2f} games/s")
        
        # Save results
        results_file = os.path.join(eval_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved evaluation results to {results_file}")
        
        return results
    
    def compare_iterations(self, iterations: Optional[List[Union[int, str]]] = None) -> None:
        """
        Compare performance across multiple iterations.
        
        Args:
            iterations: List of iterations to compare, or None for all available
        """
        if iterations is None:
            # Find all evaluation results
            result_files = list(Path(self.eval_dir).glob("iteration_*/results.json"))
            if not result_files:
                logger.warning("No evaluation results found")
                return
                
            results = []
            for result_file in result_files:
                with open(result_file, "r") as f:
                    results.append(json.load(f))
        else:
            # Load specified iterations
            results = []
            for iteration in iterations:
                if isinstance(iteration, str) and iteration == "latest":
                    # Find the latest iteration
                    checkpoint_path = self._get_checkpoint_path("latest")
                    iteration = self._get_iteration_from_path(checkpoint_path)
                
                result_file = os.path.join(self.eval_dir, f"iteration_{iteration}", "results.json")
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        results.append(json.load(f))
                else:
                    logger.warning(f"No evaluation results found for iteration {iteration}")
        
        if not results:
            logger.warning("No evaluation results to compare")
            return
            
        # Sort by iteration
        results.sort(key=lambda x: x["iteration"])
        
        # Plot performance trends
        self._plot_performance_trends(results)
    
    def _plot_performance_trends(self, results: List[Dict[str, Any]]) -> None:
        """Plot performance trends across iterations."""
        iterations = [r["iteration"] for r in results]
        win_rates = [r["win_rate"] for r in results]
        draw_rates = [r["draw_rate"] for r in results]
        loss_rates = [r["loss_rate"] for r in results]
        game_lengths = [r["avg_game_length"] for r in results]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot win/draw/loss rates
        ax1.plot(iterations, win_rates, 'g-o', label='Win Rate')
        ax1.plot(iterations, draw_rates, 'b-o', label='Draw Rate')
        ax1.plot(iterations, loss_rates, 'r-o', label='Loss Rate')
        ax1.set_title('MuZero Performance Across Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Rate (%)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot game length
        ax2.plot(iterations, game_lengths, 'm-o')
        ax2.set_title('Average Game Length Across Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Game Length (moves)')
        ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, "performance_trends.png"))
        logger.info(f"Saved performance trends to {os.path.join(self.eval_dir, 'performance_trends.png')}")
        
        # Print trend summary
        if len(results) > 1:
            first_result = results[0]
            last_result = results[-1]
            
            win_rate_change = last_result["win_rate"] - first_result["win_rate"]
            game_length_change = last_result["avg_game_length"] - first_result["avg_game_length"]
            
            logger.info("Performance trends summary:")
            logger.info(f"  Win rate: {first_result['win_rate']:.2f}% → {last_result['win_rate']:.2f}% ({win_rate_change:+.2f}%)")
            logger.info(f"  Avg game length: {first_result['avg_game_length']:.2f} → {last_result['avg_game_length']:.2f} ({game_length_change:+.2f})")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MuZero Model Evaluation")
    
    # Evaluation parameters
    parser.add_argument("--base_dir", type=str, default="muzero_training", 
                        help="Base directory for model checkpoints and evaluation results")
    parser.add_argument("--iteration", type=str, default="latest", 
                        help="Iteration to evaluate (number or 'latest')")
    parser.add_argument("--baseline_iteration", type=str, default=None, 
                        help="Baseline iteration to compare against (number, 'latest', or None for random)")
    parser.add_argument("--num_games", type=int, default=100, 
                        help="Number of games to play for evaluation")
    parser.add_argument("--num_simulations", type=int, default=50, 
                        help="Number of MCTS simulations per move")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare performance across all evaluated iterations")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for evaluation (default: auto-detect)")
                        
    # Network parameters
    parser.add_argument("--hidden_dim", type=int, default=128, 
                        help="Hidden dimension for the MuZero network")
    parser.add_argument("--latent_dim", type=int, default=64, 
                        help="Latent dimension for the MuZero network")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        base_dir=args.base_dir,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        device=args.device
    )
    
    # Convert iteration argument if it's a number
    if args.iteration != "latest":
        try:
            args.iteration = int(args.iteration)
        except ValueError:
            logger.error(f"Invalid iteration: {args.iteration}")
            exit(1)
    
    # Convert baseline iteration argument
    if args.baseline_iteration is not None and args.baseline_iteration != "latest":
        try:
            args.baseline_iteration = int(args.baseline_iteration)
        except ValueError:
            logger.error(f"Invalid baseline iteration: {args.baseline_iteration}")
            exit(1)
    
    # Evaluate model
    evaluator.evaluate_model(
        iteration=args.iteration,
        baseline_iteration=args.baseline_iteration,
        num_games=args.num_games,
        num_simulations=args.num_simulations
    )
    
    # Compare iterations if requested
    if args.compare:
        evaluator.compare_iterations() 