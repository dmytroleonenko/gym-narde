#!/usr/bin/env python3
"""
Profiler for MuZero training cycle on one episode to identify bottlenecks.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time
import os
import cProfile
import pstats
import io
from pstats import SortKey
import argparse
from contextlib import contextmanager

# Import MuZero components
from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS
from muzero.replay import ReplayBuffer
from muzero.training import get_valid_action_indices, self_play_game, update_weights


class TimerContext:
    """Context manager for timing code blocks."""
    def __init__(self, name, timers_dict):
        self.name = name
        self.timers_dict = timers_dict
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if self.name in self.timers_dict:
            self.timers_dict[self.name]['total_time'] += elapsed_time
            self.timers_dict[self.name]['count'] += 1
        else:
            self.timers_dict[self.name] = {'total_time': elapsed_time, 'count': 1}


def profile_muzero_episode(
    hidden_dim=128,
    latent_dim=64,
    num_simulations=50,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=1e-4,
    device_str="auto"
):
    """
    Profile a single episode of MuZero training to identify bottlenecks.
    
    Args:
        hidden_dim: Hidden dimension of the network
        latent_dim: Latent state dimension of the network
        num_simulations: Number of MCTS simulations per move
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        device_str: Device to run on ('cpu', 'cuda', 'mps', or 'auto')
    """
    # Set up device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    
    # Initialize environment
    env = gym.make('Narde-v0')
    observation_shape = env.observation_space.shape
    
    # For Narde, the action space is a Tuple (Discrete(576), Discrete(2))
    # The first action is from_pos * 24 + to_pos (576 combinations)
    # The second action is 0 for regular moves, 1 for bear-off moves
    action_dim = 576  # 24*24 possible move combinations
    
    # Dictionary to store timing information
    timers = {}
    
    # Create MuZero network
    with TimerContext("Network initialization", timers):
        input_dim = observation_shape[0]  # The flattened observation dimension (28 for Narde)
        network = MuZeroNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Create MCTS
    with TimerContext("MCTS initialization", timers):
        mcts = MCTS(
            network=network,
            num_simulations=num_simulations,
            discount=0.997,
            dirichlet_alpha=0.3,
            exploration_fraction=0.25,
            action_space_size=action_dim,
            device=device
        )
    
    # Create replay buffer
    with TimerContext("Replay buffer initialization", timers):
        replay_buffer = ReplayBuffer(capacity=10000)
    
    # Self-play one episode
    with TimerContext("Self-play episode", timers):
        game_history = self_play_game(
            env=env,
            network=network,
            mcts=mcts,
            num_simulations=num_simulations,
            temperature=1.0,
            discount=0.997,
            device=device
        )
    
    # Process the game history
    with TimerContext("Process game history", timers):
        replay_buffer.save_game(game_history)
    
    # Train on a batch of data
    with TimerContext("Training", timers):
        # Extract a batch of data
        if len(replay_buffer) > batch_size:
            with TimerContext("Batch preparation", timers):
                batch = replay_buffer.sample(batch_size)
                
                # Prepare batch for training
                observations = []
                actions = []
                target_values = []
                target_policies = []
                bootstrap_positions = []
                
                for obs, action, target_value, target_policy in batch:
                    observations.append(obs)
                    actions.append(action)
                    target_values.append(target_value)
                    target_policies.append(target_policy)
                    bootstrap_positions.append(0)  # Not used in this simple example
                
                # Convert to tensors
                observations = torch.FloatTensor(np.array(observations)).to(device)
                actions = torch.LongTensor(np.array(actions)).to(device)
                target_values = torch.FloatTensor(np.array(target_values)).to(device)
                target_policies = torch.FloatTensor(np.array(target_policies)).to(device)
                bootstrap_positions = torch.LongTensor(np.array(bootstrap_positions)).to(device)
                
                processed_batch = (observations, actions, target_values, target_policies, bootstrap_positions)
            
            # Update weights
            with TimerContext("Weight update", timers):
                value_loss, policy_loss, reward_loss = update_weights(
                    optimizer=optimizer,
                    network=network,
                    batch=processed_batch,
                    device=device
                )
                
                print(f"Value loss: {value_loss:.4f}, Policy loss: {policy_loss:.4f}, Reward loss: {reward_loss:.4f}")
    
    # Detailed analysis of MCTS
    print("\nProfiling MCTS operations for a single step...")
    
    # Reset environment
    observation, info = env.reset()
    valid_actions = get_valid_action_indices(env)
    
    # Profile MCTS run
    with TimerContext("MCTS single run", timers):
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(device)
        
        mcts_timers = {}
        
        # Initial inference
        with TimerContext("Initial inference", mcts_timers):
            with torch.no_grad():
                root_hidden_state, root_value, root_policy_logits = network.initial_inference(observation.unsqueeze(0))
        
        # Convert policy logits to probabilities
        with TimerContext("Policy conversion", mcts_timers):
            # Apply mask to policy logits (set invalid actions to -inf)
            action_mask = torch.zeros(action_dim, device=device)
            action_mask[valid_actions] = 1.0
            
            masked_policy_logits = root_policy_logits.clone()
            masked_policy_logits[0, ~action_mask.bool()] = float('-inf')
            root_policy = torch.softmax(masked_policy_logits, dim=1).squeeze(0)
        
        # Run simulations
        with TimerContext("MCTS simulations", mcts_timers):
            # Store batch inference and backpropagation timing
            batch_inference_time = 0
            backprop_time = 0
            
            num_simulations_profiled = min(10, num_simulations)  # Limit for detailed profiling
            
            for _ in range(num_simulations_profiled):
                # Simulate one step (simplified)
                start_time = time.time()
                parent_hidden = root_hidden_state
                action = valid_actions[0]  # Just use first action for profiling
                
                # Recurrent inference
                inference_start = time.time()
                with torch.no_grad():
                    next_hidden, reward, value, policy_logits = network.recurrent_inference(
                        parent_hidden,
                        torch.tensor([action], device=device)
                    )
                inference_time = time.time() - inference_start
                batch_inference_time += inference_time
                
                # Backpropagation (simplified timing)
                backprop_start = time.time()
                # Simulate backpropagation by doing some computation
                dummy_value = value.item()
                backprop_time += time.time() - backprop_start
        
        # Store MCTS substep timings
        timers["MCTS:initial_inference"] = mcts_timers.get("Initial inference", {"total_time": 0, "count": 0})
        timers["MCTS:policy_conversion"] = mcts_timers.get("Policy conversion", {"total_time": 0, "count": 0})
        timers["MCTS:batch_inference"] = {"total_time": batch_inference_time, "count": num_simulations_profiled}
        timers["MCTS:backpropagation"] = {"total_time": backprop_time, "count": num_simulations_profiled}
    
    # Detailed analysis of batch training step
    print("\nProfiling batch training operations...")
    
    if len(replay_buffer) > batch_size:
        # Use the previously prepared batch
        train_timers = {}
        
        # Forward pass
        with TimerContext("Forward pass", train_timers):
            hidden_states, predicted_values, predicted_policy_logits = network.initial_inference(observations)
        
        # Policy loss
        with TimerContext("Policy loss", train_timers):
            policy_targets = target_policies.argmax(dim=1)
            policy_loss = torch.nn.functional.cross_entropy(
                predicted_policy_logits,
                policy_targets
            )
        
        # Value loss
        with TimerContext("Value loss", train_timers):
            scaled_target_values = torch.clamp(target_values / 15.0, -1.0, 1.0)
            value_loss = torch.nn.functional.mse_loss(predicted_values.squeeze(-1), scaled_target_values)
        
        # Dynamics network
        with TimerContext("Dynamics network", train_timers):
            action_onehots = torch.zeros(actions.size(0), network.action_dim, device=device)
            action_onehots.scatter_(1, actions.unsqueeze(1), 1.0)
            
            next_hidden_states, predicted_rewards = network.dynamics_network(hidden_states, action_onehots)
        
        # Reward loss
        with TimerContext("Reward loss", train_timers):
            reward_loss = torch.nn.functional.mse_loss(predicted_rewards.squeeze(-1), target_values / 10.0)
        
        # Backward pass
        with TimerContext("Backward pass", train_timers):
            total_loss = value_loss + policy_loss + reward_loss
            total_loss.backward()
        
        # Optimizer step
        with TimerContext("Optimizer step", train_timers):
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
        
        # Store training substep timings
        for key, value in train_timers.items():
            timers[f"Training:{key}"] = value
    
    # Environment operations profiling
    print("\nProfiling environment operations...")
    
    env_timers = {}
    
    # Reset
    with TimerContext("Environment reset", env_timers):
        for _ in range(10):  # Do multiple to get better timing
            observation, info = env.reset()
    
    # Get valid actions
    with TimerContext("Get valid actions", env_timers):
        for _ in range(10):
            valid_actions = get_valid_action_indices(env)
    
    # Step
    with TimerContext("Environment step", env_timers):
        for _ in range(10):
            if len(valid_actions) > 0:
                action_idx = valid_actions[0]
                action = (action_idx, 0)  # Assume regular move
                next_obs, reward, done, truncated, info = env.step(action)
                if done:
                    observation, info = env.reset()
    
    # Store environment timings
    for key, value in env_timers.items():
        value["total_time"] /= 10  # Average over 10 iterations
        timers[f"Environment:{key}"] = value
    
    # Print timing results
    print("\n===== MUZERO TRAINING CYCLE PROFILING RESULTS =====")
    
    # Group timings by category
    categories = {
        "Initialization": ["Network initialization", "MCTS initialization", "Replay buffer initialization"],
        "Self-play": ["Self-play episode"],
        "MCTS": ["MCTS single run", "MCTS:initial_inference", "MCTS:policy_conversion", 
                "MCTS:batch_inference", "MCTS:backpropagation"],
        "Training": ["Training", "Process game history", "Batch preparation", "Weight update", 
                    "Training:Forward pass", "Training:Policy loss", "Training:Value loss", 
                    "Training:Dynamics network", "Training:Reward loss", "Training:Backward pass", 
                    "Training:Optimizer step"],
        "Environment": ["Environment:Environment reset", "Environment:Get valid actions", "Environment:Environment step"]
    }
    
    # Calculate total time for each category
    category_totals = {}
    for category, operations in categories.items():
        category_time = sum(timers.get(op, {"total_time": 0})["total_time"] for op in operations)
        category_totals[category] = category_time
    
    # Calculate grand total
    grand_total = sum(category_totals.values())
    
    # Print category breakdown
    print("\nTime breakdown by category:")
    print("--------------------------")
    for category, total_time in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (total_time / grand_total) * 100
        print(f"{category:<15}: {total_time:.4f}s ({percentage:.1f}%)")
    
    # Print detailed breakdown by operation
    print("\nDetailed time breakdown:")
    print("----------------------")
    for category, operations in categories.items():
        print(f"\n{category}:")
        for op in operations:
            if op in timers:
                time_data = timers[op]
                total_time = time_data["total_time"]
                count = time_data["count"]
                avg_time = total_time / count
                percentage = (total_time / grand_total) * 100
                print(f"  {op:<30}: {total_time:.6f}s total, {avg_time:.6f}s avg ({count} calls) - {percentage:.1f}%")
    
    # Print hardware recommendations based on profiling
    print("\n===== HARDWARE ACCELERATION RECOMMENDATIONS =====")
    
    # Check for operations that might benefit from GPU acceleration
    total_mcts_time = category_totals.get("MCTS", 0)
    total_training_time = category_totals.get("Training", 0)
    total_env_time = category_totals.get("Environment", 0)
    
    if total_mcts_time > total_training_time and total_mcts_time > total_env_time:
        print("\nMCTS is the primary bottleneck:")
        if timers.get("MCTS:batch_inference", {"total_time": 0})["total_time"] > 0.5 * total_mcts_time:
            print("- Batch inference in MCTS is the main bottleneck")
            print("- Consider using GPU acceleration for neural network operations")
            print("- Increase batch size for MCTS simulations")
            
    elif total_training_time > total_mcts_time and total_training_time > total_env_time:
        print("\nTraining is the primary bottleneck:")
        if timers.get("Training:Forward pass", {"total_time": 0})["total_time"] > 0.3 * total_training_time or \
           timers.get("Training:Backward pass", {"total_time": 0})["total_time"] > 0.3 * total_training_time:
            print("- Neural network forward/backward pass is the main bottleneck")
            print("- GPU acceleration would significantly improve performance")
            print("- Consider using mixed precision training for further acceleration")
    
    elif total_env_time > total_mcts_time and total_env_time > total_training_time:
        print("\nEnvironment operations are the primary bottleneck:")
        if timers.get("Environment:Get valid actions", {"total_time": 0})["total_time"] > 0.5 * total_env_time:
            print("- Get valid actions operation is the main bottleneck")
            print("- Consider optimizing the environment implementation or using vectorized environments")
            print("- Batching environment operations could provide significant speedup")
    
    # Based on the hardware reports provided
    print("\nBased on hardware acceleration reports:")
    print(f"- Running on: {device}")
    
    if device.type == "cuda":
        print("- For CUDA devices, neural networks show the best acceleration (up to 42x)")
        print("- Large models (512+ hidden dim) benefit more from GPU acceleration")
        print("- Environment operations like board rotation show moderate acceleration (3.5x)")
        print("- Complex operations like get_valid_actions perform better on CPU")
    elif device.type == "mps":
        print("- For Apple Silicon with MPS, matrix operations show good acceleration (up to 8x)")
        print("- Neural networks show moderate acceleration (up to 2x)")
        print("- Batch sizes ≥512 are recommended for good MPS performance")
        print("- Board operations benefit from MPS at large batch sizes")
    else:
        print("- Consider using GPU acceleration for neural network operations")
        print("- NVIDIA GPUs provide up to 42x speedup for large MuZero models")
        print("- Apple Silicon with MPS provides up to 2x speedup for neural networks")
        print("- Environment operations benefit from batching regardless of hardware")
    
    return timers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile MuZero training cycle")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of the network")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent state dimension of the network")
    parser.add_argument("--num_simulations", type=int, default=50, help="Number of MCTS simulations per move")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on ('cpu', 'cuda', 'mps', or 'auto')")
    
    args = parser.parse_args()
    
    # Set up cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    # Run profiling
    timers = profile_muzero_episode(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device_str=args.device
    )
    
    # Disable profiler
    pr.disable()
    
    # Print cProfile results
    print("\n===== DETAILED PYTHON PROFILING RESULTS =====")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    # Save profiling results to file
    print("\nSaving detailed profiling results to muzero_profile_results.txt")
    with open("muzero_profile_results.txt", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats() 