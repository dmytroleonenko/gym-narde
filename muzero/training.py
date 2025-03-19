#!/usr/bin/env python3
"""
Training script for MuZero on the Narde environment.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import random
import os
import time
import argparse
import json
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import deque

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set or couldn't be set

from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS
from muzero.replay import ReplayBuffer


def decode_action(action_index, move_type=0):
    """
    Convert action index to (from_pos, to_pos) or (from_pos, 'off').
    """
    from_pos = action_index // 24
    to_pos = action_index % 24
    
    if move_type == 1:
        return (from_pos, 'off')
    return (from_pos, to_pos)


def encode_action(move):
    """
    Convert (from_pos, to_pos) or (from_pos, 'off') to action index.
    """
    if move[1] == 'off':
        return move[0] * 24, 1  # move_type=1 for bear-off
    else:
        return move[0] * 24 + move[1], 0  # move_type=0 for regular move


def get_valid_action_indices(env):
    """
    Get the indices of valid actions from the environment.
    
    Args:
        env: The NardeEnv environment
        
    Returns:
        List of valid action indices
    """
    # Get access to the unwrapped environment
    unwrapped_env = env.unwrapped
    
    valid_moves = unwrapped_env.game.get_valid_moves()
    valid_actions = []
    
    for move in valid_moves:
        # Convert move to action index
        if move[1] == 'off':
            # Bear off move: (from_pos, 'off')
            action_idx = move[0] * 24  # For now, we'll encode action as if to_pos=0
        else:
            # Regular move: (from_pos, to_pos)
            action_idx = move[0] * 24 + move[1]
            
        valid_actions.append(action_idx)
        
    return valid_actions


def self_play_game(env, network, mcts, num_simulations, temperature=1.0, discount=0.997, device="cpu"):
    """
    Play a full game using MCTS and the current network.
    
    Args:
        env: The NardeEnv environment
        network: The MuZero network
        mcts: The MCTS instance
        num_simulations: Number of MCTS simulations per move
        temperature: Temperature parameter for action selection
        discount: Discount factor for rewards
        device: Device to run the network on
        
    Returns:
        game_history: List of (observation, action, reward, policy) tuples
    """
    # Reset the environment and get initial state
    current_obs, info = env.reset()
    done = False
    game_history = []
    
    # Get access to the unwrapped environment
    unwrapped_env = env.unwrapped
    
    while not done:
        # Get valid actions
        valid_actions = get_valid_action_indices(env)
        
        if not valid_actions:
            # No valid moves available, skip turn
            next_obs, reward, done, truncated, info = env.step((0, 0))  # Dummy action, will be ignored
            
            if 'skipped_turn' in info and info['skipped_turn']:
                # Turn was skipped, continue with next player
                current_obs = next_obs
                continue
                
        # Run MCTS to get a policy
        mcts_policy = mcts.run(
            observation=current_obs,
            valid_actions=valid_actions,
            add_exploration_noise=True
        )
        
        # If no valid actions, skip
        if len(valid_actions) == 0:
            continue
            
        # Select action based on the MCTS policy
        if temperature == 0:
            # Deterministic selection at temperature 0
            action_idx = np.argmax(mcts_policy)
        else:
            # Sample with temperature
            policy = mcts_policy ** (1 / temperature)
            policy /= policy.sum()
            action_idx = np.random.choice(len(mcts_policy), p=policy)
            
        # Convert action index to action tuple (move_index, move_type)
        # For Narde, we need to determine if it's a bear-off action
        # We'll check the valid moves to see if this is a bear-off
        move_type = 0  # Default to regular move
        from_pos = action_idx // 24
        
        for move in unwrapped_env.game.get_valid_moves():
            if move[0] == from_pos and move[1] == 'off':
                move_type = 1  # It's a bear-off
                break
                
        action = (action_idx, move_type)
        
        # Take the action
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store the transition
        game_history.append((current_obs, action_idx, reward, mcts_policy))
        
        # Update current observation
        current_obs = next_obs
        
        if truncated:
            done = True
            
    return game_history


def update_weights(optimizer, network, batch, device="cpu"):
    """
    Update the network weights based on a batch of data.
    
    Args:
        optimizer: The optimizer
        network: The MuZero network
        batch: A batch of (observation, action, target_value, target_policy, bootstrap_position) tuples
        device: Device to run the network on
        
    Returns:
        loss_value: The value loss
        loss_policy: The policy loss
        loss_reward: The reward loss
    """
    optimizer.zero_grad()
    
    observations, actions, target_values, target_policies, bootstrap_positions = batch
    
    # Initial inference for the representation network and prediction network
    hidden_states, predicted_values, predicted_policy_logits = network.initial_inference(observations)
    
    # Policy loss - use target policy argmax for stability
    policy_targets = target_policies.argmax(dim=1)
    policy_loss = torch.nn.functional.cross_entropy(
        predicted_policy_logits,
        policy_targets
    )
    
    # Value loss - scale to [-1, 1] range to match tanh output
    scaled_target_values = torch.clamp(target_values / 15.0, -1.0, 1.0)  # Max reward is 15 (all pieces borne off)
    value_loss = torch.nn.functional.mse_loss(predicted_values.squeeze(-1), scaled_target_values)
    
    # Convert actions to one-hot for the dynamics network - do this in single efficient operation
    action_onehots = torch.zeros(actions.size(0), network.action_dim, device=device)
    action_onehots.scatter_(1, actions.unsqueeze(1), 1.0)
    
    # Recurrent inference for dynamics network
    next_hidden_states, predicted_rewards = network.dynamics_network(hidden_states, action_onehots)
    
    # Reward loss - typically small rewards in Narde, so scale accordingly
    reward_loss = torch.nn.functional.mse_loss(predicted_rewards.squeeze(-1), target_values / 10.0)
    
    # Total loss
    total_loss = value_loss + policy_loss + reward_loss
    
    # Backpropagation
    total_loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    return value_loss.item(), policy_loss.item(), reward_loss.item()


# Add this helper function outside of any other function to make it picklable
def run_game_with_args(args):
    """
    Helper function for running self-play games in parallel.
    Unpacks args and calls run_self_play_game_process.
    
    Args:
        args: Tuple containing (game_id, seed, model_path, num_simulations, temperature)
        
    Returns:
        Result from run_self_play_game_process
    """
    game_id, seed, model_path, num_simulations, temperature = args
    return run_self_play_game_process(
        game_id=game_id,
        seed=seed,
        model_path=model_path,
        num_simulations=num_simulations,
        temperature=temperature
    )


def run_self_play_game_process(game_id, seed=None, model_path=None, num_simulations=50, temperature=1.0):
    """
    Run a self-play game in a separate process.
    
    Args:
        game_id: ID of the game (for tracking)
        seed: Random seed for reproducibility
        model_path: Path to the model to use
        num_simulations: Number of MCTS simulations
        temperature: Temperature for action selection
        
    Returns:
        Dictionary with game history
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Create environment
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    
    # Create network and load model
    network = MuZeroNetwork(input_dim, action_dim).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()  # Set to evaluation mode
    
    # Create MCTS
    mcts = MCTS(
        network=network,
        num_simulations=num_simulations,
        discount=0.997,  # Default discount
        action_space_size=action_dim,
        device=device
    )
    
    # Run self-play game
    game_history = self_play_game(
        env=env,
        network=network,
        mcts=mcts,
        num_simulations=num_simulations,
        temperature=temperature,
        discount=0.997,
        device=device
    )
    
    # Convert game history to serializable format
    serializable_history = []
    for obs, action_idx, reward, policy in game_history:
        serializable_history.append({
            "observation": obs.tolist(),
            "action_idx": int(action_idx),
            "reward": float(reward),
            "policy": policy.tolist()
        })
    
    return {
        "game_id": game_id,
        "game_history": serializable_history,
        "game_length": len(serializable_history),
        "total_reward": sum(item["reward"] for item in serializable_history)
    }


def convert_game_history_from_process(serialized_history):
    """
    Convert serialized game history back to the format expected by ReplayBuffer.
    
    Args:
        serialized_history: Serialized game history from process
        
    Returns:
        Game history in the format expected by ReplayBuffer
    """
    game_history = []
    for item in serialized_history:
        game_history.append((
            np.array(item["observation"], dtype=np.bfloat16),
            item["action_idx"],
            item["reward"],
            np.array(item["policy"], dtype=np.bfloat16)
        ))
    return game_history


def train_muzero(
    num_episodes=1000,
    replay_buffer_size=10000,
    batch_size=128,
    lr=0.001,
    num_simulations=50,
    temperature_init=1.0,
    temperature_final=0.1,
    weight_decay=1e-4,
    discount=0.997,
    save_interval=100,
    hidden_dim=256,
    device_str="auto",
    enable_profiling=False,
    profile_episodes=5,
    start_with_model=None,
    previous_episodes=0,
    total_episodes=None,
    parallel_self_play=1  # Number of parallel self-play processes
):
    """
    Train MuZero on the Narde environment.
    
    Args:
        num_episodes: Number of episodes to train for
        replay_buffer_size: Size of the replay buffer
        batch_size: Maximum batch size for training
        lr: Learning rate
        num_simulations: Number of MCTS simulations per move
        temperature_init: Initial temperature for action selection
        temperature_final: Final temperature for action selection
        weight_decay: L2 regularization strength
        discount: Discount factor for rewards
        save_interval: Save the model every save_interval episodes
        hidden_dim: Dimensionality of the latent state
        device_str: Device to run the training on ("auto", "cpu", "cuda", "mps")
        enable_profiling: Whether to enable PyTorch profiler
        profile_episodes: Number of episodes to profile
        start_with_model: Path to a model to continue training from
        previous_episodes: Number of episodes previously trained 
        total_episodes: Total episodes including previous ones (for temperature scheduling)
        parallel_self_play: Number of parallel self-play processes (1 = no parallelization)
    """
    # Use total_episodes for temperature calculation if provided
    if total_episodes is None:
        total_episodes = num_episodes
    
    # Determine device
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
    
    # Create environment for initial setup
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    
    # Create MuZero network
    network = MuZeroNetwork(input_dim, action_dim, hidden_dim=hidden_dim).to(device)
    
    # Load model if continuing training
    if start_with_model is not None and os.path.exists(start_with_model):
        print(f"Loading model weights from {start_with_model}")
        network.load_state_dict(torch.load(start_with_model, map_location=device))
    
    # Create optimizer
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size, n_step_return=5, discount=discount)
    
    # Create MCTS
    mcts = MCTS(
        network=network,
        num_simulations=num_simulations,
        discount=discount,
        action_space_size=action_dim,
        device=device
    )
    
    # Create directory for saving models
    os.makedirs("muzero/models", exist_ok=True)
    
    # For parallel self-play, we need a temp model file
    temp_model_path = None
    if parallel_self_play > 1:
        os.makedirs("muzero/temp", exist_ok=True)
        temp_model_path = "muzero/temp/muzero_model_temp.pth"
        print(f"Enabling parallel self-play with {parallel_self_play} processes")
    
    # Create directory for profiler logs
    if enable_profiling:
        os.makedirs("muzero/profiler_logs", exist_ok=True)
    
    # Training loop
    total_steps = 0
    avg_reward = deque(maxlen=100)
    avg_length = deque(maxlen=100)
    
    print(f"Starting training for {num_episodes} episodes...")
    if previous_episodes > 0:
        print(f"Continuing from episode {previous_episodes} (total will be {previous_episodes + num_episodes})")
    
    # Set up profiler based on device type
    profiler_activities = [ProfilerActivity.CPU]
    
    # Add GPU profiling if available
    if device.type == 'cuda':
        profiler_activities.append(ProfilerActivity.CUDA)
    
    # Create profiler context manager
    profiler_ctx = (
        torch.profiler.profile(
            activities=profiler_activities,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=min(profile_episodes, num_episodes),
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("muzero/profiler_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        if enable_profiling
        else DummyContextManager()  # Dummy context manager when profiling is disabled
    )
    
    # Track time for performance metrics
    total_self_play_time = 0
    total_training_time = 0
    
    with profiler_ctx as prof:
        episode = 1
        while episode <= num_episodes:
            # Calculate temperature for this episode
            # Anneal from temperature_init to temperature_final based on total progress
            effective_episode = previous_episodes + episode
            progress_fraction = effective_episode / (total_episodes * 0.5)
            progress_fraction = min(1.0, progress_fraction)  # Cap at 1.0
            
            temperature = max(
                temperature_final,
                temperature_init * (1 - progress_fraction)
            )
            
            # Profile self-play games
            with record_function("self_play_games"):
                self_play_start_time = time.time()
                
                # Run self-play games either serially or in parallel
                if parallel_self_play <= 1:
                    # Serial self-play (original implementation)
                    game_history = self_play_game(
                        env=env,
                        network=network,
                        mcts=mcts,
                        num_simulations=num_simulations,
                        temperature=temperature,
                        discount=discount,
                        device=device
                    )
                    
                    # Calculate statistics for this game
                    game_length = len(game_history)
                    total_reward = sum(transition[2] for transition in game_history)
                    
                    # Save game to replay buffer
                    with record_function("save_to_replay_buffer"):
                        replay_buffer.save_game(game_history)
                    
                    # Update metrics
                    avg_reward.append(total_reward)
                    avg_length.append(game_length)
                    total_steps += game_length
                    
                    # Only increment episode counter for serial mode
                    episode += 1
                    
                else:
                    # Parallel self-play
                    # First save the current model to a temp file for worker processes
                    torch.save(network.state_dict(), temp_model_path)
                    
                    # Determine number of episodes to run in parallel (limit to remaining episodes)
                    parallel_episodes = min(parallel_self_play, num_episodes - episode + 1)
                    
                    # Set up function for parallel self-play
                    base_seed = int(time.time()) % 10000  # Use time as base seed
                    
                    # Generate episode IDs and seeds
                    episode_ids = list(range(episode, episode + parallel_episodes))
                    seeds = [base_seed + i for i in range(parallel_episodes)]
                    
                    # Create complete argument tuples for each game
                    arg_tuples = [
                        (game_id, seed, temp_model_path, num_simulations, temperature)
                        for game_id, seed in zip(episode_ids, seeds)
                    ]
                    
                    # Run self-play games in parallel
                    game_results = []
                    with ProcessPoolExecutor(max_workers=parallel_self_play) as executor:
                        # Use map with run_game_with_args directly
                        for result in executor.map(run_game_with_args, arg_tuples):
                            game_results.append(result)
                    
                    # Process results
                    for result in game_results:
                        # Convert game history to format expected by ReplayBuffer
                        game_history = convert_game_history_from_process(result["game_history"])
                        
                        # Save game to replay buffer
                        with record_function("save_to_replay_buffer"):
                            replay_buffer.save_game(game_history)
                        
                        # Update metrics
                        game_length = result["game_length"]
                        total_reward = result["total_reward"]
                        
                        avg_reward.append(total_reward)
                        avg_length.append(game_length)
                        total_steps += game_length
                    
                    # Increment episode counter for parallel mode
                    episode += parallel_episodes
                
                self_play_time = time.time() - self_play_start_time
                total_self_play_time += self_play_time
            
            # Training phase
            training_start_time = time.time()
            
            # Train on a batch - use dynamic batch size
            buffer_size = len(replay_buffer)
            if buffer_size > 0:  # Train as long as we have at least one game
                # Use smaller of batch_size or buffer_size, but at least 1
                actual_batch_size = min(batch_size, buffer_size)
                actual_batch_size = max(1, actual_batch_size)
                
                with record_function("sample_batch"):
                    batch = replay_buffer.sample_batch(actual_batch_size, device=device)
                
                if batch is not None:
                    with record_function("update_weights"):
                        value_loss, policy_loss, reward_loss = update_weights(
                            optimizer, network, batch, device=device
                        )
                else:
                    value_loss, policy_loss, reward_loss = 0, 0, 0
            else:
                value_loss, policy_loss, reward_loss = 0, 0, 0
            
            training_time = time.time() - training_start_time
            total_training_time += training_time
            
            # Calculate the last episode number for reporting
            last_episode = previous_episodes + episode - 1
            
            # Print statistics periodically
            if last_episode % 10 == 0:
                print(f"Episode {last_episode}")
                if parallel_self_play > 1:
                    print(f"  Completed {parallel_self_play} games in parallel")
                print(f"  Average game length: {np.mean(avg_length):.1f}")
                print(f"  Average reward: {np.mean(avg_reward):.4f}")
                print(f"  Temperature: {temperature:.4f}")
                print(f"  Value loss: {value_loss:.4f}, Policy loss: {policy_loss:.4f}, Reward loss: {reward_loss:.4f}")
                print(f"  Self-play time: {self_play_time:.2f}s, Training time: {training_time:.2f}s")
                print(f"  Buffer size: {buffer_size}")
                print(f"  Total steps: {total_steps}")
                print()
                
            # Save model
            if last_episode % save_interval == 0 or last_episode == previous_episodes + num_episodes:
                model_path = f"muzero/models/muzero_model_ep{last_episode}.pth"
                torch.save(network.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            
            # Step the profiler if enabled
            if enable_profiling and episode <= profile_episodes:
                prof.step()
    
    # Print performance metrics
    total_time = total_self_play_time + total_training_time
    print("\nPerformance Summary:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Self-play time: {total_self_play_time:.2f}s ({total_self_play_time/total_time*100:.1f}%)")
    print(f"  Training time: {total_training_time:.2f}s ({total_training_time/total_time*100:.1f}%)")
    if parallel_self_play > 1:
        serial_equivalent = total_steps / np.mean(avg_length)
        speedup = serial_equivalent / (total_time / (total_self_play_time / serial_equivalent))
        print(f"  Estimated speedup from parallelization: {speedup:.2f}x")
    
    # Save final model
    torch.save(network.state_dict(), "muzero/models/muzero_model_final.pth")
    print("Training complete.")


# Simple dummy context manager for when profiling is disabled
class DummyContextManager:
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def step(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MuZero for Narde.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Size of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="Maximum batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--simulations", type=int, default=50, help="Number of MCTS simulations per move")
    parser.add_argument("--temp-init", type=float, default=1.0, help="Initial temperature for action selection")
    parser.add_argument("--temp-final", type=float, default=0.1, help="Final temperature for action selection")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--discount", type=float, default=0.997, help="Discount factor for rewards")
    parser.add_argument("--save-interval", type=int, default=100, help="Save the model every save_interval episodes")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimensionality of the latent state")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the training on (auto, cpu, cuda, mps)")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--profile-episodes", type=int, default=5, help="Number of episodes to profile")
    parser.add_argument("--start-with-model", type=str, help="Path to a model to continue training from")
    parser.add_argument("--previous-episodes", type=int, default=0, help="Number of episodes previously trained")
    
    args = parser.parse_args()
    
    train_muzero(
        num_episodes=args.episodes,
        replay_buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        lr=args.lr,
        num_simulations=args.simulations,
        temperature_init=args.temp_init,
        temperature_final=args.temp_final,
        weight_decay=args.weight_decay,
        discount=args.discount,
        save_interval=args.save_interval,
        hidden_dim=args.hidden_dim,
        device_str=args.device,
        enable_profiling=args.enable_profiling,
        profile_episodes=args.profile_episodes,
        start_with_model=args.start_with_model,
        previous_episodes=args.previous_episodes
    ) 