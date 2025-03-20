#!/usr/bin/env python3
"""
Optimized MuZero training script that uses pre-generated batches and MPS acceleration.
"""

import torch
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing
from muzero.models import MuZeroNetwork
from muzero.batch_collector import AsyncBatchCollector
from muzero.mcts_batched import BatchedMCTS, run_batched_mcts
from muzero.vectorized_env import VectorizedNardeEnv, create_vectorized_env
from muzero.replay import ReplayBuffer

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set or couldn't be set


def train_muzero_optimized(
    num_epochs=1000,
    batch_size=2048,
    buffer_size=8192,
    learning_rate=0.001,
    weight_decay=1e-4,
    checkpoint_path="muzero_model",
    log_dir="logs",
    num_parallel_envs=16,
    device_str=None,
    enable_profiling=False,
    save_interval=100,
):
    """
    Train MuZero with optimized batch processing for MPS acceleration.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Number of samples per batch (using large batches for MPS efficiency)
        buffer_size: Size of the experience buffer
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        checkpoint_path: Path to save model checkpoints
        log_dir: Directory for TensorBoard logs
        num_parallel_envs: Number of parallel environments for data collection
        device_str: Device to use for training ("cuda", "mps", or "cpu")
        enable_profiling: Whether to enable PyTorch profiling
        save_interval: How often to save model checkpoints (in epochs)
    """
    # Set up device
    if device_str is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for training")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS for training")
        else:
            device = torch.device("cpu")
            print("Using CPU for training (no GPU available)")
    else:
        device = torch.device(device_str)
        print(f"Using {device_str} for training")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create model
    collector = AsyncBatchCollector(
        batch_size=batch_size,
        buffer_size=buffer_size,
        num_parallel_envs=num_parallel_envs,
        device=device,
    )
    
    # Get input and action dimensions from collector
    input_dim = collector.collector.input_dim
    action_dim = collector.collector.action_dim
    
    # Create model
    model = MuZeroNetwork(input_dim, action_dim).to(device)
    # Convert model to bfloat16
    model = model.to(torch.bfloat16)
    
    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate / 10,
    )
    
    # Start asynchronous data collection
    collector.start_collection()
    
    # Wait for some initial data
    print("Waiting for initial data collection...")
    time.sleep(5)
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Get a batch
        observations, actions, rewards, next_observations, dones = collector.get_batch()
        
        # Enable profiling if requested
        if enable_profiling and epoch == 0:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                train_on_batch(model, optimizer, observations, actions, rewards, next_observations, dones)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            # Train on batch
            loss_components = train_on_batch(model, optimizer, observations, actions, rewards, next_observations, dones)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to tensorboard
        if loss_components:
            total_loss, value_loss, reward_loss, policy_loss = loss_components
            writer.add_scalar('Loss/total', total_loss, epoch)
            writer.add_scalar('Loss/value', value_loss, epoch)
            writer.add_scalar('Loss/reward', reward_loss, epoch)
            writer.add_scalar('Loss/policy', policy_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Calculate training time
        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('Time/epoch_seconds', epoch_duration, epoch)
        
        # Print progress
        if epoch % 10 == 0:
            if loss_components:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss:.4f}, Value: {value_loss:.4f}, "
                      f"Reward: {reward_loss:.4f}, Policy: {policy_loss:.4f}, "
                      f"Time: {epoch_duration:.2f}s, LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch}/{num_epochs}, Time: {epoch_duration:.2f}s, LR: {current_lr:.2e}")
        
        # Save model checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            save_path = f"{checkpoint_path}_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model
    save_path = f"{checkpoint_path}_final.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"Final model saved to {save_path}")
    
    # Close collector
    collector.close()
    writer.close()
    
    return model


def train_on_batch(model, optimizer, observations, actions, rewards, next_observations, dones):
    """Train model on a batch of data using large batch processing for MPS efficiency."""
    # Enable training mode
    model.train()
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Get batch size
    batch_size = observations.shape[0]
    
    # Ensure inputs are bfloat16
    observations = observations.to(torch.bfloat16)
    rewards = rewards.to(torch.bfloat16)
    
    # Get initial hidden state and predictions
    hidden, value_pred, policy_logits = model.initial_inference(observations)
    
    # Create one-hot encoding for actions
    action_one_hot = torch.zeros(batch_size, model.action_dim, device=actions.device, dtype=torch.bfloat16)
    for i in range(batch_size):
        action_one_hot[i, actions[i]] = 1.0
    
    # Get recurrent predictions
    next_hidden, reward_pred, next_value_pred, next_policy_logits = model.recurrent_inference(hidden, action_one_hot)
    
    # Calculate value loss
    value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(-1), rewards)
    next_value_loss = torch.nn.functional.mse_loss(next_value_pred.squeeze(-1), torch.zeros_like(rewards))
    combined_value_loss = (value_loss + next_value_loss) / 2.0
    
    # Calculate reward loss
    reward_loss = torch.nn.functional.mse_loss(reward_pred.squeeze(-1), rewards)
    
    # Calculate policy loss
    # For this example, we'll use a simple uniform policy target
    # In a real implementation, you would use the MCTS policy as the target
    uniform_policy = torch.ones_like(policy_logits) / model.action_dim
    policy_loss = torch.nn.functional.cross_entropy(policy_logits, uniform_policy)
    
    # Calculate total loss
    total_loss = combined_value_loss + reward_loss + policy_loss
    
    # Backpropagate
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Return loss components
    return total_loss.item(), combined_value_loss.item(), reward_loss.item(), policy_loss.item()


def optimized_self_play(
    network: MuZeroNetwork,
    num_games: int = 1,
    num_simulations: int = 50,
    mcts_batch_size: int = 16,
    env_batch_size: int = 16,
    temperature: float = 1.0,
    temperature_drop: int = 10,
    dirichlet_alpha: float = 0.3,
    exploration_fraction: float = 0.25,
    device: str = "cpu"
):
    """
    Optimized self-play function that uses batched MCTS and vectorized environments.
    
    Args:
        network: MuZero network for inference
        num_games: Number of games to play
        num_simulations: Number of MCTS simulations per move
        mcts_batch_size: Batch size for MCTS simulations
        env_batch_size: Number of parallel environments
        temperature: Temperature for action selection
        temperature_drop: Move number to drop temperature to 0
        dirichlet_alpha: Dirichlet noise alpha parameter
        exploration_fraction: Fraction of exploration noise to add
        device: Device to run on ("cpu", "cuda", "mps")
        
    Returns:
        List of game histories
    """
    # Ensure network is in eval mode for inference
    network.eval()
    
    # Create the batched MCTS
    mcts = BatchedMCTS(
        network=network,
        num_simulations=num_simulations,
        batch_size=mcts_batch_size,
        discount=0.997,
        dirichlet_alpha=dirichlet_alpha,
        exploration_fraction=exploration_fraction,
        action_space_size=576,  # 24*24 for Narde
        device=device
    )
    
    # Create vectorized environment - use min of num_games and env_batch_size
    actual_env_batch_size = min(num_games, env_batch_size)
    vec_env = create_vectorized_env(num_envs=actual_env_batch_size, device=device)
    
    # Initialize game histories
    game_histories = [[] for _ in range(num_games)]
    completed_games = 0
    active_game_indices = list(range(min(num_games, actual_env_batch_size)))
    move_counts = [0] * len(active_game_indices)
    
    # Get initial observations
    observations = vec_env.reset()
    
    # Main self-play loop - until all games are completed
    while completed_games < num_games:
        # Get valid actions for each active environment
        valid_actions_batch = vec_env.get_valid_actions_batch()
        
        # For each active environment
        actions = []
        for i, env_idx in enumerate(vec_env.active_envs):
            if i >= len(active_game_indices):
                continue  # Skip if we don't have a game associated with this environment
                
            game_idx = active_game_indices[i]
            valid_actions = valid_actions_batch[i]
            obs = observations[vec_env.active_envs[i]]
            
            # Skip if no valid moves
            if not valid_actions:
                actions.append((0, 0))  # Dummy action that will be skipped
                continue
                
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # Use temperature based on move count
            current_temperature = 0.0 if move_counts[i] >= temperature_drop else temperature
            
            # Run MCTS to get policy
            policy = mcts.run(
                observation=obs_tensor,
                valid_actions=valid_actions,
                add_exploration_noise=True
            )
            
            # Select action based on policy and temperature
            if current_temperature == 0:
                # Deterministic selection (argmax)
                action_idx = np.argmax([policy[a] for a in valid_actions])
                action = valid_actions[action_idx]
            else:
                # Sample with temperature
                policy_subset = np.array([policy[a] for a in valid_actions])
                policy_subset = policy_subset ** (1 / current_temperature)
                policy_subset = policy_subset / policy_subset.sum()
                action_idx = np.random.choice(len(valid_actions), p=policy_subset)
                action = valid_actions[action_idx]
            
            # Determine move type (regular or bear-off)
            move_type = 0  # Default to regular move
            from_pos = action // 24
            
            # Check if this move should be a bear-off
            unwrapped_env = vec_env.envs[vec_env.active_envs[i]].unwrapped
            for move in unwrapped_env.game.get_valid_moves():
                if move[0] == from_pos and move[1] == 'off':
                    move_type = 1
                    break
            
            # Add action to list
            actions.append((action, move_type))
            
            # Store the observation, action and policy in game history
            game_histories[game_idx].append((obs, action, 0, policy))  # Reward will be updated later
            move_counts[i] += 1
        
        # Execute actions in environments
        next_obs, rewards, dones, infos = vec_env.step(actions)
        
        # Update rewards in game histories
        for i, env_idx in enumerate(vec_env.active_envs):
            if i >= len(active_game_indices):
                continue
            
            game_idx = active_game_indices[i]
            if len(game_histories[game_idx]) > 0:
                # Add current reward to the last transition
                last_transition = game_histories[game_idx][-1]
                game_histories[game_idx][-1] = (last_transition[0], last_transition[1], rewards[env_idx], last_transition[3])
        
        # Handle completed games and start new ones if needed
        for i, is_done in enumerate(dones):
            if is_done and i < len(active_game_indices):
                # Mark this game as completed
                completed_games += 1
                
                # If we have more games to play, start a new one in this environment
                if completed_games < num_games:
                    # Reset the move count
                    move_counts[i] = 0
                    
                    # Assign a new game index to this environment
                    active_game_indices[i] = completed_games
                else:
                    # Remove this environment from active list since we're done with all games
                    active_game_indices[i] = -1  # Mark as inactive
        
        # Clean up the active_game_indices list (remove -1 entries)
        active_game_indices = [idx for idx in active_game_indices if idx >= 0]
        
        # Update observations for next iteration
        observations = next_obs
    
    # Close the vectorized environment
    vec_env.close()
    
    return game_histories


def optimized_training_epoch(
    network: MuZeroNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int = 128,
    device: str = "cpu"
):
    """
    Optimized training epoch that efficiently processes batches.
    
    Args:
        network: MuZero network to train
        optimizer: Optimizer for training
        replay_buffer: Replay buffer with stored transitions
        batch_size: Batch size for training
        device: Device to run on ("cpu", "cuda", "mps")
        
    Returns:
        Dictionary with loss metrics
    """
    # Ensure network is in train mode
    network.train()
    
    # Sample batch from replay buffer
    batch = replay_buffer.sample_batch(batch_size, device=device)
    
    # Process batch data
    observations, actions, target_values, target_policies, bootstrap_positions = batch
    
    # Convert to tensors if not already
    if not isinstance(observations, torch.Tensor):
        observations = torch.FloatTensor(observations).to(device)
    if not isinstance(actions, torch.Tensor):
        actions = torch.LongTensor(actions).to(device)
    if not isinstance(target_values, torch.Tensor):
        target_values = torch.FloatTensor(target_values).to(device)
    if not isinstance(target_policies, torch.Tensor):
        target_policies = torch.FloatTensor(target_policies).to(device)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Initial inference - representation and prediction networks
    with torch.no_grad():
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
    
    # Convert actions to one-hot for dynamics network - single efficient operation
    action_onehots = torch.zeros(actions.size(0), network.action_dim, device=device)
    action_onehots.scatter_(1, actions.unsqueeze(1), 1.0)
    
    # Dynamics network inference with gradient
    next_hidden_states, predicted_rewards = network.dynamics_network(hidden_states, action_onehots)
    
    # Reward loss - scale rewards appropriately
    reward_loss = torch.nn.functional.mse_loss(predicted_rewards.squeeze(-1), target_values / 10.0)
    
    # Total loss
    total_loss = value_loss + policy_loss + reward_loss
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Return losses
    return {
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item(),
        'reward_loss': reward_loss.item(),
        'total_loss': total_loss.item()
    }


def train_muzero(
    network: MuZeroNetwork,
    num_epochs: int = 100,
    games_per_epoch: int = 10,
    num_simulations: int = 50,
    mcts_batch_size: int = 16,
    env_batch_size: int = 8,
    training_batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    checkpoint_dir: str = "checkpoints"
):
    """
    Train MuZero with optimized implementation.
    
    Args:
        network: MuZero network to train
        num_epochs: Number of training epochs
        games_per_epoch: Number of self-play games per epoch
        num_simulations: Number of MCTS simulations per move
        mcts_batch_size: Batch size for MCTS simulations
        env_batch_size: Number of parallel environments for self-play
        training_batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to run on ("cpu", "cuda", "mps")
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Training metrics history
    """
    # Create optimizer
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training metrics history
    metrics_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Self-play phase
        self_play_start_time = time.time()
        game_histories = optimized_self_play(
            network=network,
            num_games=games_per_epoch,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
            env_batch_size=env_batch_size,
            device=device
        )
        self_play_time = time.time() - self_play_start_time
        
        # Add game histories to replay buffer
        for game_history in game_histories:
            replay_buffer.save_game(game_history)
        
        # Training phase
        training_start_time = time.time()
        epoch_metrics = {
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'reward_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Only train if we have enough samples
        if len(replay_buffer) >= training_batch_size:
            # Perform multiple training updates
            num_updates = max(1, min(len(replay_buffer) // training_batch_size, 10))
            
            for _ in range(num_updates):
                update_metrics = optimized_training_epoch(
                    network=network,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=training_batch_size,
                    device=device
                )
                
                # Accumulate metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += update_metrics[key] / num_updates
        
        training_time = time.time() - training_start_time
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"muzero_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': epoch_metrics
        }, checkpoint_path)
        
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        
        # Log progress
        print(f"Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f"  Self-play: {self_play_time:.2f}s - Training: {training_time:.2f}s")
        print(f"  Value loss: {epoch_metrics['value_loss']:.4f}, Policy loss: {epoch_metrics['policy_loss']:.4f}, Reward loss: {epoch_metrics['reward_loss']:.4f}")
        
        # Save metrics
        epoch_metrics.update({
            'epoch': epoch,
            'epoch_time': epoch_time,
            'self_play_time': self_play_time,
            'training_time': training_time
        })
        metrics_history.append(epoch_metrics)
    
    return metrics_history


def efficient_evaluate(
    network: MuZeroNetwork,
    num_games: int = 100,
    num_simulations: int = 50,
    mcts_batch_size: int = 16,
    env_batch_size: int = 16,
    opponent: str = "random",
    device: str = "cpu"
):
    """
    Efficiently evaluate MuZero against a specified opponent.
    
    Args:
        network: MuZero network for evaluation
        num_games: Number of games to play
        num_simulations: Number of MCTS simulations per move
        mcts_batch_size: Batch size for MCTS simulations
        env_batch_size: Number of parallel environments
        opponent: Opponent type ("random", "mcts", etc.)
        device: Device to run on ("cpu", "cuda", "mps")
        
    Returns:
        Evaluation metrics dictionary
    """
    # Set network to eval mode
    network.eval()
    
    # Create vectorized environment
    vec_env = create_vectorized_env(num_envs=env_batch_size, device=device)
    
    # Create batched MCTS
    mcts = BatchedMCTS(
        network=network,
        num_simulations=num_simulations,
        batch_size=mcts_batch_size,
        action_space_size=576,
        device=device
    )
    
    # Tracking metrics
    muzero_wins = 0
    opponent_wins = 0
    draws = 0
    game_lengths = []
    completed_games = 0
    
    # Get initial observations
    observations = vec_env.reset()
    
    # Track current player (0 = MuZero, 1 = opponent)
    current_players = [0] * vec_env.num_envs
    move_counts = [0] * vec_env.num_envs
    
    # Main evaluation loop
    while completed_games < num_games:
        # Get valid actions for each active environment
        valid_actions_batch = vec_env.get_valid_actions_batch()
        
        # For each active environment
        actions = []
        for i, env_idx in enumerate(vec_env.active_envs):
            valid_actions = valid_actions_batch[i]
            obs = observations[vec_env.active_envs[i]]
            
            # Skip if no valid moves
            if not valid_actions:
                actions.append((0, 0))  # Dummy action that will be skipped
                continue
            
            # Determine whose turn it is
            is_muzero_turn = current_players[i] == 0
            
            if is_muzero_turn:
                # MuZero's turn - use MCTS
                obs_tensor = torch.FloatTensor(obs).to(device)
                
                # Run MCTS
                policy = mcts.run(
                    observation=obs_tensor,
                    valid_actions=valid_actions,
                    add_exploration_noise=False  # No exploration during evaluation
                )
                
                # Select best action
                action_idx = np.argmax([policy[a] for a in valid_actions])
                action = valid_actions[action_idx]
            else:
                # Opponent's turn
                if opponent == "random":
                    # Random opponent
                    action = np.random.choice(valid_actions)
                elif opponent == "mcts":
                    # MCTS opponent (weaker version with fewer simulations)
                    obs_tensor = torch.FloatTensor(obs).to(device)
                    policy = run_batched_mcts(
                        observation=obs_tensor,
                        network=network,
                        valid_actions=valid_actions,
                        num_simulations=num_simulations // 2,  # Fewer simulations for opponent
                        device=device,
                        batch_size=mcts_batch_size
                    )
                    action_idx = np.argmax([policy[a] for a in valid_actions])
                    action = valid_actions[action_idx]
            
            # Determine move type (regular or bear-off)
            move_type = 0  # Default to regular move
            from_pos = action // 24
            
            # Check if this move should be a bear-off
            unwrapped_env = vec_env.envs[vec_env.active_envs[i]].unwrapped
            for move in unwrapped_env.game.get_valid_moves():
                if move[0] == from_pos and move[1] == 'off':
                    move_type = 1
                    break
            
            # Add action to list
            actions.append((action, move_type))
            
            # Update move count and current player
            move_counts[i] += 1
            current_players[i] = 1 - current_players[i]  # Switch player
        
        # Execute actions
        next_obs, rewards, dones, infos = vec_env.step(actions)
        
        # Handle completed games
        for i, is_done in enumerate(dones):
            if is_done and i < env_batch_size:
                env_idx = vec_env.active_envs[i] if i < len(vec_env.active_envs) else i
                
                # Determine winner
                reward = rewards[env_idx]
                if reward > 0:
                    # Positive reward means player 1 (MuZero if it was player 1) won
                    if current_players[i] == 1:  # If next player is opponent, MuZero won
                        muzero_wins += 1
                    else:
                        opponent_wins += 1
                elif reward < 0:
                    # Negative reward means player 2 (opponent if MuZero was player 1) won
                    if current_players[i] == 1:  # If next player is opponent, opponent won
                        opponent_wins += 1
                    else:
                        muzero_wins += 1
                else:
                    # Zero reward means draw
                    draws += 1
                
                # Record game length
                game_lengths.append(move_counts[i])
                
                # Reset for next game
                move_counts[i] = 0
                current_players[i] = 0  # MuZero always starts
                
                # Update completed games count
                completed_games += 1
        
        # Update observations
        observations = next_obs
    
    # Close environment
    vec_env.close()
    
    # Calculate win rate
    win_rate = muzero_wins / num_games if num_games > 0 else 0
    
    # Return evaluation metrics
    return {
        'num_games': num_games,
        'muzero_wins': muzero_wins,
        'opponent_wins': opponent_wins,
        'draws': draws,
        'win_rate': win_rate,
        'avg_game_length': sum(game_lengths) / len(game_lengths) if game_lengths else 0
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MuZero with optimized batch processing")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--buffer-size", type=int, default=8192, help="Size of experience buffer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--checkpoint", type=str, default="muzero_model", help="Checkpoint path")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--parallel-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, or cpu)")
    parser.add_argument("--profiling", action="store_true", help="Enable PyTorch profiling")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval in epochs")
    
    args = parser.parse_args()
    
    train_muzero_optimized(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
        num_parallel_envs=args.parallel_envs,
        device_str=args.device,
        enable_profiling=args.profiling,
        save_interval=args.save_interval,
    ) 