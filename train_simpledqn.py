#!/usr/bin/env python3
"""
Simple DQN training script for the Narde environment.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import argparse
import time
import os
import json

# Check if MPS is available (for Mac)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define the DQN model with layer normalization for stability
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle single state case for target computation
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

# Define the ReplayBuffer with prioritization for bear-off moves
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.terminal_buffer = deque(maxlen=min(capacity // 4, 1000))  # Store terminal states separately
        self.terminal_priorities = deque(maxlen=min(capacity // 4, 1000))
    
    def push(self, state, action, reward, next_state, done, is_bear_off=False):
        experience = (state, action, reward, next_state, done)
        
        # Calculate priority based on experience type
        # Higher priority for terminal states and bear-off moves
        if done:
            self.terminal_buffer.append(experience)
            priority = 3.0  # Highest priority for terminal states
            self.terminal_priorities.append(priority)
        else:
            self.buffer.append(experience)
            # Prioritize bear-offs and higher magnitude rewards
            priority = 2.0 if is_bear_off else 1.0
            # Give slightly higher priority to experiences with larger rewards
            priority += min(0.5, abs(reward) / 2.0)
            self.priorities.append(priority)
    
    def sample(self, batch_size):
        # Reserve some slots for terminal states if available
        terminal_ratio = 0.25  # 25% of batch will be terminal states if possible
        terminal_size = min(int(batch_size * terminal_ratio), len(self.terminal_buffer))
        regular_size = batch_size - terminal_size
        
        samples = []
        
        # Sample terminal states if available
        if terminal_size > 0:
            terminal_probs = np.array(self.terminal_priorities) / sum(self.terminal_priorities)
            terminal_indices = np.random.choice(
                len(self.terminal_buffer), 
                terminal_size, 
                p=terminal_probs, 
                replace=len(self.terminal_buffer) < terminal_size
            )
            samples.extend([self.terminal_buffer[idx] for idx in terminal_indices])
        
        # Sample regular experiences
        if regular_size > 0 and len(self.buffer) > 0:
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(
                len(self.buffer), 
                regular_size, 
                p=probs, 
                replace=len(self.buffer) < regular_size
            )
            samples.extend([self.buffer[idx] for idx in indices])
        
        return samples
    
    def __len__(self):
        return len(self.buffer) + len(self.terminal_buffer)

# Create a logger for detailed gameplay logging
class GameplayLogger:
    def __init__(self, log_file="narde_gameplay.log"):
        self.log_file = log_file
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        # Clear any existing log
        with open(self.log_file, 'w') as f:
            f.write("Narde Gameplay Log\n")
            f.write("=================\n\n")
        
    def log_step(self, episode, step, board, dice, valid_moves, selected_move=None, action=None, reward=None, no_moves_reason=None, current_player=None):
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode}, Step {step}\n")
            f.write(f"Board: {board.tolist()}\n")
            f.write(f"Dice: {dice}\n")
            f.write(f"Valid Moves: {valid_moves}\n")
            
            if no_moves_reason:
                f.write(f"No Valid Moves: {no_moves_reason}\n")
            
            if selected_move is not None:
                f.write(f"Selected Move: {selected_move}\n")
            
            if action is not None:
                f.write(f"Action: {action}\n")
            
            if reward is not None:
                f.write(f"Reward: {reward}\n")
            
            if current_player is not None:
                f.write(f"Current Player: {current_player}\n")
            
            f.write("-" * 60 + "\n\n")
    
    def log_skip_turn(self, episode, step, reason="No valid moves"):
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode}, Step {step}\n")
            f.write(f"TURN SKIPPED: {reason}\n")
            f.write("-" * 60 + "\n\n")
    
    def log_episode_end(self, episode, reason, total_steps, total_reward):
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode} ended after {total_steps} steps.\n")
            f.write(f"Reason: {reason}\n")
            f.write(f"Total Reward: {total_reward}\n")
            f.write("=" * 60 + "\n\n")

# Helper function to compute n-step returns
def compute_n_step_return(n_step_buffer, gamma, next_q):
    R = 0.0
    for i, (_, _, r, _, _) in enumerate(n_step_buffer):
        R += (gamma ** i) * r
    R += (gamma ** len(n_step_buffer)) * next_q
    return R

# Soft update for target network
def soft_update(target_net, policy_net, tau=0.005):
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def train_dqn(env, episodes=100, max_steps=600, batch_size=128, gamma=0.99, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
              lr=0.001, buffer_size=10000, target_update=10, debug_log=False,
              n_steps=3, use_soft_update=True, tau=0.005, disable_shaping=False):
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space[0].n * env.action_space[1].n  # Combined action space size
    
    # Initialize networks
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Setup optimizer and replay buffer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Setup gameplay logger if debug_log is True
    logger = GameplayLogger("narde_detailed_gameplay.log") if debug_log else None
    
    # Tracking metrics
    epsilon = epsilon_start
    episode_rewards = []
    episode_steps = []
    episode_borne_off = []
    bear_off_count = []
    white_wins = 0  # Wins when agent starts as White
    black_wins = 0  # Wins when agent starts as Black
    losses = []
    
    # Start training
    print("Starting training...")
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        episode_reward = 0
        step_count = 0
        episode_loss = []
        episode_bear_offs = 0
        
        # Track the previous board state for reward shaping
        prev_borne_off = 0
        
        # Track consecutive skips due to no valid moves
        consecutive_skips = 0
        
        # Initialize buffer for this episode's experiences, separated by player
        # Each entry: (state, action, immediate_reward, next_state, done, is_bear_off)
        white_experiences = []
        black_experiences = []
        
        # Start with White (player +1)
        current_player = 1
        
        # Track who starts the episode (always White in this implementation)
        first_player_is_white = True
        # This tracks who was the original White player for this turn
        current_original_white_player = current_player  # Initialize to current player (White/1)
        
        # Initialize n-step buffer for multi-step returns
        n_step_buffer = deque(maxlen=n_steps)
        
        for step in range(max_steps):
            # Get valid moves from the environment
            env_unwrapped = env.unwrapped
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            # Get the current dice and board state for logging
            current_dice = env_unwrapped.dice
            current_board = env_unwrapped.game.board.copy()
            
            if not valid_moves:
                consecutive_skips += 1
                
                # Determine the reason for no valid moves
                no_moves_reason = "No dice available" if not current_dice else f"No legal moves possible with dice {current_dice}"
                
                if debug_log:
                    logger.log_step(
                        episode=episode,
                        step=step,
                        board=current_board,
                        dice=current_dice,
                        valid_moves=valid_moves,
                        no_moves_reason=no_moves_reason,
                        current_player=current_player
                    )
                    logger.log_skip_turn(episode, step, no_moves_reason)
                
                # If both players have no valid moves consecutively, end the episode
                if consecutive_skips >= 2:
                    if debug_log:
                        logger.log_episode_end(
                            episode=episode,
                            reason="Both players skipped consecutively",
                            total_steps=step_count,
                            total_reward=episode_reward
                        )
                    break
                    
                # Otherwise, rotate the board for the next player and continue
                env_unwrapped.game.rotate_board_for_next_player()
                
                # Switch player identity
                current_player *= -1
                
                # Force reset of dice when rotation happens due to skipped turn
                # This is to fix the issue where unused dice from previous player are carried over
                env_unwrapped.dice = []  # Clear existing dice
                
                # Re-roll the dice for the next player using the correct method
                if hasattr(env_unwrapped, "_roll_dice"):
                    env_unwrapped._roll_dice()  # Use environment's method which handles doubles
                else:
                    # Fallback to manually rolling dice if method doesn't exist
                    d1 = random.randint(1, 6)
                    d2 = random.randint(1, 6)
                    if d1 == d2:
                        env_unwrapped.dice = [d1, d1, d1, d1]  # Double dice give four moves
                    else:
                        env_unwrapped.dice = [d1, d2]
                
                if debug_log:
                    logger.log_step(
                        episode=episode,
                        step=step,
                        board=env_unwrapped.game.board,
                        dice=env_unwrapped.dice,
                        valid_moves=[],
                        no_moves_reason=f"Dice reset after turn skip, new dice: {env_unwrapped.dice}",
                        current_player=current_player
                    )
                
                continue
            
            # Reset the consecutive skips counter because we have valid moves
            consecutive_skips = 0
            
            # Count bear-off moves for potential prioritization
            bear_off_moves = [m for m in valid_moves if m[1] == 'off']
            
            # Epsilon-greedy action selection, only choosing from valid moves
            selected_move = None
            
            if random.random() < epsilon:
                # Random valid action, but prioritize bear-offs if available with 70% probability
                if bear_off_moves and random.random() < 0.7:
                    selected_move = random.choice(bear_off_moves)
                else:
                    selected_move = random.choice(valid_moves)
                    
                from_pos, to_pos = selected_move
                
                # Convert to action format
                if to_pos == 'off':
                    action = (from_pos * 24 + 0, 1)  # bear off move
                    episode_bear_offs += 1
                    is_bear_off = True
                else:
                    action = (from_pos * 24 + to_pos, 0)  # regular move
                    is_bear_off = False
            else:
                # Greedy action among valid moves
                with torch.no_grad():
                    # Set the policy network to evaluation mode for inference
                    policy_net.eval()
                    q_values = policy_net(state)
                    policy_net.train()  # Set back to training mode
                    
                    # Mask for valid actions
                    valid_action_mask = torch.zeros(action_dim, device=device)
                    
                    for move in valid_moves:
                        from_pos, to_pos = move
                        if to_pos == 'off':
                            # Bear off move
                            action_idx = from_pos * 24 + 0  # Using 0 as placeholder for bear off
                            move_type = 1
                            valid_action_mask[action_idx * 2 + move_type] = 1
                        else:
                            # Regular move
                            action_idx = from_pos * 24 + to_pos
                            move_type = 0
                            valid_action_mask[action_idx * 2 + move_type] = 1
                    
                    # Apply mask and get best valid action
                    masked_q_values = q_values.clone()
                    # Make sure the mask has the same shape as q_values
                    valid_action_mask = valid_action_mask.reshape(q_values.shape)
                    masked_q_values[valid_action_mask == 0] = float('-inf')
                    action_index = masked_q_values.argmax().item()
                    
                    # Convert back to the tuple format
                    move_type = action_index % 2
                    move_idx = action_index // 2
                    
                    if move_type == 0:  # Regular move
                        from_pos = move_idx // 24
                        to_pos = move_idx % 24
                        action = (move_idx, move_type)
                        is_bear_off = False
                        selected_move = (from_pos, to_pos)
                    else:  # Bear off
                        from_pos = move_idx // 24
                        action = (move_idx, move_type)
                        episode_bear_offs += 1
                        is_bear_off = True
                        selected_move = (from_pos, 'off')
            
            # Log the step details if debugging is enabled
            if debug_log:
                logger.log_step(
                    episode=episode,
                    step=step,
                    board=current_board,
                    dice=current_dice,
                    valid_moves=valid_moves,
                    selected_move=selected_move,
                    action=action,
                    current_player=current_player
                )
            
            # Track the dice and board state before taking action
            dice_before = env_unwrapped.dice.copy() if env_unwrapped.dice else []
            board_before = env_unwrapped.game.board.copy()
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Check if the environment automatically rotated the board inside step()
            auto_rotated = False
            # Check if "skipped_turn" or "skipped_multiple_turns" in info
            if "skipped_turn" in info or "skipped_multiple_turns" in info:
                auto_rotated = True
                # The environment automatically rotated the board
                current_player *= -1  # Switch player
                # Update the original White player tracking after rotation
                current_original_white_player *= -1
                if debug_log:
                    logger.log_step(
                        episode=episode,
                        step=step,
                        board=env_unwrapped.game.board,
                        dice=env_unwrapped.dice,
                        valid_moves=[],
                        no_moves_reason="Environment automatically rotated board due to skipped turn",
                        current_player=current_player
                    )
            
            # Check if dice were depleted and new dice rolled (dice count increased)
            elif len(dice_before) > 0 and len(dice_before) <= 2 and len(env_unwrapped.dice) >= 2 and len(env_unwrapped.dice) > len(dice_before):
                # This suggests dice were depleted and new dice were rolled
                auto_rotated = True
                current_player *= -1  # Switch player
                # Update the original White player tracking after rotation
                current_original_white_player *= -1
                if debug_log:
                    logger.log_step(
                        episode=episode,
                        step=step,
                        board=env_unwrapped.game.board,
                        dice=env_unwrapped.dice,
                        valid_moves=[],
                        no_moves_reason="Environment automatically rotated board due to dice depletion",
                        current_player=current_player
                    )
                    
            # Log board state after move
            if debug_log:
                logger.log_step(
                    episode=episode,
                    step=step,
                    board=env_unwrapped.game.board,
                    dice=env_unwrapped.dice,
                    valid_moves=[],
                    reward=reward,
                    current_player=current_player
                )
            
            # Immediate reward shaping - give additional reward for bearing off checkers
            shaped_reward = reward
            if not disable_shaping:
                current_borne_off = env_unwrapped.game.borne_off_white
                if current_borne_off > prev_borne_off:
                    # Reduce the bonus for bearing off to prevent overshadowing terminal reward
                    shaped_reward += 0.1 * (current_borne_off - prev_borne_off)
                prev_borne_off = current_borne_off
                
                # Add rewards based on the type of move
                if action[1] == 0:  # Regular move (not bear off)
                    move_idx = action[0]
                    from_pos = move_idx // 24
                    to_pos = move_idx % 24
                    
                    # Reward for moving from the head (furthest positions from home)
                    if from_pos >= 18:
                        shaped_reward += 0.02  # Reduced reward for moving from the head
                    
                    # Reward for moving toward home (decreasing position)
                    if from_pos > to_pos:
                        shaped_reward += 0.01  # Reduced reward for moving toward home
                    
                    # Stack management rewards
                    checker_counts = {}
                    for pos, count in enumerate(current_board):
                        if count > 0:  # Only count player's checkers (positive values)
                            checker_counts[pos] = count
                    
                    # Count stacks before move
                    stacks_before_move = sum(1 for pos, count in checker_counts.items() if count > 1)
                    
                    # Simulate move
                    simulated_board = current_board.copy()
                    if from_pos < len(simulated_board) and to_pos < len(simulated_board):
                        simulated_board[from_pos] -= 1
                        simulated_board[to_pos] += 1
                        
                        # Count stacks after move
                        checker_counts_after = {}
                        for pos, count in enumerate(simulated_board):
                            if count > 0:
                                checker_counts_after[pos] = count
                                
                        stacks_after_move = sum(1 for pos, count in checker_counts_after.items() if count > 1)
                        
                        # Apply stack-related rewards
                        if stacks_after_move > stacks_before_move:
                            shaped_reward -= 0.02  # Penalty for creating stacks
                        elif stacks_after_move < stacks_before_move:
                            shaped_reward += 0.02  # Reward for reducing stacks
                
                # Special reward for bearing off moves
                if action[1] == 1:  # Bear off move
                    shaped_reward += 0.03  # Reduced from 0.12 to prevent overshadowing terminal reward
            
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Normalize shaped reward to prevent Q-value explosion
            shaped_reward = max(min(shaped_reward, 1.0), -1.0)
            
            # Store experience directly in replay buffer with perspective awareness
            # Explicitly account for player perspective by adjusting the reward sign
            # For current_player == -1 (Black), we invert the reward since board is rotated
            perspective_reward = shaped_reward if current_player == 1 else -shaped_reward
            
            # Convert action to combined index for buffer storage
            combined_action = action[0] * 2 + action[1]  # Combined action index
            
            # Append the current transition to the n-step buffer
            n_step_buffer.append((state.cpu().numpy(), combined_action, perspective_reward, next_state.cpu().numpy(), done or truncated))
            
            # Process n-step returns if buffer is full or episode is done
            if len(n_step_buffer) == n_steps or done or truncated:
                if not (done or truncated):
                    # Compute Q-value for the last state using the target network
                    with torch.no_grad():
                        target_net.eval()
                        next_state_tensor = torch.FloatTensor(n_step_buffer[-1][3]).to(device)
                        next_q = target_net(next_state_tensor).max().item()
                else:
                    # If terminal state, no future reward
                    next_q = 0.0
                
                # Compute n-step return
                n_step_return = compute_n_step_return(n_step_buffer, gamma, next_q)
                
                # Store the first transition with n-step return
                s0, a0, _, _, done0 = n_step_buffer[0]
                replay_buffer.push(s0, a0, n_step_return, next_state.cpu().numpy(), done or truncated, is_bear_off)
                
                # Remove the oldest transition
                n_step_buffer.popleft()
            
            # If game is over, add a stronger terminal reward
            if done:
                # Determine winner (in original game perspective)
                winner = 1 if reward > 0 else -1
                
                # Track wins based on original player identity
                if winner == 1:  # White won
                    if current_original_white_player == 1:  # Original White is still White
                        white_wins += 1
                    else:  # Original White is now Black
                        black_wins += 1
                else:  # Black won
                    if current_original_white_player == 1:  # Original White is still White
                        black_wins += 1
                    else:  # Original White is now Black
                        white_wins += 1
                
                # Add terminal state with strong terminal reward
                # Use 3.0 as terminal reward to make it more significant than the shaping rewards
                terminal_reward = 3.0 if winner == current_player else -3.0
                
                # Store terminal state in replay buffer
                replay_buffer.push(
                    next_state.cpu().numpy(),
                    0,  # Dummy action (doesn't matter for terminal state)
                    terminal_reward,
                    next_state.cpu().numpy(),  # Same state for terminal
                    True,  # Done
                    False  # Not a bear off
                )
                
                if debug_log:
                    logger.log_episode_end(
                        episode=episode,
                        reason=f"{'White' if winner == 1 else 'Black'} won",
                        total_steps=step_count,
                        total_reward=episode_reward
                    )
                break
            
            state = next_state
            episode_reward += shaped_reward
            step_count += 1
            
            # Learn if enough samples are available
            if len(replay_buffer) > batch_size:
                # Sample from replay buffer
                transitions = replay_buffer.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
                
                for s, a, r, ns, d in transitions:
                    batch_states.append(s)
                    batch_actions.append(a)
                    batch_rewards.append(r)
                    batch_next_states.append(ns)
                    batch_dones.append(d)
                
                # Convert to tensors
                states = torch.FloatTensor(np.array(batch_states)).to(device)
                actions = torch.LongTensor(batch_actions).to(device)
                rewards = torch.FloatTensor(batch_rewards).to(device)
                next_states = torch.FloatTensor(np.array(batch_next_states)).to(device)
                dones = torch.FloatTensor(batch_dones).to(device)
                
                # Compute Q values with gradient clipping for stability
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Gradient clipping inside the Q-value computation
                with torch.no_grad():
                    # Set networks to evaluation mode for inference
                    policy_net.eval()
                    target_net.eval()
                    
                    # Get next max Q-values with Double DQN approach for stability
                    next_actions = policy_net(next_states).max(1)[1].detach()
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze().detach()
                    
                    # Clip Q-values to prevent explosion
                    next_q = torch.clamp(next_q, -10.0, 10.0)
                    
                    # Standard Q-learning target
                    target_q = rewards + gamma * next_q * (1 - dones)
                    
                    # Clip targets to reasonable range
                    target_q = torch.clamp(target_q, -10.0, 10.0)
                    
                    # Set networks back to training mode
                    policy_net.train()
                    target_net.train()
                
                # Compute loss
                loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
                episode_loss.append(loss.item())
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                
                optimizer.step()
                
                # Apply soft update to target network if enabled
                if use_soft_update:
                    soft_update(target_net, policy_net, tau)
            
            # Handle board rotation and player switching
            if not (done or truncated):
                if not env_unwrapped.dice:
                    env_unwrapped.game.rotate_board_for_next_player()
                    current_player *= -1
                    current_original_white_player *= -1
                    
                    if debug_log:
                        logger.log_step(
                            episode=episode,
                            step=step,
                            board=env_unwrapped.game.board,
                            dice=env_unwrapped.dice,
                            valid_moves=[],
                            no_moves_reason=f"Board rotated, current_player={current_player}, white_player={current_original_white_player}",
                            current_player=current_player
                        )
            
            if truncated:
                if debug_log:
                    logger.log_episode_end(
                        episode=episode,
                        reason="Max steps reached",
                        total_steps=step_count,
                        total_reward=episode_reward
                    )
                break
        
        # Update target network with hard update if soft updates are disabled
        if not use_soft_update and episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        episode_borne_off.append(env_unwrapped.game.borne_off_white)
        bear_off_count.append(episode_bear_offs)
        
        if episode_loss:
            losses.append(sum(episode_loss) / len(episode_loss))
        else:
            losses.append(0)
        
        # Print statistics every 10 episodes
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_steps = sum(episode_steps[-10:]) / 10
            avg_loss = sum(losses[-10:]) / 10
            avg_bear_offs = sum(bear_off_count[-10:]) / 10
            avg_borne_off = sum(episode_borne_off[-10:]) / 10
            
            print(f"Episode: {episode}")
            print(f"  Mean Loss: {avg_loss:.4f}")
            print(f"  Average Steps: {avg_steps:.2f}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Epsilon: {epsilon:.2f}")
            print(f"  Average Bear Offs Per Episode: {avg_bear_offs:.2f}")
            print(f"  Average Checkers Borne Off: {avg_borne_off:.2f}")
            print(f"  Win Rate (last 10): {sum(1 for r in episode_rewards[-10:] if r > 0)/10:.2f}")
            print(f"  Agent Wins as White: {white_wins}")
            print(f"  Agent Wins as Black: {black_wins}")  
            print(f"  Max Steps in Batch: {max(episode_steps[-10:])}")
            print("-" * 60)
        
        # Save model every 25 episodes
        if episode % 25 == 0:
            torch.save(policy_net.state_dict(), f"narde_dqn_model_ep{episode}.pth")
    
    # Save the final trained model
    torch.save(policy_net.state_dict(), "narde_dqn_model.pth")
    print("Training completed. Model saved as narde_dqn_model.pth")
    
    # Final statistics
    print("\nTraining Summary:")
    print(f"Total Episodes: {episodes}")
    print(f"Agent Wins as White: {white_wins}")
    print(f"Agent Wins as Black: {black_wins}")
    print(f"Win Rate as White: {white_wins/episodes:.2%}")
    print(f"Win Rate as Black: {black_wins/episodes:.2%}")
    print(f"Average Steps Per Episode: {sum(episode_steps)/episodes:.2f}")
    print(f"Maximum Steps in an Episode: {max(episode_steps)}")
    
    return policy_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Narde")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_decay", type=float, default=0.9997, help="Epsilon decay rate")
    parser.add_argument("--target_update", type=int, default=5, help="Target network update frequency")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--debug", action="store_true", help="Enable detailed gameplay logging")
    parser.add_argument("--n_steps", type=int, default=3, help="Number of steps for n-step returns")
    parser.add_argument("--use_soft_update", action="store_true", help="Use soft updates for target network")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter for target network")
    parser.add_argument("--disable_shaping", action="store_true", help="Disable reward shaping")
    args = parser.parse_args()
    
    # Set random seeds if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # Create environment
    env = gym.make('Narde-v0')
    
    # Start training
    start_time = time.time()
    trained_model = train_dqn(
        env=env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        debug_log=args.debug,
        n_steps=args.n_steps,
        use_soft_update=args.use_soft_update,
        tau=args.tau,
        disable_shaping=args.disable_shaping
    )
    elapsed_time = time.time() - start_time
    
    print(f"Total training time: {elapsed_time:.2f} seconds") 