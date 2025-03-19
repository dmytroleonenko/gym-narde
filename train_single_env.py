#!/usr/bin/env python3
"""
Simplified training script for Narde game using a single environment.
This script is focused on debugging and tracking training progress.
"""

import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from tqdm import tqdm

# Make sure gym_narde is registered
import gym_narde
from narde_rl.networks.decomposed_dqn import DecomposedDQN

# Force MPS device if available
if torch.backends.mps.is_available():
    print("MPS device found - using MPS")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA device found - using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

class SimpleReplayBuffer:
    """Simple replay buffer for storing experience tuples."""
    
    def __init__(self, buffer_size=10000, batch_size=64, device=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cpu")
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """Sample a batch of experiences from the buffer."""
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class SimpleDQNAgent:
    """Simple DQN agent for the Narde game."""
    
    def __init__(self, state_size, action_size, lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        
        # Create Q-network model - DecomposedDQN handles device internally
        self.model = DecomposedDQN(state_size, action_size)
        self.device = self.model.device  # Get the device from the model
        
        # Create memory buffer with the device
        self.memory = SimpleReplayBuffer(buffer_size=10000, batch_size=64, device=self.device)
        
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Create target network
        self.target_model = DecomposedDQN(state_size, action_size)
        self.update_target_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training metrics
        self.loss_history = []
        
    def update_target_model(self):
        """Copy weights from model to target model."""
        self.target_model.network.load_state_dict(self.model.network.state_dict())
    
    def act(self, state, valid_moves=None, training=True):
        """Select an action based on the current state."""
        # Epsilon-greedy action selection
        if training and np.random.rand() < self.epsilon:
            # Exploration: select a random valid move
            if valid_moves and len(valid_moves) > 0:
                move = random.choice(valid_moves)
                print(f"EXPLORE: Selected random move {move} from {len(valid_moves)} valid moves")
                
                # Convert move to gym action format
                from_pos, to_pos = move
                if to_pos == 'off':
                    move_index = from_pos * 24
                    move_type = 1  # Bearing off
                else:
                    move_index = from_pos * 24 + to_pos
                    move_type = 0  # Regular move
                
                return (int(move_index), int(move_type))
            else:
                # No valid moves, return a dummy action
                return (0, 0)
        
        # Exploitation: select best action based on Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model.forward(state_tensor)
        
        # If we have valid moves, mask out invalid moves
        if valid_moves and len(valid_moves) > 0:
            # Create mask for valid actions
            valid_action_mask = torch.zeros(self.action_size, device=self.device)
            
            for move in valid_moves:
                from_pos, to_pos = move
                if to_pos == 'off':
                    move_index = from_pos * 24
                    move_type = 1
                else:
                    move_index = from_pos * 24 + to_pos 
                    move_type = 0
                
                # Calculate action index in flattened space
                action_idx = int(move_index)
                valid_action_mask[action_idx] = 1
            
            # Apply mask (set invalid actions to very negative values)
            masked_q_values = q_values.squeeze().clone()
            masked_q_values[valid_action_mask == 0] = -1e6
            
            # Get the best valid action
            action_idx = torch.argmax(masked_q_values).item()
            
            # Convert back to move format for debugging
            from_pos = action_idx // 24
            to_pos = action_idx % 24
            
            # Find the corresponding valid move
            for move in valid_moves:
                move_from, move_to = move
                if move_from == from_pos and (move_to == to_pos or (move_to == 'off' and to_pos == 0)):
                    print(f"EXPLOIT: Selected best move {move} with Q-value {masked_q_values[action_idx].item():.4f}")
                    break
            
            return (action_idx, 0)  # Default to regular move type
        
        # If no valid moves, return dummy action
        return (0, 0)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        # Convert action tuple to single index for storage
        action_idx = action[0]  # Use only the move_index part
        self.memory.add(state, action_idx, reward, next_state, done)
    
    def replay(self, batch_size=None):
        """Train the model on a batch of experiences."""
        if batch_size is None:
            batch_size = self.memory.batch_size
        
        if len(self.memory) < batch_size:
            return 0  # Not enough samples
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Compute target Q values
        with torch.no_grad():
            target_q_values = self.target_model.forward(next_states)
            max_target_q = target_q_values.max(1)[0]
            targets = rewards + self.gamma * max_target_q * (1 - dones)
        
        # Compute current Q values
        self.optimizer.zero_grad()
        q_values = self.model.forward(states)
        
        # Only update the Q values for the actions taken
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss and backpropagate
        loss = self.criterion(q_value, targets)
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        self.loss_history.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

def train(agent, env, episodes=100, max_steps=500, debug=False, update_freq=10):
    """Train the agent on the environment."""
    rewards_history = []
    steps_history = []
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    start_time = time.time()
    
    print(f"Starting training for {episodes} episodes with max {max_steps} steps per episode")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    
    for episode in tqdm(range(episodes), desc="Training Progress"):
        # Reset environment
        state, info = env.reset()
        total_reward = 0
        step_count = 0
        
        if debug:
            print(f"\nEpisode {episode+1}/{episodes}")
            print(f"Initial dice: {env.unwrapped.dice}")
        
        # Main episode loop
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env.unwrapped.game.get_valid_moves()
            
            if debug and step < 5:
                print(f"\nStep {step}, Valid moves: {valid_moves}")
            
            # Get action from agent
            action = agent.act(state, valid_moves, training=True)
            
            if debug and step < 5:
                print(f"Selected action: {action}")
            
            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            if debug and step < 5:
                print(f"Reward: {reward}, Done: {done}")
                print(f"Dice after move: {env.unwrapped.dice}")
            
            # Store experience in agent's memory
            agent.remember(state, action, reward, next_state, done or truncated)
            
            # Update state
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Break if episode is done
            if done or truncated:
                if reward > 0:
                    win_count += 1
                elif reward < 0:
                    loss_count += 1
                else:
                    draw_count += 1
                break
        
        # Train the model after each episode
        loss = agent.replay()
        
        # Update target model periodically
        if episode % update_freq == 0:
            agent.update_target_model()
        
        # Record episode statistics
        rewards_history.append(total_reward)
        steps_history.append(step_count)
        
        # Print episode summary
        if debug or (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"\nEpisode {episode+1}: Reward={total_reward:.2f}, Steps={step_count}")
            print(f"Epsilon: {agent.epsilon:.4f}, Loss: {loss:.6f}")
            print(f"Time elapsed: {elapsed_time:.1f}s")
            print(f"Win/Loss/Draw: {win_count}/{loss_count}/{draw_count}")
    
    # Training complete
    print("\nTraining complete!")
    print(f"Win/Loss/Draw: {win_count}/{loss_count}/{draw_count}")
    
    # Return training statistics
    return {
        'rewards': rewards_history,
        'steps': steps_history,
        'win_count': win_count,
        'loss_count': loss_count,
        'draw_count': draw_count,
        'loss_history': agent.loss_history
    }

def plot_results(stats, save_path=None):
    """Plot training results."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(stats['rewards'])
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(stats['steps'])
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    # Plot loss
    ax3.plot(stats['loss_history'])
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Train a DQN agent on Narde game')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--model-path', type=str, default='saved_models/simple_dqn_narde.pt', 
                        help='Path to save the trained model')
    parser.add_argument('--plot-path', type=str, default='saved_models/simple_dqn_training.png',
                       help='Path to save the training plot')
    
    args = parser.parse_args()
    
    # Create directory for saved models
    os.makedirs('saved_models', exist_ok=True)
    
    # Initialize environment
    env = gym.make('Narde-v0', debug=args.debug)
    
    # Define state and action dimensions
    state_dim = 28  # Observation shape
    action_dim = 576  # Maximum possible actions (24*24)
    
    # Create DQN agent
    agent = SimpleDQNAgent(state_dim, action_dim, lr=args.lr)
    agent.epsilon_decay = args.epsilon_decay
    
    print(f"Agent created with learning rate: {agent.learning_rate}")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print(f"Using device: {agent.device}")
    
    try:
        # Train the agent
        stats = train(agent, env, episodes=args.episodes, max_steps=args.max_steps, 
                      debug=args.debug, update_freq=10)
        
        # Save the trained model
        torch.save(agent.model.network.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
        
        # Plot and save results
        plot_results(stats, save_path=args.plot_path)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
        # Save interrupted model
        interrupted_path = args.model_path.replace('.pt', '_interrupted.pt')
        torch.save(agent.model.network.state_dict(), interrupted_path)
        print(f"Interrupted model saved to {interrupted_path}")
    
    finally:
        env.close()

if __name__ == "__main__":
    main() 