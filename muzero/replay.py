import numpy as np
import torch
import random
from collections import deque
from typing import List, Tuple, Dict, Union, Any

class ReplayBuffer:
    """
    Replay buffer for MuZero training.
    Stores game trajectories and samples them for training.
    """
    def __init__(self, capacity: int, n_step_return: int = 5, discount: float = 0.997):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.n_step_return = n_step_return
        self.discount = discount
        self.terminal_buffer = deque(maxlen=min(capacity // 4, 1000))  # Store terminal states separately
        self.terminal_priorities = deque(maxlen=min(capacity // 4, 1000))
        
        # Pre-allocate memory for faster sampling
        self.last_device = "cpu"
        self.pinned_memory = False
        
    def save_game(self, game_history):
        """
        Save a complete game history to the buffer.
        
        Args:
            game_history: A list of (observation, action, reward, policy) tuples
        """
        # Pre-process game data for faster sampling
        processed_game = []
        for obs, action, reward, policy in game_history:
            # Convert observations and policies to numpy arrays with correct dtype
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.bfloat16)
            if not isinstance(policy, np.ndarray):
                policy = np.array(policy, dtype=np.bfloat16)
            
            processed_game.append((obs, action, reward, policy))
        
        # Add priority based on rewards and terminal state
        priority = 1.0
        
        if len(processed_game) > 0:
            # Increase priority for games with higher rewards
            total_reward = sum(transition[2] for transition in processed_game)
            priority += min(2.0, abs(total_reward) / 2.0)
            
            # Check if it's a terminal state (game over)
            is_terminal = processed_game[-1][2] != 0  # Non-zero reward at the end typically indicates game over
            
            if is_terminal:
                self.terminal_buffer.append(processed_game)
                self.terminal_priorities.append(priority * 2.0)  # Higher priority for terminal states
            else:
                self.buffer.append(processed_game)
                self.priorities.append(priority)
    
    def sample_game(self):
        """
        Sample a game from the buffer based on priorities.
        
        Returns:
            A complete game history
        """
        # Decide whether to sample from terminal buffer or regular buffer
        terminal_prob = 0.3  # 30% chance to sample from terminal buffer
        sample_terminal = (random.random() < terminal_prob) and len(self.terminal_buffer) > 0
        
        if sample_terminal:
            # Sample from terminal buffer with prioritization
            if len(self.terminal_priorities) == 0:
                return None
            probs = np.array(self.terminal_priorities) / sum(self.terminal_priorities)
            idx = np.random.choice(len(self.terminal_buffer), p=probs)
            return self.terminal_buffer[idx]
        else:
            # Sample from regular buffer with prioritization
            if len(self.buffer) == 0:
                if len(self.terminal_buffer) > 0:
                    # Fall back to terminal buffer if regular buffer is empty
                    if len(self.terminal_priorities) == 0:
                        return None
                    probs = np.array(self.terminal_priorities) / sum(self.terminal_priorities)
                    idx = np.random.choice(len(self.terminal_buffer), p=probs)
                    return self.terminal_buffer[idx]
                else:
                    # If both buffers are empty, return None
                    return None
            
            if len(self.priorities) == 0:
                return None
            probs = np.array(self.priorities) / sum(self.priorities)
            idx = np.random.choice(len(self.buffer), p=probs)
            return self.buffer[idx]
    
    def sample_position(self, game_history):
        """
        Sample a position from a game.
        
        Args:
            game_history: A complete game history
            
        Returns:
            A position index
        """
        # Give higher weight to more recent positions
        if len(game_history) <= 1:
            return 0
            
        position_weights = np.linspace(1.0, 2.0, len(game_history))
        position_probs = position_weights / position_weights.sum()
        position_idx = np.random.choice(len(game_history), p=position_probs)
        return position_idx
    
    def sample_n_step_transition(self, game_history, position_idx):
        """
        Sample an n-step transition from a position in a game.
        
        Args:
            game_history: A complete game history
            position_idx: The starting position index
            
        Returns:
            An n-step transition (current_obs, action, target_value, target_policy, bootstrap_position)
        """
        bootstrap_position = min(position_idx + self.n_step_return, len(game_history) - 1)
        transition = game_history[position_idx]
        
        # Calculate the n-step return
        n_step_return = 0
        for i in range(position_idx, bootstrap_position):
            reward = game_history[i][2]
            n_step_return += (self.discount ** (i - position_idx)) * reward
            
        # Add the bootstrapped value if not at the end of the game
        if bootstrap_position < len(game_history) - 1:
            # This would be the actual bootstrapped value, but in MuZero we predict it
            # Using 0 as a placeholder
            bootstrap_value = 0  # This will be predicted by the model
            n_step_return += (self.discount ** (bootstrap_position - position_idx)) * bootstrap_value
            
        # The current observation, action, target value, and MCTS policy
        current_obs = transition[0]
        action = transition[1]
        target_policy = transition[3]
        
        # Return the transition with n-step return
        return (current_obs, action, n_step_return, target_policy, bootstrap_position - position_idx)
    
    def enable_cuda_pinned_memory(self):
        """
        Enable CUDA pinned memory for faster GPU transfers.
        Only call this when using GPU training.
        """
        self.pinned_memory = True
    
    def sample_batch(self, batch_size: int, device: str = "cpu"):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: The number of transitions to sample
            device: The device to put the tensors on
            
        Returns:
            A batch of transitions
        """
        # Save device for future optimizations
        self.last_device = device
        
        # Check if we need pinned memory
        # Properly handle both string and torch.device objects
        use_cuda = False
        if isinstance(device, str):
            use_cuda = device.startswith('cuda')
        elif isinstance(device, torch.device):
            use_cuda = device.type == 'cuda'
        
        if use_cuda and not self.pinned_memory:
            self.enable_cuda_pinned_memory()
            
        games = []
        game_positions = []
        
        # Sample games and positions
        for _ in range(batch_size):
            game = self.sample_game()
            if game is None:
                # If buffer is empty, return None
                return None
                
            position = self.sample_position(game)
            games.append(game)
            game_positions.append(position)
        
        # Extract transitions more efficiently - preallocate memory when possible
        batch = [
            self.sample_n_step_transition(game, position)
            for game, position in zip(games, game_positions)
        ]
        
        # Check if all observations have the same shape to use stack
        same_shape_obs = all(batch[0][0].shape == transition[0].shape for transition in batch)
        same_shape_policy = all(batch[0][3].shape == transition[3].shape for transition in batch)
        
        # Prepare batch for training with optimized tensor creation
        if same_shape_obs:
            # Use np.stack for observations (faster for uniform-shaped arrays)
            observations = np.stack([transition[0] for transition in batch]).astype(np.bfloat16)
        else:
            # Fall back to array if shapes are different
            observations = np.array([transition[0] for transition in batch], dtype=np.bfloat16)
        
        # Actions and bootstrap positions are simple integers, directly convert to arrays
        actions = np.array([transition[1] for transition in batch], dtype=np.int64)
        target_values = np.array([transition[2] for transition in batch], dtype=np.bfloat16)
        
        if same_shape_policy:
            # Use np.stack for policies (faster for uniform-shaped arrays)
            target_policies = np.stack([transition[3] for transition in batch]).astype(np.bfloat16)
        else:
            # Fall back to array if shapes are different
            target_policies = np.array([transition[3] for transition in batch], dtype=np.bfloat16)
            
        bootstrap_positions = np.array([transition[4] for transition in batch], dtype=np.int64)
        
        # Create tensors with optimal memory handling
        if self.pinned_memory and use_cuda:
            # Use pinned memory for faster GPU transfer
            observations_tensor = torch.as_tensor(observations, device='cpu').pin_memory().to(device, non_blocking=True)
            actions_tensor = torch.as_tensor(actions, device='cpu').pin_memory().to(device, non_blocking=True)
            target_values_tensor = torch.as_tensor(target_values, device='cpu').pin_memory().to(device, non_blocking=True)
            target_policies_tensor = torch.as_tensor(target_policies, device='cpu').pin_memory().to(device, non_blocking=True)
            bootstrap_positions_tensor = torch.as_tensor(bootstrap_positions, device='cpu').pin_memory().to(device, non_blocking=True)
        else:
            # Direct transfer for CPU or when pinned memory is not enabled
            observations_tensor = torch.as_tensor(observations, device=device)
            actions_tensor = torch.as_tensor(actions, device=device)
            target_values_tensor = torch.as_tensor(target_values, device=device)
            target_policies_tensor = torch.as_tensor(target_policies, device=device)
            bootstrap_positions_tensor = torch.as_tensor(bootstrap_positions, device=device)
        
        return observations_tensor, actions_tensor, target_values_tensor, target_policies_tensor, bootstrap_positions_tensor
    
    def __len__(self):
        return len(self.buffer) + len(self.terminal_buffer) 