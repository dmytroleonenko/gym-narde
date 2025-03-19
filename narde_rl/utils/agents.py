"""
Reinforcement learning agents for Narde game.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import multiprocessing as mp

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using sum tree for efficient sampling."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity  # Buffer capacity
        self.alpha = alpha  # Controls how much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = beta  # Controls importance sampling correction (0 = no correction, 1 = full correction)
        self.beta_increment = beta_increment  # Beta annealing
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority
        
        # Initialize buffer and priorities
        self.buffer = [None] * capacity
        self.priorities = np.ones(capacity, dtype=np.float32)  # Start with priority 1.0 instead of 0
        self.position = 0  # Current write position
        self.size = 0  # Current buffer size
        
    def add(self, experience, priority=None):
        """Add new experience to buffer with priority."""
        # Use max priority for new experiences
        max_priority = self.max_priority if priority is None else priority
        
        # Write to current position
        self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample batch according to priorities."""
        if self.size == 0:
            return [], [], []
            
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get experiences for sampled indices
        experiences = [self.buffer[idx] for idx in indices]
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for specified indices."""
        for idx, priority in zip(indices, priorities):
            # Add small epsilon to avoid zero priority
            self.priorities[idx] = priority + self.epsilon
            # Update max priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return self.size


class DQNAgent:
    """
    Deep Q-Network agent for reinforcement learning in Narde game.
    """
    
    def __init__(self, state_size, action_size, network_class, use_decomposed=False):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            network_class: Class to use for the Q-network (QNetwork or DecomposedDQN)
            use_decomposed: Whether to use the decomposed action space approach
        """
        self.state_size = state_size
        self.action_size = action_size
        self.use_decomposed_network = use_decomposed
        
        # Replay buffer - either prioritized or regular
        self.use_prioritized_replay = False
        self.memory = PrioritizedReplayBuffer(50000) if self.use_prioritized_replay else deque(maxlen=50000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration
        self.epsilon_decay = 0.995  # Exploration decay
        self.learning_rate = 3e-4  # Learning rate
        self.batch_size = 384  # Batch size
        self.move_space_size = 576  # 24*24 possible moves
        
        # Training settings
        self.grad_accumulation_steps = 4  # Accumulate gradients over 4 batches
        self.accumulated_steps = 0
        self.update_target_freq = 10  # Update target network more frequently
        
        # Determine device (CUDA, MPS or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            if mp.current_process().name == 'MainProcess':
                print("DQNAgent using MPS device")
        else:
            self.device = torch.device("cpu")
            
        # Initialize networks
        self.model = network_class(state_size, action_size)
        self.target_model = network_class(state_size, action_size)
        
        self.update_target_model()
        
        # Optimizer - use a lower learning rate for MPS to improve stability
        lr_adjusted = self.learning_rate
        if self.device.type == "mps":
            lr_adjusted = self.learning_rate * 0.5  # Lower learning rate for MPS
            if mp.current_process().name == 'MainProcess':
                print(f"Using adjusted learning rate for MPS: {lr_adjusted}")
            
        # Create optimizer with network parameters
        self.optimizer = optim.Adam(self.model.network.parameters(), lr=lr_adjusted)
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        self.loss_fn = nn.MSELoss(reduction='none')  # Use 'none' to apply importance sampling weights
        
        # Initialize TD error statistics
        self.last_td_error_mean = 0.0
        self.last_td_error_max = 0.0

        # Update target network every 5 steps for more frequent training
        self.train_step = 0

    def update_target_model(self):
        """Update the target model with weights from the main model"""
        self.target_model.network.load_state_dict(self.model.network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        experience = (state, action, reward, next_state, done)
        
        if self.use_prioritized_replay:
            # Add with max priority for new experiences
            self.memory.add(experience)
        else:
            # Standard experience replay
            self.memory.append(experience)

    def act(self, state, valid_moves=None, training=False, debug=False):
        """Return action based on epsilon-greedy policy with valid moves only"""
        # If no valid moves available, return skip action
        if not valid_moves or len(valid_moves) == 0:
            return (0, 0)
        
        # Handle empty state case (shouldn't happen but as a safeguard)
        if state is None or (isinstance(state, (list, np.ndarray)) and len(state) == 0):
            # Return a skip action if state is empty
            return (0, 0)
            
        # Epsilon-greedy exploration
        if training and np.random.rand() <= self.epsilon:
            # Choose a random valid move - this is already in (from_pos, to_pos) format
            selected_move = random.choice(valid_moves)
            if debug and mp.current_process().name == 'MainProcess':
                print(f"EXPLORE: Selected random move {selected_move} from {len(valid_moves)} valid moves")
            return selected_move
        
        # Exploitation - choose best action based on Q-values
        if self.use_decomposed_network:
            # Get Q-values
            q_values = self.model.get_q_values(state)
            
            # Create a mapping from moves to their Q-values
            move_q_values = {}
            for move in valid_moves:
                from_pos, to_pos = move
                
                # Calculate action_code for this move
                if to_pos == 'off':
                    action_code = from_pos * 24
                else:
                    action_code = from_pos * 24 + to_pos
                    
                # Map this move to its Q-value
                move_q_values[move] = q_values[action_code]
            
            # Find move with highest Q-value
            best_move = max(move_q_values.items(), key=lambda x: x[1])[0]
            
            if debug and mp.current_process().name == 'MainProcess':
                if np.random.random() < 0.1:  # Only print occasional debug info
                    print(f"EXPLOIT: Selected best move {best_move} with Q-value {move_q_values[best_move]:.6f}")
            
            return best_move
        else:
            # For standard network, use tensor operations for efficiency
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                self.model.network.eval()
                q_values = self.model.network(state_tensor).cpu().data.numpy()[0]
                self.model.network.train()
            
            # Filter to only valid moves
            valid_q_values = {}
            for move in valid_moves:
                from_pos, to_pos = move
                # Convert move to flat index
                if to_pos == 'off':
                    move_idx = from_pos * self.move_space_size  # Bearing off has special idx
                else:
                    move_idx = from_pos * 24 + to_pos
                valid_q_values[move] = q_values[move_idx]
                
            # Choose move with highest Q-value among valid moves
            best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
            
            if debug and mp.current_process().name == 'MainProcess':
                if np.random.random() < 0.1:  # Only print occasional debug info
                    print(f"EXPLOIT: Selected best move {best_move} with Q-value {valid_q_values[best_move]:.6f}")
            
            return best_move

    def replay(self):
        """Train the model with experiences from memory"""
        
        # Check if we have enough samples
        if self.use_prioritized_replay:
            if len(self.memory) < self.batch_size:
                return 0.0  # Return 0.0 if not enough samples
        else:
            if len(self.memory) < self.batch_size:
                return 0.0  # Return 0.0 if not enough samples
        
        # Sample from memory
        if self.use_prioritized_replay:
            # Use prioritized experience replay
            experiences, indices, weights = self.memory.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            # Use uniform sampling
            experiences = random.sample(self.memory, self.batch_size)
            weights_tensor = torch.ones(self.batch_size).to(self.device)
        
        # Unpack experiences
        batch_size = len(experiences)
        states = np.empty((batch_size, self.state_size), dtype=np.float32)
        actions = np.empty((batch_size, 2), dtype=np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        next_states = np.empty((batch_size, self.state_size), dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)
        
        for i, e in enumerate(experiences):
            states[i] = e[0]
            actions[i] = e[1]
            rewards[i] = e[2]
            next_states[i] = e[3]
            dones[i] = e[4]
        
        # Convert to tensors in one batch operation
        states = torch.from_numpy(states).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        # Extract action components
        move1_indices = torch.from_numpy(actions[:, 0]).to(self.device)
        move2_indices = torch.from_numpy(actions[:, 1]).to(self.device)
        
        # Initialize loss variable
        loss_value = 0.0
        
        # For decomposed network, we need to handle the training differently
        if self.use_decomposed_network:
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # STEP 1: Update move1 network
            # Get current Q-values for first moves - keep on device
            current_move1_q_values = self.model.network(states)
            move1_q_values = current_move1_q_values.gather(1, move1_indices.unsqueeze(1)).squeeze(1)
            
            # Get max Q-values for first moves from target network - keep on device
            with torch.no_grad():
                next_move1_q_values = self.target_model.network(next_states)
                max_next_move1_q = next_move1_q_values.max(1)[0]
            
            # Compute target Q-values for first moves - keep on device
            target_move1_q = rewards + (1 - dones) * self.gamma * max_next_move1_q
            
            # Compute loss for first moves - keep on device
            move1_loss = self.loss_fn(move1_q_values, target_move1_q)
            
            # Apply importance sampling weights if using prioritized replay
            if self.use_prioritized_replay:
                move1_loss = move1_loss * weights_tensor
            
            # Compute mean loss - keep on device
            move1_loss = move1_loss.mean()
            loss_value = move1_loss.item()  # This detaches from computation graph
            
            # Compute TD errors for prioritized replay
            if self.use_prioritized_replay:
                with torch.no_grad():
                    td_errors = torch.abs(move1_q_values - target_move1_q).cpu().numpy()
                    self.memory.update_priorities(indices, td_errors)
                    
                    # Track TD error statistics
                    self.last_td_error_mean = td_errors.mean()
                    self.last_td_error_max = td_errors.max()
            
            # Backward pass and optimization - optimize for GPU usage
            if self.device.type == "mps":
                # MPS-specific simplified training - no gradients accumulation
                move1_loss.backward()
                # Clip gradients with a lower max norm for stability
                torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Only print progress in main process
                if mp.current_process().name == 'MainProcess':
                    print(".", end="", flush=True)  # Progress indicator without newline
            elif self.scaler is not None:
                # Use mixed precision training for CUDA
                self.scaler.scale(move1_loss).backward()
                if self.accumulated_steps == self.grad_accumulation_steps - 1:
                    # Clip gradients to stabilize training
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.accumulated_steps = 0
                else:
                    self.accumulated_steps += 1
        else:
            # For the original flat approach
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Get current Q values
            current_q_values = self.model.network(states)
            
            # Get next Q values from target model in one batch operation
            with torch.no_grad():
                next_q_values = self.target_model.network(next_states)
                max_next_q = next_q_values.max(1)[0]
            
                # Compute target Q values
                target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Gather the Q values for the actions taken
            q_values = current_q_values.gather(1, move1_indices.unsqueeze(1)).squeeze(1)
            
            # Compute loss
            loss = self.loss_fn(q_values, target_q)
            
            # Apply importance sampling weights if using prioritized replay
            if self.use_prioritized_replay:
                loss = loss * weights_tensor
            
            # Compute mean loss
            loss = loss.mean()
            loss_value = loss.item()
            
            # Backward pass and optimization
            loss.backward()
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), max_norm=10.0)
            self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()
        
        return loss_value 