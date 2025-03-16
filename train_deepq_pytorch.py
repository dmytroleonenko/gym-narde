import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def compute_progress(game, current_player):
    """
    Compute remaining progress (sum of pips) and borne-off count.
    """
    board = game.get_perspective_board(current_player)
    remaining_distance = sum(i * max(0, board[i]) for i in range(24))
    borne_off = game.borne_off_white if current_player == 1 else game.borne_off_black
    return remaining_distance, borne_off

def compute_coverage_reward(board):
    """
    Count number of distinct points with at least 2 checkers.
    """
    return sum(1 for point in board if point >= 2)

def compute_block_reward(board):
    """
    Compute bonus for consecutive points with at least 2 checkers.
    """
    block_bonus = 0
    consecutive_count = 0
    for i in range(len(board)):
        if board[i] >= 2:
            consecutive_count += 1
        else:
            if consecutive_count > 1:
                block_bonus += (consecutive_count ** 1.5)
            consecutive_count = 0
    if consecutive_count > 1:
        block_bonus += (consecutive_count ** 1.5)
    return block_bonus
from collections import deque
import gym_narde  # Important: Import the custom environment
from gym_narde.envs.narde import rotate_board  # Import for debugging

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def render_board(board, highlight_from=None, highlight_to=None):
    """
    Render the Narde board in a triangular backgammon style layout.
    
    Args:
        board: The board state array (24 positions, 0-indexed)
        highlight_from: Position to highlight as the source of a move
        highlight_to: Position to highlight as the destination of a move
    
    Returns:
        String representation of the board
    """
    # Define piece and board style elements
    white_piece = f"{Colors.WHITE}{Colors.BOLD}○{Colors.RESET}"
    black_piece = f"{Colors.BLACK}{Colors.BOLD}●{Colors.RESET}"
    highlight_from_bg = Colors.BG_YELLOW
    highlight_to_bg = Colors.BG_CYAN
    
    # Build the board
    board_str = "\n"
    
    # Add the top border with point numbers (13-24)
    board_str += f"  {Colors.BOLD}13 14 15 16 17 18    19 20 21 22 23 24{Colors.RESET}\n"
    board_str += "  " + "┌" + "─" * 17 + "┬" + "─" * 17 + "┐\n"
    
    # Top row showing triangular pattern
    board_str += "  │" + "\\/" * 6 + "│" + "\\/" * 6 + "│\n"
    
    # Top section rows for pieces (points 13-24)
    for row in range(4):
        board_str += "  │"
        
        # Points 13-18
        for point in range(13, 19):
            point_idx = point - 1  # Convert to 0-indexed for array access
            
            # Check if this position needs highlighting
            if point_idx == highlight_from:
                board_str += highlight_from_bg
            elif point_idx == highlight_to:
                board_str += highlight_to_bg
            
            # Display piece if present at this position in the stack
            if board[point_idx] > 0 and board[point_idx] >= row+1:
                board_str += white_piece + " "
            elif board[point_idx] < 0 and abs(board[point_idx]) >= row+1:
                board_str += black_piece + " "
            else:
                board_str += "  "
            
            # Reset background color if highlighted
            if point_idx == highlight_from or point_idx == highlight_to:
                board_str += Colors.RESET
        
        board_str += "│"
        
        # Points 19-24
        for point in range(19, 25):
            point_idx = point - 1  # Convert to 0-indexed for array access
            
            # Check if this position needs highlighting
            if point_idx == highlight_from:
                board_str += highlight_from_bg
            elif point_idx == highlight_to:
                board_str += highlight_to_bg
            
            # Display piece if present at this position in the stack
            if board[point_idx] > 0 and board[point_idx] >= row+1:
                board_str += white_piece + " "
            elif board[point_idx] < 0 and abs(board[point_idx]) >= row+1:
                board_str += black_piece + " "
            else:
                board_str += "  "
            
            # Reset background color if highlighted
            if point_idx == highlight_from or point_idx == highlight_to:
                board_str += Colors.RESET
        
        board_str += "│\n"
    
    # Middle bar
    board_str += "  ├" + "─" * 17 + "┼" + "─" * 17 + "┤\n"
    
    # Bottom section with points 12-1
    for row in range(4, 0, -1):  # Display bottom rows
        board_str += "  │"
        
        # Points 12-7 (display in reverse order)
        for point in range(12, 6, -1):
            point_idx = point - 1  # Convert to 0-indexed for array access
            
            # Check if this position needs highlighting
            if point_idx == highlight_from:
                board_str += highlight_from_bg
            elif point_idx == highlight_to:
                board_str += highlight_to_bg
            
            # Display piece if present at this position in the stack
            if board[point_idx] > 0 and board[point_idx] >= row:
                board_str += white_piece + " "
            elif board[point_idx] < 0 and abs(board[point_idx]) >= row:
                board_str += black_piece + " "
            else:
                board_str += "  "
            
            # Reset background color if highlighted
            if point_idx == highlight_from or point_idx == highlight_to:
                board_str += Colors.RESET
        
        board_str += "│"
        
        # Points 6-1 (display in reverse order)
        for point in range(6, 0, -1):
            point_idx = point - 1  # Convert to 0-indexed for array access
            
            # Check if this position needs highlighting
            if point_idx == highlight_from:
                board_str += highlight_from_bg
            elif point_idx == highlight_to:
                board_str += highlight_to_bg
            
            # Display piece if present at this position in the stack
            if board[point_idx] > 0 and board[point_idx] >= row:
                board_str += white_piece + " "
            elif board[point_idx] < 0 and abs(board[point_idx]) >= row:
                board_str += black_piece + " "
            else:
                board_str += "  "
            
            # Reset background color if highlighted
            if point_idx == highlight_from or point_idx == highlight_to:
                board_str += Colors.RESET
        
        board_str += "│\n"
    
    # Bottom triangular pattern
    board_str += "  │" + "/\\" * 6 + "│" + "/\\" * 6 + "│\n"
    
    # Bottom border with point numbers (12-1)
    board_str += "  └" + "─" * 17 + "┴" + "─" * 17 + "┘\n"
    board_str += f"  {Colors.BOLD}12 11 10  9  8  7     6  5  4  3  2  1{Colors.RESET}\n"
    
    return board_str
    
    # Bottom border with point numbers (12-1)
    board_str += "  └" + "─" * 17 + "┴" + "─" * 17 + "┘\n"
    board_str += f"  {Colors.BOLD}12 11 10  9  8  7     6  5  4  3  2  1{Colors.RESET}\n"
    
    return board_str

# Create the environment
env = gym.make('Narde-v0', render_mode=None)

class DecomposedDQN(nn.Module):
    def __init__(self, state_size=28, move_space_size=576):
        super(DecomposedDQN, self).__init__()
        self.move_space_size = move_space_size
        
        # Shared feature extractor
        self.feature_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # First move head
        self.move1_head = nn.Linear(256, move_space_size)
        
        # Second move head - takes state and first move as input
        self.move2_head = nn.Linear(256 + move_space_size, move_space_size)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, selected_move1=None):
        """
        Forward pass through the network
        
        Args:
            x: State tensor
            selected_move1: If provided, used to select specific move1 for move2 prediction
                            If None, return Q-values for all move1 options
        
        Returns:
            If selected_move1 is None: move1_q_values
            If selected_move1 is provided: move2_q_values
        """
        batch_size = x.size(0)
        features = self.feature_network(x)
        
        # Get Q-values for first move
        move1_q_values = self.move1_head(features)
        
        # If we're not selecting move2 based on move1, just return move1 values
        if selected_move1 is None:
            return move1_q_values
        
        # Create one-hot encoding of selected move1
        move1_onehot = torch.zeros(batch_size, self.move_space_size, device=x.device)
        move1_onehot.scatter_(1, selected_move1.unsqueeze(1), 1)
        
        # Concatenate features with one-hot encoding of move1
        combined_features = torch.cat((features, move1_onehot), dim=1)
        
        # Get Q-values for second move
        move2_q_values = self.move2_head(combined_features)
        
        return move2_q_values
    
    def get_combined_q_values(self, x):
        """
        Get Q-values for all possible move1/move2 combinations (for backward compatibility)
        
        Args:
            x: State tensor (batch_size, state_size)
            
        Returns:
            combined_q_values: Tensor of shape (batch_size, move_space_size * move_space_size)
                              representing Q-values for all move combinations
        """
        batch_size = x.size(0)
        features = self.feature_network(x)
        
        # Get Q-values for first move
        move1_q_values = self.move1_head(features)
        
        # For each possible move1, calculate move2 Q-values
        combined_q_values = []
        
        for move1_idx in range(self.move_space_size):
            # Create one-hot encoding for this move1
            move1_onehot = torch.zeros(batch_size, self.move_space_size, device=x.device)
            move1_onehot[:, move1_idx] = 1
            
            # Combine features with move1 encoding
            combined_features = torch.cat((features, move1_onehot), dim=1)
            
            # Get Q-values for move2 given this move1
            move2_q_values = self.move2_head(combined_features)
            
            # For each move1, multiply by corresponding move2 Q-values
            for move2_idx in range(self.move_space_size):
                move2_q = move2_q_values[:, move2_idx]
                # This is the Q-value for the combined move1 + move2
                # Index in the flattened space would be move1_idx * move_space_size + move2_idx
                combined_q_values.append(move1_q_values[:, move1_idx] + move2_q)
                
        # Stack and reshape to (batch_size, move_space_size * move_space_size)
        return torch.stack(combined_q_values, dim=1)

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
    def __init__(self, state_size=28, action_size=576 * 576, use_decomposed_network=True, use_prioritized_replay=True):
        self.state_size = state_size
        self.action_size = action_size
        
        # Replay buffer - either prioritized or regular
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(capacity=50000)
        else:
            self.memory = deque(maxlen=50000)  # Standard experience replay buffer
            
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.use_decomposed_network = use_decomposed_network
        self.move_space_size = 576  # 24*24 possible moves
        
        # Determine device (CUDA, MPS or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Main network
        if use_decomposed_network:
            print("Using decomposed action space network")
            self.model = DecomposedDQN(state_size, self.move_space_size).to(self.device)
            self.target_model = DecomposedDQN(state_size, self.move_space_size).to(self.device)
            
        self.update_target_model()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss(reduction='none')  # Use 'none' to apply importance sampling weights
        
        # Initialize TD error statistics
        self.last_td_error_mean = 0.0
        self.last_td_error_max = 0.0

        # Update target network every 10 steps
        self.update_target_freq = 10
        self.train_step = 0

    def update_target_model(self):
        """Update the target model with weights from the main model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        experience = (state, action, reward, next_state, done)
        
        if self.use_prioritized_replay:
            # Add with max priority for new experiences
            self.memory.add(experience)
        else:
            # Standard experience replay
            self.memory.append(experience)

    def act(self, state, valid_moves=None, env=None, dice=None, current_player=None, training=True):
        """Return action based on epsilon-greedy policy with valid moves only, using decomposed action selection for efficiency"""
        # Generate all valid move combinations
        valid_move_combinations = []
        valid_first_moves = {}  # Dictionary to track unique first moves
        
        # If valid_moves is provided, only consider valid moves
        if valid_moves is not None and env is not None:
            # Add a single valid move with no second move if we can't determine second moves
            if not dice or len(dice) == 0:
                # No dice or empty dice, just use the first valid move with no second move
                if valid_moves:
                    move1 = valid_moves[0]
                    from_pos1, to_pos1 = move1
                    move1_code = from_pos1 * 24 + (0 if to_pos1 == 'off' else to_pos1)
                    valid_move_combinations.append((move1_code, 0))
                    valid_first_moves[move1_code] = []  # No second moves
            else:
                # Process first move options
                for move1 in valid_moves:
                    # Convert move to code (handle 'off' special case)
                    from_pos1, to_pos1 = move1
                    if to_pos1 == 'off':
                        # Special case for bearing off
                        move1_code = from_pos1 * 24  # Use position 0 as placeholder for 'off'
                    else:
                        move1_code = from_pos1 * 24 + to_pos1
                    
                    # Create a temporary copy of dice for this move simulation
                    temp_dice = dice.copy() if dice else []
                    
                    # Determine which die was used for this move
                    try:
                        # Calculate the move distance
                        if to_pos1 == 'off': 
                            # Special case for bearing off - can use any die that's large enough or exact
                            # For bearing off, we need a die >= (point_position + 1)
                            move_distance = from_pos1 + 1  # The die needed to bear off
                            
                            # Find a die that's large enough (exact or larger)
                            matching_die = None
                            for d in temp_dice:
                                if d >= move_distance:
                                    matching_die = d
                                    break
                                    
                            # If no die is large enough but we have a valid bearing off move,
                            # it must be because we're using the exact die or largest die rule
                            if matching_die is None and temp_dice:
                                # Use the largest available die
                                matching_die = max(temp_dice)
                        else:
                            # Normal move - calculate exact distance
                            move_distance = abs(from_pos1 - to_pos1)
                            # Try to find exactly matching die
                            matching_die = None
                            for d in temp_dice:
                                if d == move_distance:
                                    matching_die = d
                                    break
                            # If no exact match, use first die
                            if matching_die is None and temp_dice:
                                matching_die = temp_dice[0]
                                
                        # Remove the used die from the list
                        if matching_die is not None and matching_die in temp_dice:
                            temp_dice.remove(matching_die)
                    except Exception:
                        # If any issues, use an empty list for remaining dice
                        temp_dice = []
                    
                    # Get valid moves for remaining dice (if any)
                    remaining_moves = []
                    if temp_dice:
                        try:
                            remaining_moves = env.unwrapped.game.get_valid_moves(temp_dice, current_player)
                        except Exception:
                            # If getting remaining moves fails, just use empty list
                            remaining_moves = []
                    
                    # Initialize list of second moves for this first move
                    valid_first_moves[move1_code] = []
                    
                    # If no remaining moves or dice, only use the first move
                    if not remaining_moves:
                        valid_move_combinations.append((move1_code, 0))  # Use 0 as placeholder for no second move
                        valid_first_moves[move1_code].append(0)  # Add "no second move" option
                    else:
                        # Add combinations with all valid second moves
                        for move2 in remaining_moves:
                            from_pos2, to_pos2 = move2
                            if to_pos2 == 'off':
                                move2_code = from_pos2 * 24  # Use position 0 for 'off'
                            else:
                                move2_code = from_pos2 * 24 + to_pos2
                            valid_move_combinations.append((move1_code, move2_code))
                            valid_first_moves[move1_code].append(move2_code)
        
        # If no valid move combinations, return a default "pass" action
        if not valid_move_combinations:
            return (0, 0)  # Skip turn
            
        # Choose a random valid move during exploration
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_move_combinations)
        
        # Otherwise, choose the best valid action according to Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if self.use_decomposed_network:
            # Use the decomposed network approach
            with torch.no_grad():
                # Get Q-values for first moves
                move1_q_values = self.model(state_tensor)
                
                # Extract Q-values for valid first moves
                valid_move1_indices = list(valid_first_moves.keys())
                
                # If no valid moves, return a default action
                if not valid_move1_indices:
                    return (0, 0)
                
                # Convert to tensor for indexing
                valid_move1_tensor = torch.tensor(valid_move1_indices, device=self.device)
                valid_move1_q_values = move1_q_values.squeeze(0).index_select(0, valid_move1_tensor)
                
                # Find best first move
                best_move1_idx = torch.argmax(valid_move1_q_values).item()
                best_move1_code = valid_move1_indices[best_move1_idx]
                
                # Get valid second moves for this first move
                valid_move2_codes = valid_first_moves[best_move1_code]
                
                # If no valid second moves, return just the first move
                if not valid_move2_codes:
                    return (best_move1_code, 0)
                
                # Get Q-values for second moves based on selected first move
                selected_move1 = torch.tensor([best_move1_code % self.move_space_size], device=self.device)
                move2_q_values = self.model(state_tensor, selected_move1)
                
                # Extract Q-values for valid second moves
                valid_move2_tensor = torch.tensor(valid_move2_codes, device=self.device)
                valid_move2_q_values = move2_q_values.squeeze(0).index_select(0, valid_move2_tensor)
                
                # Find best second move
                best_move2_idx = torch.argmax(valid_move2_q_values).item()
                best_move2_code = valid_move2_codes[best_move2_idx]
                
                return (best_move1_code, best_move2_code)
                
        else:
            # Use the original flat approach for backward compatibility
            with torch.no_grad():
                if self.use_decomposed_network:
                    # Use the combined Q-values method for compatibility
                    q_values = self.model.get_combined_q_values(state_tensor)
                else:
                    q_values = self.model(state_tensor)
                
            # Vectorized approach for selecting the best valid action
            # Step 1: Convert valid moves to action indices
            action_indices = []
            for move1_code, move2_code in valid_move_combinations:
                action_idx = move1_code * 576 + move2_code
                # Only include indices that are within bounds
                if action_idx < q_values.size(1):
                    action_indices.append(action_idx)
                    
            # If no valid action indices, select a random move
            if not action_indices:
                return random.choice(valid_move_combinations)
                
            # Step 2: Create a tensor of valid action indices
            valid_action_indices = torch.tensor(action_indices, device=self.device)
            
            # Step 3: Use gather to extract Q-values for all valid actions at once
            valid_q_values = q_values.squeeze(0).index_select(0, valid_action_indices)
            
            # Step 4: Find the index of the action with the highest Q-value
            best_idx = torch.argmax(valid_q_values).item()
            
            # Step 5: Get the corresponding action
            best_action_idx = action_indices[best_idx]
            
            # Convert back to the original action format
            move1_code = best_action_idx // 576
            move2_code = best_action_idx % 576
            
            return (move1_code, move2_code)

    def replay(self):
        """Train the model with experiences from memory using vectorized operations,
        with support for prioritized replay and decomposed action network"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample minibatch from memory (differently depending on replay type)
        if self.use_prioritized_replay:
            experiences, indices, weights = self.memory.sample(self.batch_size)
            # Convert importance sampling weights to tensor
            is_weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = random.sample(self.memory, self.batch_size)
            indices = None
            is_weights = torch.ones(self.batch_size).to(self.device)  # No weighting for uniform sampling
        
        # Extract data from experiences using vectorized operations
        # Unzip the experiences using zip(*...)
        states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*experiences)
        
        # Convert to numpy arrays and then to tensors
        states = torch.FloatTensor(np.array(states_list, dtype=np.float32)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards_list, dtype=np.float32)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states_list, dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor(np.array(dones_list, dtype=np.float32)).to(self.device)
        
        # Note: We don't convert actions to tensor yet, as we need to process them differently
        # based on the network type (decomposed or flat)
        
        # For storing TD errors (to update priorities)
        td_errors = torch.zeros(self.batch_size).to(self.device)
        
        if self.use_decomposed_network:
            # Decomposed action learning approach
            # Extract move1 and move2 from actions using vectorized operations
            # Convert list of (move1, move2) tuples to two separate arrays
            actions_array = np.array(actions_list)
            move1_indices = torch.LongTensor(actions_array[:, 0]).to(self.device)
            move2_indices = torch.LongTensor(actions_array[:, 1]).to(self.device)
            
            # STEP 1: Update move1 network
            # Get current Q-values for first moves
            current_move1_q_values = self.model(states)
            move1_q_values = current_move1_q_values.gather(1, move1_indices.unsqueeze(1)).squeeze(1)
            
            # Get max Q-values for first moves from target network
            with torch.no_grad():
                next_move1_q_values = self.target_model(next_states)
                max_next_move1_q = next_move1_q_values.max(1)[0]
            
            # Calculate target for move1
            target_move1_q_values = rewards + (1 - dones) * self.gamma * max_next_move1_q
            
            # Calculate TD error for move1 (for prioritized replay)
            td_error_move1 = torch.abs(target_move1_q_values - move1_q_values)
            
            # STEP 2: Update move2 network based on selected move1
            # Get current Q-values for second moves
            current_move2_q_values = self.model(states, move1_indices)
            move2_q_values = current_move2_q_values.gather(1, move2_indices.unsqueeze(1)).squeeze(1)
            
            # Get max Q-values for second moves from target network
            with torch.no_grad():
                # For the next state, first get the best move1
                next_move1_indices = next_move1_q_values.argmax(dim=1)
                # Then get the best move2 based on that move1
                next_move2_q_values = self.target_model(next_states, next_move1_indices)
                max_next_move2_q = next_move2_q_values.max(1)[0]
            
            # Calculate target for move2
            target_move2_q_values = rewards + (1 - dones) * self.gamma * max_next_move2_q
            
            # Calculate TD error for move2 (for prioritized replay)
            td_error_move2 = torch.abs(target_move2_q_values - move2_q_values)
            
            # Use combined TD error for prioritized replay updates, but clip to prevent extremely large values
            td_errors = torch.clamp(td_error_move1 + td_error_move2, 0.0, 100.0)
            
            # Compute additional metrics:
            td_error_mean = td_errors.mean().item()
            td_error_max = td_errors.abs().max().item()
            self.last_td_error_mean = td_error_mean
            self.last_td_error_max = td_error_max

            # Apply importance sampling weights to losses
            self.optimizer.zero_grad()
            
            # Calculate element-wise losses
            loss_move1_elements = self.loss_fn(move1_q_values, target_move1_q_values)
            loss_move2_elements = self.loss_fn(move2_q_values, target_move2_q_values)
            
            # Apply gradient clipping to prevent exploding gradients
            loss_move1 = (is_weights * loss_move1_elements).mean()
            loss_move2 = (is_weights * loss_move2_elements).mean()
            
            # Combined loss
            loss = loss_move1 + loss_move2
            
            # Backward pass with gradient clipping
            loss.backward()
            # Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
        else:
            # Original flat approach for backward compatibility
            # Convert actions to indices - vectorized with list comprehension
            action_indices = np.array([action_to_idx(action) for action in actions_list])
            actions = torch.LongTensor(action_indices).to(self.device)
            
            # Get current Q values
            current_q_values = self.model(states)
            
            # Get next Q values from target model in one batch operation
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                max_next_q = next_q_values.max(1)[0]
                
            # Calculate target Q values: if done, target = reward, else target = reward + gamma * max(next_q)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Create a mask for the actions taken
            mask = torch.zeros(self.batch_size, self.action_size, device=self.device)
            mask.scatter_(1, actions.unsqueeze(1), 1)
            
            # Set the target for only the actions that were taken
            current_q_values_for_actions = torch.sum(current_q_values * mask, dim=1)
            
            # Calculate TD error for prioritized replay (clamped to prevent explosion)
            td_errors = torch.clamp(torch.abs(target_q_values - current_q_values_for_actions), 0.0, 100.0)
            
            # Compute additional metrics:
            td_error_mean = td_errors.mean().item()
            td_error_max = td_errors.abs().max().item()
            self.last_td_error_mean = td_error_mean
            self.last_td_error_max = td_error_max

            # Apply importance sampling weights for prioritized replay
            self.optimizer.zero_grad()
            elementwise_loss = self.loss_fn(current_q_values_for_actions, target_q_values)  # Already has reduction='none'
            loss = (is_weights * elementwise_loss).mean()
            loss.backward()
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
        
        # Update priorities in prioritized replay buffer
        if self.use_prioritized_replay and indices is not None:
            priorities = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()
        
        return loss.item()

def action_to_idx(action):
    """Convert action tuple to index"""
    move1_code, move2_code = action
    return move1_code * 576 + move2_code

def idx_to_action(idx):
    """Convert index to action tuple"""
    move1_code = idx // 576
    move2_code = idx % 576
    return (move1_code, move2_code)

def action_to_idx(action):
    """Convert action tuple to index"""
    move1_code, move2_code = action
    return move1_code * 576 + move2_code

def idx_to_action(idx):
    """Convert index to action tuple"""
    move1_code = idx // 576
    move2_code = idx % 576
    return (move1_code, move2_code)

def create_moves_from_action(action):
    """Convert action to moves format for the environment"""
    move1_code, move2_code = action
    
    # Convert move1_code to from_pos1, to_pos1
    from_pos1 = move1_code // 24
    to_pos1 = move1_code % 24
    
    # Special case: to_pos1 = 0 might represent 'off'
    # This would be the case especially for bearing off moves
    # We'll check if from_pos1 is in the home area (0-5) and if to_pos1 is 0
    if 0 <= from_pos1 <= 5 and to_pos1 == 0:
        to_pos1 = 'off'
    
    # Convert move2_code to from_pos2, to_pos2 (if move2 is valid)
    if move2_code > 0:  # 0 means no second move
        from_pos2 = move2_code // 24
        to_pos2 = move2_code % 24
        
        # Same special case for move2
        if 0 <= from_pos2 <= 5 and to_pos2 == 0:
            to_pos2 = 'off'
    else:
        # No second move, use placeholder values
        from_pos2 = 0
        to_pos2 = 0
    
    # Create move tuples as expected by the environment
    move1 = (from_pos1, to_pos1)
    move2 = (from_pos2, to_pos2)
    
    return move1, move2

def check_move_valid(env, move, dice, current_player):
    """Check if a move is valid according to the game rules"""
    valid_moves = env.unwrapped.game.get_valid_moves(dice, current_player)
    return move in valid_moves

def main(episodes=10000, max_steps=1000, epsilon=1.0, epsilon_decay=0.995, learning_rate=1e-4, 
         verbose=False, verbose_steps=100, log=False, log_file='training_log.txt', 
         use_decomposed_network=True, use_prioritized_replay=True):
    
    # Set up logging if needed
    log_file_handle = None
    if log:
        log_file_handle = open(log_file, 'w')
        # Helper function to log output
        def log_output(message):
            if log_file_handle:
                log_file_handle.write(message + '\n')
                log_file_handle.flush()
    else:
        # If not logging, just define a no-op function
        def log_output(message):
            pass
    
    # Create agent
    state_size = 28  # 24 board positions + 2 dice + 2 borne_off values
    action_size = 576 * 576  # All possible move combinations (24*24 for move1 * 24*24 for move2)
    agent = DQNAgent(state_size, action_size, 
                     use_decomposed_network=use_decomposed_network,
                     use_prioritized_replay=use_prioritized_replay)
    
    # Set custom parameters if provided
    agent.epsilon = epsilon
    agent.epsilon_decay = epsilon_decay
    agent.learning_rate = learning_rate
    agent.optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)
    
    # Create statistics tracking
    episode_rewards = []
    episode_lengths = []
    loss_values = []
    games_won = 0
    
    # Save directory
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Training loop
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = None
        
        # Initialize tracking for piece bear-off
        total_white_off = 0
        total_black_off = 0
        
        for step in range(max_steps):
            # Use dice from the environment
            valid_moves = env.unwrapped.game.get_valid_moves(env.dice, env.unwrapped.current_player)
            
            # --- New Reward Shaping Start ---
            # For both cases (valid or non-valid moves), we compute shaping rewards.
            # If no valid moves, we can simply call env.step(action) using (0,0).
            if len(valid_moves) == 0:
                action = (0, 0)
            else:
                action = agent.act(state, valid_moves=valid_moves, env=env,
                                   dice=env.dice, current_player=env.unwrapped.current_player,
                                   training=True)
            # Capture OLD board and progress before stepping:
            old_board = env.unwrapped.game.get_perspective_board(env.unwrapped.current_player)
            old_head_count = old_board[23]  # Assuming White's head is index 23.
            prev_distance, prev_borne_off = compute_progress(env.unwrapped.game, env.unwrapped.current_player)
            
            # Step the environment.
            next_state, env_reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Capture NEW board and progress after stepping:
            new_board = env.unwrapped.game.get_perspective_board(env.unwrapped.current_player)
            new_head_count = new_board[23]
            new_distance, new_borne_off = compute_progress(env.unwrapped.game, env.unwrapped.current_player)

            # Compute incremental rewards:
            progress_reward = 0.001 * (prev_distance - new_distance)
            borne_off_increment = new_borne_off - prev_borne_off
            borne_reward = 0.1 * borne_off_increment
            coverage_reward = 0.0005 * compute_coverage_reward(new_board)
            # Here we compute block_reward as a bonus difference; in this simple example, we use the current board.
            block_reward = 0.0001 * compute_block_reward(new_board)
            head_bonus = 0.0
            if new_head_count < old_head_count:
                head_bonus = 0.05 * (old_head_count - new_head_count)
            
            # Final shaped reward:
            reward = env_reward + progress_reward + borne_reward + coverage_reward + block_reward + head_bonus
            # --- New Reward Shaping End ---
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train on remembered experiences
            loss = agent.replay()
            if loss is not None:
                episode_loss = loss
                loss_values.append(loss)
            
            state = next_state
            total_reward += reward
            
            # Print step stats in verbose mode or log
            if (verbose and (step + 1) % verbose_steps == 0) or log:
                # Ensure we're getting the updated board state after the step is executed
                board = next_state  # next_state is the updated state returned from env.step() 
                borne_off_white = env.unwrapped.game.borne_off_white
                borne_off_black = env.unwrapped.game.borne_off_black
                player = "White" if env.unwrapped.current_player == 1 else "Black"
                
                # Create the step header
                header = f"\nStep: {step+1}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Player: {player}"
                stats = f"White pieces off board: {env.unwrapped.game.borne_off_white}, Black pieces off board: {env.unwrapped.game.borne_off_black}"
                
                # Create the loss string if available
                loss_str = f"Loss: {loss:.8f}" if loss is not None else ""
                
                # Create the dice string
                dice_str = f"Dice: {dice}"
                
                # Create the move string
                move_str = ""
                highlight_from = None
                highlight_to = None
                
                if len(valid_moves) > 0:
                    # Get the moves from the action
                    move1, move2 = create_moves_from_action(action)
                    
                    # Set up highlighting positions
                    if isinstance(move1[0], int) and move1[0] < 24:  # Make sure it's a valid position
                        highlight_from = move1[0]
                    
                    if isinstance(move1[1], int) and move1[1] < 24:  # Make sure it's a valid position
                        highlight_to = move1[1] 
                        
                    # Format move representation for better readability
                    if move1[1] == 'off':
                        move1_str = f"({move1[0]}, off)"
                    else:
                        move1_str = f"({move1[0]}, {move1[1]})"
                        
                    if move2[0] == 0 and move2[1] == 0:
                        # No second move
                        move_str = f"Move: {move1_str}"
                    else:
                        if move2[1] == 'off':
                            move2_str = f"({move2[0]}, off)"
                        else:
                            move2_str = f"({move2[0]}, {move2[1]})"
                        move_str = f"Move: {move1_str} -> {move2_str}"
                    
                    # Add sample of available valid moves
                    if len(valid_moves) > 0:
                        sample_size = min(3, len(valid_moves))
                        sample_moves = [f"({m[0]}, {m[1]})" for m in valid_moves[:sample_size]]
                        move_str += f"\nSample valid moves: {', '.join(sample_moves)}" + \
                                  f"{' (+ ' + str(len(valid_moves)-sample_size) + ' more)' if len(valid_moves) > sample_size else ''}"
                else:
                    move_str = "Move: Skip turn (no valid moves)"
                
                # Create the board visualization with move highlighting
                board_viz = render_board(board, highlight_from=highlight_from, highlight_to=highlight_to)
                
                # Add off-board pieces info
                white_off = env.unwrapped.game.borne_off_white
                black_off = env.unwrapped.game.borne_off_black
                
                off_board_info = (
                    f"  White pieces off board: {white_off}\n"
                    f"  Black pieces off board: {black_off}"
                )
                
                # Combine all output
                output = f"{header}\n{stats}\n{loss_str}\n{dice_str}\n{move_str}\n{board_viz}\n{off_board_info}\n{'-' * 60}"
                
                # Log or print the output
                if log:
                    log_output(output)
                
                if verbose and (step + 1) % verbose_steps == 0:
                    # If verbose and it's a reporting step, print the output to the console
                    print(output)
                
                # Only show debug info for no valid moves in verbose mode
                if verbose and (step + 1) % verbose_steps == 0 and len(valid_moves) == 0:
                    # Debug: Show why no moves are valid by analyzing dice and board state
                    print("\nDebug - Why no moves are valid by analyzing dice and board state:")
                    
                    # Check if there are any checkers of the current player on the board
                    # Use next_state for the current board state
                    current_player = env.unwrapped.current_player
                    if current_player == 1:
                        player_checkers = [i for i, val in enumerate(next_state) if val > 0]
                        opponent_checkers = [i for i, val in enumerate(next_state) if val < 0]
                        player_name = "White"
                    else:
                        # For Black, the board is already in White's perspective
                        player_checkers = [i for i, val in enumerate(next_state) if val < 0]
                        opponent_checkers = [i for i, val in enumerate(next_state) if val > 0]
                        player_name = "Black"
                    
                    print(f"{player_name} checkers at positions: {player_checkers}")
                    print(f"Opponent checkers at positions: {opponent_checkers}")
                    print(f"Dice: {dice}")
                    
                    # Check if possible moves would violate the block rule
                    all_moves = []
                    for pos in player_checkers:
                        for die in dice:
                            new_pos = pos - die  # Movement is counter-clockwise (decreasing index)
                            if 0 <= new_pos < 24:
                                all_moves.append((pos, new_pos))
                    
                    print(f"Potential moves before filtering: {all_moves}")
                    
                    # Check head rule
                    if current_player == 1:
                        head_pos = 23
                    else:
                        head_pos = 11
                    
                    # Check head rule
                    if head_pos in player_checkers:
                        is_first_turn = env.unwrapped.game.first_turn_white if current_player == 1 else env.unwrapped.game.first_turn_black
                        if is_first_turn:
                            if sorted(dice) in [[3,3], [4,4], [6,6]]:
                                print("First turn with doubles 3, 4, or 6 - can move 2 checkers from head")
                            else:
                                print("First turn - can move only 1 checker from head")
                        else:
                            print("Not first turn - can move only 1 checker from head")
                    
                    # Check if moves might be filtered by the block rule
                    # Create a board in the current player's perspective for testing
                    if current_player == 1:
                        test_board = next_state.copy()
                    else:
                        # For Black, we need to rotate the board perspective
                        test_board = rotate_board(next_state)
                    
                    # Test each potential move for block rule violations
                    for move in all_moves:
                        test_board_copy = test_board.copy()
                        from_pos, to_pos = move
                        
                        # Simulate the move
                        test_board_copy[from_pos] -= 1
                        test_board_copy[to_pos] += 1
                        
                        # Check if this move would create an illegal block
                        if env.unwrapped.game._violates_block_rule(test_board_copy):
                            print(f"Move {move} would create an illegal block (Rule 8 violation)")
                    
            if done:
                # Track statistics
                games_won += 1
                break
        
        # Update epsilon once per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Update statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        
        # Compute and log loss improvement metrics if available
        improvement_10_str = ""
        if (e + 1) >= 10 and len(loss_values) >= 20:
            current_10_avg = np.mean(loss_values[-10:])
            previous_10_avg = np.mean(loss_values[-20:-10])
            improvement_10 = ((previous_10_avg - current_10_avg) / previous_10_avg) * 100
            improvement_10_str = f", Loss improvement (last 10): {improvement_10:+.2f}%"
        else:
            improvement_10_str = ", Loss improvement (last 10): N/A"

        improvement_50_str = ""
        if (e + 1) >= 50 and len(loss_values) >= 100:
            current_50_avg = np.mean(loss_values[-50:])
            previous_50_avg = np.mean(loss_values[-100:-50])
            improvement_50 = ((previous_50_avg - current_50_avg) / previous_50_avg) * 100
            improvement_50_str = f", Loss improvement (last 50): {improvement_50:+.2f}%"
        else:
            improvement_50_str = ", Loss improvement (last 50): N/A"
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Steps: {step+1}, "
              f"Epsilon: {agent.epsilon:.2f}, Loss: {episode_loss if episode_loss else 'N/A'}, "
              f"TD Error Mean: {agent.last_td_error_mean:.4f}, TD Error Max: {agent.last_td_error_max:.4f}"
              f"{improvement_10_str}{improvement_50_str}")
        
        # In verbose or log modes, include final board state
        if log or (e+1) % 10 == 0:
                # Get the actual board state
            board = env.unwrapped.game.board
            
            # Create a nice summary
            episode_summary = (
                f"\n{'=' * 60}\n"
                f"Episode {e+1}/{episodes} Complete\n"
                f"Score: {total_reward:.2f}\n"
                f"Steps: {step+1}\n"
                f"Epsilon: {agent.epsilon:.4f}\n"
                f"Game Status:\n"
                f"  White pieces off board: {env.unwrapped.game.borne_off_white}\n"
                f"  Black pieces off board: {env.unwrapped.game.borne_off_black}\n"
                f"Final Board State:\n{render_board(board)}\n"
                f"{'=' * 60}"
            )
            
            # Log or print the summary
            if log:
                log_output(episode_summary)
                
            if verbose and (e+1) % 10 == 0:
                print(episode_summary)
        
        # Print overall statistics every 100 episodes
        if (e+1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            avg_loss = sum(loss_values[-100:]) / max(1, len(loss_values[-100:]))
            
            print(f"\n=== Statistics after {e+1} episodes ===")
            print(f"Average reward (last 100 episodes): {avg_reward:.2f}")
            print(f"Average episode length (last 100 episodes): {avg_length:.2f}")
            print(f"Average loss (last 100 episodes): {avg_loss:.6f}")
            print(f"Total games won: {games_won}")
            
            # Add replay buffer stats for prioritized replay
            if agent.use_prioritized_replay:
                print(f"Replay buffer size: {len(agent.memory)}")
                print(f"Current beta value: {agent.memory.beta:.4f}")
                print(f"Max priority: {agent.memory.max_priority:.4f}")
                
            print("=" * 40 + "\n")
        
        # Save model every 500 episodes
        if (e+1) % 500 == 0:
            model_path = os.path.join(save_dir, f'narde_model_{e+1}.pt')
            torch.save(agent.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'narde_model_final.pt')
    torch.save(agent.model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Print final statistics
    print("\n=== Final Training Statistics ===")
    print(f"Total episodes: {episodes}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total games won: {games_won}")
    print("=" * 40)
    
    # Close log file if it's open
    if log and log_file_handle:
        log_file_handle.close()
        print(f"Log saved to {log_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a DQN agent for Narde')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed statistics periodically')
    parser.add_argument('--verbose-steps', type=int, default=100, help='Print verbose stats every N steps')
    parser.add_argument('--log', action='store_true', help='Log detailed output to a file instead of displaying')
    parser.add_argument('--log-file', type=str, default='training_log.txt', help='File to log output to')
    parser.add_argument('--standard-network', action='store_true', help='Use standard DQN network instead of decomposed action space')
    parser.add_argument('--no-prioritized-replay', action='store_true', help='Disable prioritized experience replay')
    parser.add_argument('--alpha', type=float, default=0.6, help='Priority exponent alpha for prioritized replay (0=uniform, 1=greedy)')
    parser.add_argument('--beta', type=float, default=0.4, help='Initial importance sampling correction beta (0=no correction, 1=full correction)')
    parser.add_argument('--beta-increment', type=float, default=0.001, help='Amount to increase beta per update for annealing')
    
    args = parser.parse_args()
    
    # Print which optimizations are enabled
    print(f"{'Decomposed DQN':25}: {'ENABLED' if not args.standard_network else 'DISABLED'}")
    print(f"{'Prioritized Experience Replay':25}: {'ENABLED' if not args.no_prioritized_replay else 'DISABLED'}")
    
    # Configure prioritized experience replay parameters if using it
    if not args.no_prioritized_replay:
        # Monkey patch PrioritizedReplayBuffer before DQNAgent creation
        PrioritizedReplayBuffer.__init__.__defaults__ = (
            50000,  # capacity
            args.alpha,  # alpha
            args.beta,   # beta
            args.beta_increment,  # beta_increment
            0.01  # epsilon 
        )
        print(f"  - Alpha: {args.alpha} (higher = more prioritization)")
        print(f"  - Initial Beta: {args.beta} (starts low, anneals to 1.0)")
        print(f"  - Beta increment: {args.beta_increment} (per update)")
    
    main(
        episodes=args.episodes,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.lr,
        verbose=args.verbose,
        verbose_steps=args.verbose_steps,
        log=args.log,
        log_file=args.log_file,
        use_decomposed_network=not args.standard_network,
        use_prioritized_replay=not args.no_prioritized_replay
    )
