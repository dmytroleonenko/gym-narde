#!/usr/bin/env python3
"""
Optimized Narde Environment that selectively uses JAX acceleration
based on benchmark findings.

This environment uses:
- NumPy for simple operations (board rotation, individual moves)
- JAX for operations that benefit from batching and acceleration (block rule checks, neural net)
"""

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces
import time
from typing import Optional, Tuple, List, Dict, Any, Union

# Constants for the Narde game
BOARD_SIZE = 24
NUM_PIECES = 15
HOME_SIZE = 6


class OptimizedNardeEnv(gym.Env):
    """
    Narde environment that selectively uses JAX for operations that benefit from it.
    Simple operations use NumPy, while complex batched operations use JAX.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """Initialize the Narde environment."""
        self.observation_space = spaces.Box(
            low=-NUM_PIECES, high=NUM_PIECES, shape=(BOARD_SIZE,), dtype=np.int32
        )
        
        # Define 30 possible actions: 
        # - 4 possible dice combinations (1,2), (1,3), (2,3), (2,2)
        # - For each combination, there are multiple possible moves
        self.action_space = spaces.Discrete(30)
        
        # Track game state
        self.board = np.zeros(BOARD_SIZE, dtype=np.int32)
        self.player = 1  # Player 1 starts
        self.render_mode = render_mode
        self.dice_roll = None
        self.valid_actions = None
        
        # JAX-accelerated block rule check (compiled once)
        self._setup_jax_functions()
        
        # Reset the environment
        self.reset()
    
    def _setup_jax_functions(self):
        """Set up JAX-accelerated functions."""
        # Create JAX functions for operations that benefit from acceleration
        
        # Block rule check (used during batch simulations in MCTS)
        def check_block_rule_jax(board, player=1):
            """Check if player is violating the block rule (JAX version)."""
            # Use static masks for both home areas
            player1_home_mask = jnp.zeros_like(board, dtype=jnp.bool_)
            player1_home_mask = player1_home_mask.at[18:24].set(True)
            
            player2_home_mask = jnp.zeros_like(board, dtype=jnp.bool_)
            player2_home_mask = player2_home_mask.at[0:6].set(True)
            
            # Select the appropriate home mask based on player
            home_mask = jnp.where(player == 1, player1_home_mask, player2_home_mask)
            
            # Determine the opponent's piece sign
            opponent_sign = jnp.where(player == 1, -1, 1)
            
            # Create a mask for opponent's pieces (pieces with opponent's sign)
            is_opponent_piece = jnp.sign(board) == opponent_sign
            
            # Count opponent pieces in home
            opponent_pieces_in_home = jnp.sum(jnp.logical_and(is_opponent_piece, home_mask))
            
            # Count total opponent pieces
            opponent_total_pieces = jnp.sum(is_opponent_piece)
            
            # Check if all opponent pieces are in our home and there's at least one opponent piece
            return jnp.logical_and(
                opponent_pieces_in_home == opponent_total_pieces,
                opponent_total_pieces > 0
            )
            
        # Compile the function with JIT
        self._check_block_rule_jax = jax.jit(check_block_rule_jax)
        
        # Vectorized version for batch processing
        self._check_block_rule_jax_batch = jax.vmap(check_block_rule_jax, in_axes=(0, None))
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        
        # Initialize the board
        self.board = np.zeros(BOARD_SIZE, dtype=np.int32)
        
        # Set up player 1's pieces (positive values)
        self.board[0:HOME_SIZE] = np.array([3, 3, 3, 2, 2, 2], dtype=np.int32)
        
        # Set up player 2's pieces (negative values)
        self.board[BOARD_SIZE-HOME_SIZE:BOARD_SIZE] = np.array([-3, -3, -3, -2, -2, -2], dtype=np.int32)
        
        # Player 1 starts
        self.player = 1
        
        # Roll dice to start
        self.dice_roll = self._roll_dice()
        
        # Calculate valid actions
        self.valid_actions = self._get_valid_actions()
        
        return self._get_observation(), {"valid_actions": self.valid_actions}
    
    def _get_observation(self):
        """Get the current observation of the board state."""
        # For player 2, we flip the board and negate the values to keep the 
        # perspective consistent from the current player's view
        if self.player == 1:
            return self.board.copy()
        else:
            # Use NumPy for this simple operation (faster than JAX for single boards)
            return -np.flip(self.board)
    
    def _roll_dice(self):
        """Roll two dice and return the values."""
        dice1 = self.np_random.integers(1, 4)  # 1, 2, or 3
        dice2 = self.np_random.integers(1, 4)  # 1, 2, or 3
        return (dice1, dice2)
    
    def _check_block_rule_numpy(self, board, player=1):
        """
        Check if the player is violating the block rule.
        The block rule prevents a player from having all their opponent's pieces in their home.
        """
        # For player 1, home positions are 18-23
        # For player 2, home positions are 0-5
        home_positions = range(18, 24) if player == 1 else range(0, 6)
        
        # Count pieces of the opponent in our home
        opponent_val = -1 if player == 1 else 1
        opponent_pieces_in_home = sum(1 for pos in home_positions if board[pos] * opponent_val > 0)
        
        # Count total pieces of the opponent
        opponent_total_pieces = sum(1 for val in board if val * opponent_val > 0)
        
        # The block rule is violated if all of the opponent's pieces are in our home
        return opponent_pieces_in_home == opponent_total_pieces and opponent_total_pieces > 0
    
    def check_block_rule(self, board=None, player=None, use_jax=False):
        """
        Check if the player is violating the block rule.
        Will use JAX or NumPy depending on the use_jax flag.
        """
        # Use current board and player if not specified
        if board is None:
            board = self.board
        if player is None:
            player = self.player
            
        if use_jax:
            # Convert to JAX array if needed
            if isinstance(board, np.ndarray):
                board = jnp.array(board)
            return bool(self._check_block_rule_jax(board, player))
        else:
            # Use NumPy version for single board checks (more efficient)
            return self._check_block_rule_numpy(board, player)
    
    def check_block_rule_batch(self, boards, player=1):
        """
        Check the block rule for a batch of boards.
        Uses JAX for efficient batch processing.
        """
        # Convert to JAX array if needed
        if isinstance(boards, np.ndarray):
            boards = jnp.array(boards)
            
        # Use the vectorized JAX implementation
        results = self._check_block_rule_jax_batch(boards, player)
        return np.array(results)
    
    def _rotate_board(self, board=None):
        """Rotate the board for the next player's turn."""
        if board is None:
            board = self.board
            
        # Use NumPy for this simple operation (faster than JAX for single boards)
        return -np.flip(board)
    
    def _get_valid_actions(self):
        """Get all valid actions given the current dice roll."""
        # Simplified implementation - in a real environment, this would check 
        # all possible move combinations based on the dice roll
        valid_actions = []
        
        # For demonstration purposes, we'll just return a subset of actions
        # In a real implementation, this would calculate all legal moves
        for i in range(self.action_space.n):
            # Check if action i is valid based on the current board state and dice roll
            # This is a placeholder - actual implementation would be more complex
            if self._is_valid_action(i):
                valid_actions.append(i)
                
        return valid_actions
    
    def _is_valid_action(self, action):
        """Check if an action is valid."""
        # Simplified implementation - in reality this would validate the action
        # based on the dice roll and current board state
        return True  # Placeholder
    
    def _apply_action(self, action):
        """Apply an action to the board."""
        # Simplified implementation - in reality this would move the pieces
        # according to the action and dice roll
        # For demonstration, we'll just make a simple random change
        
        # Find a position with the current player's pieces
        player_pieces = np.where(self.board * self.player > 0)[0]
        if len(player_pieces) == 0:
            return False  # No pieces to move
            
        # Choose a random position to move from
        from_pos = self.np_random.choice(player_pieces)
        
        # Calculate a destination position
        # In a real game, this would be based on the dice roll and action
        to_pos = (from_pos + self.dice_roll[0]) % BOARD_SIZE
        
        # Move a piece from from_pos to to_pos
        self.board[from_pos] -= self.player  # Remove a piece from source
        self.board[to_pos] += self.player  # Add a piece to destination
        
        return True
    
    def step(self, action):
        """Take a step in the environment by applying an action."""
        if action not in self.valid_actions:
            # Invalid action - use a random valid action instead
            if self.valid_actions:
                action = self.np_random.choice(self.valid_actions)
            else:
                # No valid actions, end the game
                return self._get_observation(), -1, True, False, {"valid_actions": []}
        
        # Apply the action
        success = self._apply_action(action)
        
        # Check if the game is finished
        done = self._is_game_over()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # If not done, switch to the next player
        if not done:
            # Rotate the board for the next player's turn
            self.board = self._rotate_board()
            self.player = -self.player  # Switch player
            
            # Roll dice for the next player
            self.dice_roll = self._roll_dice()
            
            # Calculate valid actions for the next player
            self.valid_actions = self._get_valid_actions()
        else:
            # Game is over, no valid actions
            self.valid_actions = []
            
        # Return the observation, reward, done, truncated, and info
        return self._get_observation(), reward, done, False, {"valid_actions": self.valid_actions}
    
    def _is_game_over(self):
        """Check if the game is over."""
        # The game is over if all pieces of either player are in their home area
        # or if there are no valid moves
        
        # Check if player 1 has all pieces in home
        player1_pieces = np.sum(self.board > 0)
        player1_home_pieces = np.sum(self.board[18:24] > 0)
        
        # Check if player 2 has all pieces in home
        player2_pieces = np.sum(self.board < 0)
        player2_home_pieces = np.sum(self.board[0:6] < 0)
        
        # Game is over if either player has all pieces in home
        if player1_pieces > 0 and player1_pieces == player1_home_pieces:
            return True
        if player2_pieces > 0 and player2_pieces == player2_home_pieces:
            return True
            
        # Game is also over if there are no valid moves
        return len(self.valid_actions) == 0
    
    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        # For simplicity, the reward is the difference in the number of pieces in home
        # between the current player and the opponent
        
        if self.player == 1:
            player_home_pieces = np.sum(self.board[18:24] > 0)
            opponent_home_pieces = np.sum(self.board[0:6] < 0)
        else:
            player_home_pieces = np.sum(self.board[0:6] < 0)
            opponent_home_pieces = np.sum(self.board[18:24] > 0)
        
        return player_home_pieces - opponent_home_pieces
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # For human rendering, we would display the board graphically
            # This is a simplified ASCII rendering for demonstration
            print("Board state:")
            board_str = ""
            for i in range(BOARD_SIZE):
                val = self.board[i]
                if val > 0:
                    board_str += f" {val:2d}"
                elif val < 0:
                    board_str += f"{val:3d}"
                else:
                    board_str += "  ·"
                
                if i == 5:
                    board_str += " | "
                elif i == 11:
                    board_str += "\n"
                elif i == 17:
                    board_str += " | "
            print(board_str)
            print(f"Player {self.player}'s turn, Dice: {self.dice_roll}")
            print(f"Valid actions: {self.valid_actions}")
        
        # For rgb_array mode, we would return an image of the board
        # This is not implemented in this simplified version


# Example batch functions for MCTS or other simulation scenarios
def simulate_moves_batch(env, boards, player, actions, use_jax=True):
    """
    Simulate a batch of moves for MCTS or other batch operations.
    This demonstrates where JAX acceleration would be beneficial.
    
    Args:
        env: The Narde environment
        boards: A batch of board states to simulate from
        player: The player making the moves
        actions: The actions to apply to each board
        use_jax: Whether to use JAX acceleration
        
    Returns:
        new_boards: The resulting board states
        violates_block_rule: Whether each move violates the block rule
    """
    batch_size = len(boards)
    new_boards = []
    
    # Apply each action to the corresponding board
    # This would be done more efficiently in a real implementation
    for i in range(batch_size):
        board = boards[i].copy()
        # Apply action (simplified)
        # In a real implementation, this would use the actual game logic
        new_board = board.copy()
        new_boards.append(new_board)
    
    # Convert to numpy array
    new_boards = np.array(new_boards)
    
    # Check block rule for all boards efficiently with JAX
    if use_jax and batch_size >= 128:  # Only use JAX for larger batches
        violates_block_rule = env.check_block_rule_batch(new_boards, player)
    else:
        # For smaller batches, NumPy is more efficient
        violates_block_rule = np.array([
            env._check_block_rule_numpy(board, player) 
            for board in new_boards
        ])
    
    return new_boards, violates_block_rule 

# Add numpy batch simulation function for the MuZero interface
def simulate_moves_batch_numpy(boards, actions, dice=None):
    """
    Simulate a batch of moves using NumPy (no JAX).
    This function is used when batch sizes are small and JAX would be slower.
    
    Args:
        boards: Batch of board states as numpy array of shape (batch_size, 24)
        actions: Batch of actions as numpy array of shape (batch_size,)
        dice: Optional batch of dice rolls as numpy array of shape (batch_size, 2)
        
    Returns:
        next_boards: Batch of next board states
        rewards: Batch of rewards
        dones: Batch of done flags
    """
    batch_size = boards.shape[0]
    next_boards = np.zeros_like(boards)
    rewards = np.zeros(batch_size)
    dones = np.zeros(batch_size, dtype=bool)
    
    # Create environment once for simulation
    env = OptimizedNardeEnv()
    
    # Process each board-action pair
    for i in range(batch_size):
        # Set the board state
        env.board = boards[i].copy()
        
        # Set dice if provided
        if dice is not None:
            env.dice = dice[i].copy()
            
        # Execute the action - handle new Gym return format
        next_board, reward, terminated, truncated, _ = env.step(actions[i])
        done = terminated or truncated
        
        # Store results
        next_boards[i] = next_board
        rewards[i] = reward
        dones[i] = done
    
    return next_boards, rewards, dones 