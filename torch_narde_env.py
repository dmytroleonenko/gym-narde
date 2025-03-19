#!/usr/bin/env python3
"""
PyTorch implementation of the Narde environment with MPS acceleration.
This implementation is optimized for hardware acceleration on Mac M-series chips.
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from typing import Tuple, List, Dict, Optional, Union, Any

# Determine the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) device for Narde environment")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for Narde environment")
else:
    device = torch.device("cpu")
    print("Using CPU device for Narde environment")

# CPU device for operations that might be faster on CPU or aren't supported on MPS
cpu_device = torch.device("cpu")

# Batch size threshold for using hardware acceleration (based on benchmarks)
BATCH_SIZE_THRESHOLD = 512

class TorchNardeEnv(gym.Env):
    """
    PyTorch implementation of the Narde environment with hardware acceleration.
    This implementation uses PyTorch tensors and operations optimized for GPU/MPS execution.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, use_acceleration: bool = True):
        """
        Initialize the Narde environment.
        
        Args:
            render_mode: The rendering mode ('human' or 'rgb_array')
            use_acceleration: Whether to use hardware acceleration (MPS/CUDA) when available
        """
        self.render_mode = render_mode
        self.use_acceleration = use_acceleration
        self.device = device if use_acceleration else cpu_device
        
        # Board dimensions
        self.board_width = 8
        self.board_height = 3
        self.board_size = self.board_width * self.board_height
        
        # Action space: source position (24) * dice value (6) * direction (2) = 288 possible actions
        # We'll filter valid actions during the get_valid_actions step
        self.action_space = spaces.Discrete(288)
        
        # State representation: 24 board positions, each can have:
        # 0: empty, 1-15: player 1 pieces, -1 to -15: player 2 pieces
        # Plus additional state: current player, dice values
        # Board: 24 positions
        # Current player: 1
        # Dice values: 2 values
        # Pieces captured: 2 values (one per player)
        # Game phase: 1 value (0: placing, 1: moving)
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=-15, high=15, shape=(24,), dtype=np.int8),
            'current_player': spaces.Discrete(2),
            'dice': spaces.Box(low=1, high=6, shape=(2,), dtype=np.int8),
            'captured': spaces.Box(low=0, high=15, shape=(2,), dtype=np.int8),
            'phase': spaces.Discrete(2)
        })
        
        # Initialize game state
        self.reset()
        
        # Precompute direction vectors for moving pieces
        # Direction 0: forward (player 1)
        # Direction 1: backward (player 2)
        self.directions = torch.tensor([1, -1], device=self.device)
        
        # Precompute board rotation indices for each rotation amount (1-6)
        self.rotation_indices = {}
        board_indices = torch.arange(24, device=self.device)
        for rotation in range(1, 7):
            rotated_indices = (board_indices + rotation) % 24
            self.rotation_indices[rotation] = rotated_indices
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        
        # Initialize the board with zeros (empty)
        self.board = torch.zeros(24, dtype=torch.int8, device=self.device)
        
        # Current player: 0 for player 1, 1 for player 2
        self.current_player = 0
        
        # Roll dice
        self.dice = self._roll_dice()
        
        # Initialize captured pieces
        self.captured = torch.zeros(2, dtype=torch.int8, device=self.device)
        
        # Game phase: 0 for placing phase, 1 for moving phase
        self.phase = 0
        
        # Turn counter
        self.turn_count = 0
        
        # Game over flag
        self.game_over = False
        
        # Convert to observation format
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _roll_dice(self) -> torch.Tensor:
        """Roll two dice and return their values."""
        if hasattr(self, 'np_random'):
            dice = torch.tensor(
                self.np_random.integers(1, 7, size=2), 
                dtype=torch.int8, 
                device=self.device
            )
        else:
            # Fallback if np_random is not initialized
            dice = torch.tensor(
                [np.random.randint(1, 7), np.random.randint(1, 7)], 
                dtype=torch.int8, 
                device=self.device
            )
        return dice
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Convert the internal state to the observation format."""
        # Convert tensors to numpy arrays for the gym observation
        return {
            'board': self.board.cpu().numpy(),
            'current_player': np.array(self.current_player, dtype=np.int8),
            'dice': self.dice.cpu().numpy(),
            'captured': self.captured.cpu().numpy(),
            'phase': np.array(self.phase, dtype=np.int8)
        }
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.
        
        Args:
            action: Action index from the action space
            
        Returns:
            observation: The new observation
            reward: The reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        if self.game_over:
            # If game is over, return the final observation
            observation = self._get_observation()
            return observation, 0.0, True, False, {}
        
        # Check if action is valid
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # Invalid action, return the current observation with a penalty
            observation = self._get_observation()
            return observation, -1.0, False, False, {'valid_actions': valid_actions}
        
        # Decode action
        source_pos, dice_value, direction = self._decode_action(action)
        
        # Apply action
        reward = self._apply_action(source_pos, dice_value, direction)
        
        # Check game termination
        terminated = self._check_game_over()
        
        # Get the new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'valid_actions': self.get_valid_actions(),
            'turn_count': self.turn_count
        }
        
        return observation, reward, terminated, False, info
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode an action index into source position, dice value, and direction.
        
        Args:
            action: Action index from 0 to 287
            
        Returns:
            source_pos: Source position (0-23)
            dice_value: Dice value (0-5 corresponding to 1-6)
            direction: Direction (0: forward, 1: backward)
        """
        source_pos = action // 12
        remainder = action % 12
        dice_value = remainder // 2
        direction = remainder % 2
        return source_pos, dice_value + 1, direction
    
    def _apply_action(self, source_pos: int, dice_value: int, direction: int) -> float:
        """
        Apply an action to the game state.
        
        Args:
            source_pos: Source position (0-23)
            dice_value: Dice value (1-6)
            direction: Direction (0: forward, 1: backward)
            
        Returns:
            reward: Reward for the action
        """
        reward = 0.0
        
        # Get the direction modifier based on player and direction
        dir_mod = self.directions[direction]
        if self.current_player == 1:
            dir_mod = -dir_mod  # Reverse for player 2
        
        # Calculate target position
        target_pos = (source_pos + dir_mod * dice_value) % 24
        
        # Check if the source has the current player's pieces
        source_pieces = self.board[source_pos].item()
        is_current_player_piece = (source_pieces > 0 and self.current_player == 0) or \
                                (source_pieces < 0 and self.current_player == 1)
        
        if not is_current_player_piece:
            return -1.0  # Invalid move
        
        # Get the absolute number of pieces
        num_pieces = abs(source_pieces)
        
        # Check the target position
        target_pieces = self.board[target_pos].item()
        
        # If the target has opponent's pieces
        if (target_pieces < 0 and self.current_player == 0) or \
           (target_pieces > 0 and self.current_player == 1):
            # Single piece can be captured
            if abs(target_pieces) == 1:
                # Capture opponent's piece
                if self.current_player == 0:
                    self.captured[1] += 1  # Player 1 captures Player 2's piece
                    self.board[target_pos] = torch.tensor(1, dtype=torch.int8, device=self.device)
                else:
                    self.captured[0] += 1  # Player 2 captures Player 1's piece
                    self.board[target_pos] = torch.tensor(-1, dtype=torch.int8, device=self.device)
                
                # Decrement source position
                if source_pieces > 0:
                    self.board[source_pos] = torch.tensor(source_pieces - 1, dtype=torch.int8, device=self.device)
                else:
                    self.board[source_pos] = torch.tensor(source_pieces + 1, dtype=torch.int8, device=self.device)
                
                reward = 0.1  # Reward for capturing
            else:
                # Cannot capture multiple pieces
                return -1.0  # Invalid move
        else:
            # Target is empty or has player's own pieces
            if target_pieces == 0:
                # Move to empty space
                if source_pieces > 0:
                    self.board[target_pos] = torch.tensor(1, dtype=torch.int8, device=self.device)
                    self.board[source_pos] = torch.tensor(source_pieces - 1, dtype=torch.int8, device=self.device)
                else:
                    self.board[target_pos] = torch.tensor(-1, dtype=torch.int8, device=self.device)
                    self.board[source_pos] = torch.tensor(source_pieces + 1, dtype=torch.int8, device=self.device)
            elif (target_pieces > 0 and self.current_player == 0) or \
                 (target_pieces < 0 and self.current_player == 1):
                # Stack with own pieces (up to 15)
                if self.current_player == 0:
                    # Check if stacking would exceed 15
                    if target_pieces + 1 <= 15:
                        self.board[target_pos] = torch.tensor(target_pieces + 1, dtype=torch.int8, device=self.device)
                        self.board[source_pos] = torch.tensor(source_pieces - 1, dtype=torch.int8, device=self.device)
                    else:
                        return -1.0  # Invalid move (would exceed max stack)
                else:
                    # Check if stacking would exceed 15
                    if target_pieces - 1 >= -15:
                        self.board[target_pos] = torch.tensor(target_pieces - 1, dtype=torch.int8, device=self.device)
                        self.board[source_pos] = torch.tensor(source_pieces + 1, dtype=torch.int8, device=self.device)
                    else:
                        return -1.0  # Invalid move (would exceed max stack)
        
        # Check if the move was successful
        if reward >= 0:
            # Switch player
            self.current_player = 1 - self.current_player
            
            # Roll dice for the next player
            self.dice = self._roll_dice()
            
            # Increment turn counter
            self.turn_count += 1
            
            # Switch to moving phase after certain number of turns
            if self.turn_count >= 24 and self.phase == 0:
                self.phase = 1
        
        return reward
    
    def _check_game_over(self) -> bool:
        """Check if the game is over and return termination status."""
        # Count pieces for each player
        player1_pieces = torch.sum(torch.where(self.board > 0, self.board, torch.tensor(0, device=self.device)))
        player2_pieces = torch.sum(torch.where(self.board < 0, torch.abs(self.board), torch.tensor(0, device=self.device)))
        
        # Add captured pieces
        player1_total = player1_pieces + self.captured[0]
        player2_total = player2_pieces + self.captured[1]
        
        # Check win conditions (one player has no pieces left)
        if player1_total == 0 or player2_total == 0:
            self.game_over = True
            return True
        
        # Check for draw (maximum turns reached or stalemate)
        if self.turn_count >= 1000:
            self.game_over = True
            return True
        
        return False
    
    def get_valid_actions(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Get all valid actions in the current state.
        
        Args:
            batch_size: Optional batch size for processing. If provided and larger than the threshold,
                        will use hardware acceleration if available.
                        
        Returns:
            valid_actions: Tensor of valid action indices
        """
        # Choose device based on batch size if provided
        compute_device = self.device
        if batch_size is not None:
            if batch_size < BATCH_SIZE_THRESHOLD:
                compute_device = cpu_device
        
        # List to collect valid actions
        valid_actions = []
        
        # Current player's pieces have positive values for player 1, negative for player 2
        is_player1 = self.current_player == 0
        
        # Get positions with current player's pieces
        if is_player1:
            positions = torch.nonzero(self.board > 0).squeeze(-1)
        else:
            positions = torch.nonzero(self.board < 0).squeeze(-1)
        
        # Convert to CPU for easier iteration
        positions_cpu = positions.cpu().numpy()
        dice_cpu = self.dice.cpu().numpy()
        
        # For each position with player's pieces
        for pos in positions_cpu:
            source_pieces = abs(self.board[pos].item())
            if source_pieces == 0:
                continue
                
            # For each dice value
            for dice_idx, dice_value in enumerate(dice_cpu):
                # For each direction
                for direction in [0, 1]:  # 0: forward, 1: backward
                    # Get direction modifier
                    dir_mod = self.directions[direction].item()
                    if self.current_player == 1:
                        dir_mod = -dir_mod  # Reverse for player 2
                    
                    # Calculate target position
                    target_pos = (pos + dir_mod * dice_value) % 24
                    
                    # Check the target position
                    target_pieces = self.board[target_pos].item()
                    
                    # Check if move is valid
                    is_valid = False
                    
                    # If target is empty
                    if target_pieces == 0:
                        is_valid = True
                    # If target has current player's pieces
                    elif (target_pieces > 0 and is_player1) or (target_pieces < 0 and not is_player1):
                        # Check if adding a piece would exceed the max stack (15)
                        if is_player1 and target_pieces + 1 <= 15:
                            is_valid = True
                        elif not is_player1 and target_pieces - 1 >= -15:
                            is_valid = True
                    # If target has opponent's pieces (potential capture)
                    elif (target_pieces < 0 and is_player1) or (target_pieces > 0 and not is_player1):
                        # Can only capture single pieces
                        if abs(target_pieces) == 1:
                            is_valid = True
                    
                    # Add valid action to the list
                    if is_valid:
                        # Encode action: source position, dice value, direction
                        action = pos * 12 + (dice_value - 1) * 2 + direction
                        valid_actions.append(action)
        
        # Convert to tensor on the specified device
        if not valid_actions:
            # No valid actions, return empty tensor
            return torch.tensor([], dtype=torch.int64, device=compute_device)
        
        return torch.tensor(valid_actions, dtype=torch.int64, device=compute_device)
    
    def rotate_board(self, board: torch.Tensor, rotation: int) -> torch.Tensor:
        """
        Rotate the board by the specified amount.
        This operation is optimized for hardware acceleration.
        
        Args:
            board: Board tensor to rotate (batch_size, 24) or (24,)
            rotation: Rotation amount (1-6)
            
        Returns:
            Rotated board tensor
        """
        # Determine if we have a batch or a single board
        batch_mode = len(board.shape) > 1
        batch_size = board.shape[0] if batch_mode else 0
        
        # Choose device based on batch size
        compute_device = self.device
        if not batch_mode or batch_size < BATCH_SIZE_THRESHOLD:
            compute_device = cpu_device
        
        # Move tensor to compute device if needed
        if board.device != compute_device:
            board = board.to(compute_device)
        
        # Get rotation indices
        indices = self.rotation_indices.get(rotation)
        if indices is None or indices.device != compute_device:
            # Compute and cache rotation indices
            board_indices = torch.arange(24, device=compute_device)
            indices = (board_indices + rotation) % 24
            self.rotation_indices[rotation] = indices
        
        # Apply rotation
        if batch_mode:
            # For batched boards: rotate each board in the batch
            rotated = torch.index_select(board, 1, indices)
        else:
            # For a single board
            rotated = torch.index_select(board, 0, indices)
        
        return rotated
    
    def check_block_rule(self, board: torch.Tensor, batch_mode: bool = False) -> torch.Tensor:
        """
        Check if the block rule is violated (6 consecutive positions occupied by same player).
        This implementation is optimized for hardware acceleration.
        
        Args:
            board: Board tensor to check
            batch_mode: Whether board is a batch of boards
            
        Returns:
            Boolean tensor indicating if block rule is violated for each player
            [player1_violation, player2_violation] or batch of these
        """
        # Choose device based on batch size
        compute_device = self.device
        if not batch_mode or board.shape[0] < BATCH_SIZE_THRESHOLD:
            compute_device = cpu_device
        
        # Move tensor to compute device if needed
        if board.device != compute_device:
            board = board.to(compute_device)
        
        # Initialize result tensors
        if batch_mode:
            batch_size = board.shape[0]
            result = torch.zeros((batch_size, 2), dtype=torch.bool, device=compute_device)
        else:
            result = torch.zeros(2, dtype=torch.bool, device=compute_device)
        
        # Define window size for checking consecutive positions
        window_size = 6
        
        # Vectorized implementation
        if batch_mode:
            # For batched boards
            for start_pos in range(24):
                # Get positions in the window
                positions = [(start_pos + i) % 24 for i in range(window_size)]
                
                # Check for player 1 (positive values)
                p1_window = board[:, positions]
                p1_violation = (p1_window > 0).all(dim=1)
                result[:, 0] = result[:, 0] | p1_violation
                
                # Check for player 2 (negative values)
                p2_window = board[:, positions]
                p2_violation = (p2_window < 0).all(dim=1)
                result[:, 1] = result[:, 1] | p2_violation
        else:
            # For a single board
            for start_pos in range(24):
                # Get positions in the window
                positions = [(start_pos + i) % 24 for i in range(window_size)]
                
                # Check for player 1 (positive values)
                p1_window = board[positions]
                p1_violation = (p1_window > 0).all()
                result[0] = result[0] | p1_violation
                
                # Check for player 2 (negative values)
                p2_window = board[positions]
                p2_violation = (p2_window < 0).all()
                result[1] = result[1] | p2_violation
        
        return result
    
    def render(self):
        """Render the game state."""
        if self.render_mode is None:
            return
        
        # Implement rendering logic here
        # For 'human' mode, print the board to console
        if self.render_mode == "human":
            board_np = self.board.cpu().numpy()
            print(f"\nCurrent board state (Turn {self.turn_count}):")
            print(f"Current player: {'Player 1' if self.current_player == 0 else 'Player 2'}")
            print(f"Dice: {self.dice.cpu().numpy()}")
            print(f"Captured: Player 1 - {self.captured[0].item()}, Player 2 - {self.captured[1].item()}")
            print("  " + " ".join([f"{i:2d}" for i in range(8)]))
            
            # Print the board rows
            for row in range(3):
                row_str = f"{row} "
                for col in range(8):
                    pos = row * 8 + col
                    piece = board_np[pos]
                    if piece > 0:
                        row_str += f"+{piece:1d} "
                    elif piece < 0:
                        row_str += f"{piece:2d} "
                    else:
                        row_str += " . "
                print(row_str)
            
            # Print valid actions
            valid_actions = self.get_valid_actions().cpu().numpy()
            print(f"Valid actions: {valid_actions}")
        
        # For 'rgb_array' mode, return a visualization (not implemented here)
        elif self.render_mode == "rgb_array":
            # Implement RGB array rendering (for machine learning visualization)
            pass
    
    def close(self):
        """Clean up resources."""
        pass


# Example usage
if __name__ == "__main__":
    # Create the environment
    env = TorchNardeEnv(render_mode="human")
    
    # Reset the environment
    obs, info = env.reset()
    
    # Play a few random steps
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100:
        # Get valid actions
        valid_actions = env.get_valid_actions().cpu().numpy()
        
        if len(valid_actions) > 0:
            # Choose a random valid action
            action = np.random.choice(valid_actions)
            
            # Take a step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render
            env.render()
            
            # Check termination
            done = terminated or truncated
        else:
            print("No valid actions available!")
            break
        
        steps += 1
    
    print(f"Game ended after {steps} steps with total reward {total_reward}")
    env.close() 