"""
Vectorized Environment for Narde MuZero training.
This implementation processes multiple environments in parallel for efficient batch operations.
"""

import numpy as np
import torch
import gym_narde
import gymnasium as gym
from typing import List, Tuple, Dict, Any, Optional, Union


class VectorizedNardeEnv:
    """
    Vectorized Narde Environment that maintains multiple environments 
    and enables batched operations for MuZero training.
    """
    def __init__(self, num_envs: int = 16, device: str = "cpu"):
        """
        Initialize a set of parallel environments.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to store tensors on ("cpu", "cuda", "mps")
        """
        self.num_envs = num_envs
        self.device = device
        
        # Create environments
        self.envs = [gym.make('Narde-v0') for _ in range(num_envs)]
        
        # Placeholders for environment state
        self.observations = None
        self.dones = np.zeros(num_envs, dtype=bool)
        self.infos = [{} for _ in range(num_envs)]
        
        # Track environment status
        self.active_envs = list(range(num_envs))
        
        # Initialize all environments
        self.reset()
        
        # Precompute common tensors for efficiency
        self._setup_cached_tensors()
        
    def _setup_cached_tensors(self):
        """Setup commonly used tensors to avoid recreating them."""
        # For board rotation operations
        self.rotation_indices = {
            # Precomputed permutation indices for rotating the board
            1: torch.tensor([23-i for i in range(24)], device=self.device),  # 180 degree rotation
        }
        # Add more cached tensors as needed
        
    def reset(self, env_indices: List[int] = None) -> np.ndarray:
        """
        Reset specified environments, or all if none specified.
        
        Args:
            env_indices: Indices of environments to reset (None = all)
            
        Returns:
            Batch of observations
        """
        if env_indices is None:
            env_indices = list(range(self.num_envs))
            
        # Reset each specified environment
        for idx in env_indices:
            obs, info = self.envs[idx].reset()
            
            # Initialize observations array if needed
            if self.observations is None:
                obs_shape = (self.num_envs,) + obs.shape
                self.observations = np.zeros(obs_shape, dtype=np.float32)
                
            # Store observation and info
            self.observations[idx] = obs
            self.infos[idx] = info
            self.dones[idx] = False
            
        # Make sure the environment is in the active list
        self.active_envs = sorted(list(set(self.active_envs) | set(env_indices)))
            
        return self.observations.copy()
    
    def step(self, actions: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all active environments with the given actions.
        
        Args:
            actions: List of actions for each active environment
            
        Returns:
            (observations, rewards, dones, infos)
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        
        # Step each active environment
        for i, env_idx in enumerate(self.active_envs):
            if i < len(actions):  # Make sure we have an action for this env
                action = actions[i]
                
                # Step the environment
                obs, reward, done, truncated, info = self.envs[env_idx].step(action)
                
                # Store results
                self.observations[env_idx] = obs
                rewards[env_idx] = reward
                self.dones[env_idx] = done or truncated
                self.infos[env_idx] = info
                
                # Auto-reset environments that are done
                if done or truncated:
                    self.reset([env_idx])
        
        # Update active environments list (remove any that are done)
        self.active_envs = [idx for idx in self.active_envs if not self.dones[idx]]
        
        return self.observations.copy(), rewards, self.dones.copy(), self.infos.copy()
    
    def get_valid_actions_batch(self) -> List[List[int]]:
        """
        Get valid actions for all active environments.
        
        Returns:
            List of valid action lists for each active environment
        """
        valid_actions_batch = []
        
        for env_idx in self.active_envs:
            # Get the valid moves from the environment
            unwrapped_env = self.envs[env_idx].unwrapped
            valid_moves = unwrapped_env.game.get_valid_moves()
            
            # Convert moves to action indices
            valid_actions = []
            for move in valid_moves:
                if move[1] == 'off':
                    # Bear off move (from_pos, 'off')
                    action_idx = move[0] * 24
                else:
                    # Regular move (from_pos, to_pos)
                    action_idx = move[0] * 24 + move[1]
                valid_actions.append(action_idx)
                
            valid_actions_batch.append(valid_actions)
            
        return valid_actions_batch
    
    def encode_actions(self, moves: List[Tuple]) -> List[Tuple[int, int]]:
        """
        Convert (from_pos, to_pos) or (from_pos, 'off') moves to action tuples.
        
        Args:
            moves: List of moves [(from_pos, to_pos), ...] or [(from_pos, 'off'), ...]
            
        Returns:
            List of (action_idx, move_type) tuples
        """
        actions = []
        for move in moves:
            if move[1] == 'off':
                # Bear off move
                action_idx = move[0] * 24
                move_type = 1
            else:
                # Regular move
                action_idx = move[0] * 24 + move[1]
                move_type = 0
            actions.append((action_idx, move_type))
        return actions
    
    def decode_actions(self, actions: List[Tuple[int, int]]) -> List[Tuple]:
        """
        Convert action tuples to (from_pos, to_pos) or (from_pos, 'off') moves.
        
        Args:
            actions: List of (action_idx, move_type) tuples
            
        Returns:
            List of moves [(from_pos, to_pos), ...] or [(from_pos, 'off'), ...]
        """
        moves = []
        for action_idx, move_type in actions:
            from_pos = action_idx // 24
            if move_type == 1:
                # Bear off move
                moves.append((from_pos, 'off'))
            else:
                # Regular move
                to_pos = action_idx % 24
                moves.append((from_pos, to_pos))
        return moves
    
    def rotate_boards_batch(self, boards: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Rotate multiple boards efficiently (for player 2's perspective).
        
        Args:
            boards: Batch of board states [batch_size, 24]
            
        Returns:
            Rotated boards with negated piece values
        """
        if isinstance(boards, np.ndarray):
            # NumPy implementation
            rotated_boards = -boards[:, ::-1]
            return rotated_boards
        elif isinstance(boards, torch.Tensor):
            # PyTorch implementation with precalculated indices
            if boards.device != self.device:
                boards = boards.to(self.device)
                
            # Use cached indices for more efficient rotation
            rotated_boards = -boards.index_select(1, self.rotation_indices[1])
            return rotated_boards
        else:
            raise TypeError(f"Unsupported type: {type(boards)}")
    
    def get_observations_tensor(self) -> torch.Tensor:
        """
        Get all observations as a PyTorch tensor.
        
        Returns:
            Tensor of all observations [num_envs, obs_dim]
        """
        return torch.tensor(self.observations, dtype=torch.float32, device=self.device)
    
    def get_active_observations_tensor(self) -> torch.Tensor:
        """
        Get observations from active environments as a PyTorch tensor.
        
        Returns:
            Tensor of active observations [num_active_envs, obs_dim]
        """
        active_obs = self.observations[self.active_envs]
        return torch.tensor(active_obs, dtype=torch.float32, device=self.device)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
            
    def __len__(self):
        """Return the number of environments."""
        return self.num_envs


# Helper functions for using the vectorized environment

def create_vectorized_env(num_envs=16, device="cpu") -> VectorizedNardeEnv:
    """
    Create a vectorized Narde environment.
    
    Args:
        num_envs: Number of parallel environments
        device: Device to use for tensor operations
        
    Returns:
        Vectorized environment instance
    """
    return VectorizedNardeEnv(num_envs=num_envs, device=device)


def batch_inference_step(env: VectorizedNardeEnv, network, explorer="random", num_simulations=10, batch_size=16):
    """
    Run one step of batch inference on all active environments.
    
    Args:
        env: Vectorized environment
        network: MuZero network
        explorer: Exploration strategy ("random", "mcts", etc.)
        num_simulations: Number of MCTS simulations if using MCTS
        batch_size: Batch size for MCTS simulations
        
    Returns:
        List of action tuples to take in each active environment
    """
    from muzero.mcts_batched import run_batched_mcts
    
    # Get valid actions for all active environments
    valid_actions_batch = env.get_valid_actions_batch()
    
    # Get observations as tensor
    observations = env.get_active_observations_tensor()
    
    # Choose actions based on explorer type
    actions = []
    
    if explorer == "random":
        # Random actions for each environment
        for i, valid_actions in enumerate(valid_actions_batch):
            if valid_actions:
                action_idx = np.random.choice(valid_actions)
                # Determine if it's a bear-off move
                move_type = 1 if action_idx % 24 == 0 else 0  # Simplified check
                actions.append((action_idx, move_type))
            else:
                # No valid actions, return dummy action (will be skipped)
                actions.append((0, 0))
    
    elif explorer == "mcts":
        # Use MCTS to select actions
        for i, valid_actions in enumerate(valid_actions_batch):
            if valid_actions:
                # Run batched MCTS for this observation
                policy = run_batched_mcts(
                    observation=observations[i],
                    network=network,
                    valid_actions=valid_actions,
                    num_simulations=num_simulations,
                    device=env.device,
                    batch_size=batch_size
                )
                
                # Select action with highest policy probability
                action_idx = np.argmax([policy[a] for a in valid_actions])
                actual_action = valid_actions[action_idx]
                
                # Determine if it's a bear-off move
                # Would need more complex logic with actual environment access
                move_type = 0  # Default to regular move
                from_pos = actual_action // 24
                
                # Check if this move should be a bear-off
                for move in env.envs[env.active_envs[i]].unwrapped.game.get_valid_moves():
                    if move[0] == from_pos and move[1] == 'off':
                        move_type = 1
                        break
                        
                actions.append((actual_action, move_type))
            else:
                # No valid actions, return dummy action (will be skipped)
                actions.append((0, 0))
    
    return actions 