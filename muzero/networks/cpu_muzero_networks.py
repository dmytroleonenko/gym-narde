"""
CPU-compatible implementation of MuZero networks.

This module provides a NumPy-based implementation of MuZero networks that can run on any CPU
without requiring JAX or hardware acceleration. This is useful for testing and development
on platforms where JAX is not fully supported, such as Apple Silicon.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List


class CPUMuZeroNetworks:
    """CPU-compatible implementation of MuZero networks using NumPy."""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_actions: int,
        min_value: float = -1.0,
        max_value: float = 1.0,
        categorical_size: int = 51,
        seed: int = 42
    ):
        """Initialize MuZero networks.
        
        Args:
            input_dim: Dimension of input state
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output state representation
            num_actions: Number of possible actions
            min_value: Minimum value for categorical representation
            max_value: Maximum value for categorical representation
            categorical_size: Number of bins in categorical distribution
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_actions = num_actions
        self.min_value = min_value
        self.max_value = max_value
        self.categorical_size = categorical_size
        
        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        
        # Initialize network parameters with simple Xavier initialization
        # We don't implement the actual network functionality here - this is just a placeholder
        # for testing purposes
        self.params = {
            "rep_net": self._init_network_params([input_dim, hidden_dim, hidden_dim, output_dim]),
            "dyn_net": self._init_network_params([output_dim + num_actions, hidden_dim, hidden_dim, output_dim]),
            "rew_net": self._init_network_params([output_dim + num_actions, hidden_dim, 1]),
            "val_net": self._init_network_params([output_dim, hidden_dim, 1]),
            "pol_net": self._init_network_params([output_dim, hidden_dim, num_actions]),
        }
    
    def _init_network_params(self, layer_sizes):
        """Initialize parameters for a feed-forward network."""
        params = []
        for i in range(len(layer_sizes) - 1):
            input_size, output_size = layer_sizes[i], layer_sizes[i+1]
            scale = np.sqrt(2.0 / (input_size + output_size))
            params.append({
                "weights": self.rng.normal(0, scale, (input_size, output_size)),
                "biases": np.zeros(output_size)
            })
        return params
    
    def scalar_to_categorical(self, x: np.ndarray) -> np.ndarray:
        """Convert scalar to categorical distribution.
        
        Args:
            x: Scalar values to convert, shape (...,)
            
        Returns:
            Categorical distributions, shape (..., categorical_size)
        """
        x = np.clip(x, self.min_value, self.max_value)
        # Scale to [0, 1]
        x = (x - self.min_value) / (self.max_value - self.min_value)
        # Scale to [0, categorical_size - 1]
        x = x * (self.categorical_size - 1)
        # Floor and convert to int
        lower_idx = np.floor(x).astype(np.int32)
        upper_idx = np.minimum(lower_idx + 1, self.categorical_size - 1)
        lower_weight = upper_idx - x
        upper_weight = x - lower_idx
        
        # Create categorical distribution
        probs = np.zeros(x.shape + (self.categorical_size,), dtype=np.float32)
        
        # Handle batched or unbatched inputs
        if len(x.shape) == 0:
            probs[lower_idx] = lower_weight
            probs[upper_idx] = upper_weight
        elif len(x.shape) == 1:
            for i in range(x.shape[0]):
                probs[i, lower_idx[i]] = lower_weight[i]
                probs[i, upper_idx[i]] = upper_weight[i]
        else:
            # Reshape to handle arbitrary batch dimensions
            flat_x = x.reshape(-1)
            flat_lower_idx = lower_idx.reshape(-1)
            flat_upper_idx = upper_idx.reshape(-1)
            flat_lower_weight = lower_weight.reshape(-1)
            flat_upper_weight = upper_weight.reshape(-1)
            flat_probs = probs.reshape(-1, self.categorical_size)
            
            for i in range(flat_x.shape[0]):
                flat_probs[i, flat_lower_idx[i]] = flat_lower_weight[i]
                flat_probs[i, flat_upper_idx[i]] = flat_upper_weight[i]
            
            probs = flat_probs.reshape(x.shape + (self.categorical_size,))
            
        return probs
    
    def categorical_to_scalar(self, probs: np.ndarray) -> np.ndarray:
        """Convert categorical distribution to scalar.
        
        Args:
            probs: Categorical distributions, shape (..., categorical_size)
            
        Returns:
            Scalar values, shape (...)
        """
        support = np.linspace(self.min_value, self.max_value, self.categorical_size)
        # Handle arbitrary batch dimensions
        batch_dims = probs.shape[:-1]
        result = np.sum(probs * support.reshape((1,) * len(batch_dims) + (-1,)), axis=-1)
        return result
    
    def compute_muzero_loss(
        self,
        value_targets: np.ndarray,  # Shape (batch_size, num_unroll_steps + 1)
        reward_targets: np.ndarray,  # Shape (batch_size, num_unroll_steps)
        policy_targets: np.ndarray,  # Shape (batch_size, num_unroll_steps + 1, num_actions)
        value_preds: np.ndarray,    # Shape (batch_size, num_unroll_steps + 1)
        reward_preds: np.ndarray,   # Shape (batch_size, num_unroll_steps)
        policy_preds: np.ndarray,   # Shape (batch_size, num_unroll_steps + 1, num_actions)
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the MuZero loss function.
        
        Args:
            value_targets: Target values
            reward_targets: Target rewards
            policy_targets: Target policies
            value_preds: Predicted values
            reward_preds: Predicted rewards
            policy_preds: Predicted policies
            weights: Weights for each loss component
            
        Returns:
            Total loss and dictionary of loss components
        """
        # Default weights if not provided
        if weights is None:
            weights = {
                "value": 1.0,
                "reward": 1.0,
                "policy": 1.0
            }
        
        # MSE loss for value and reward
        value_loss = np.mean((value_targets - value_preds) ** 2)
        reward_loss = np.mean((reward_targets - reward_preds) ** 2)
        
        # Cross-entropy loss for policy
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        policy_preds_clipped = np.clip(policy_preds, epsilon, 1.0 - epsilon)
        policy_loss = -np.mean(np.sum(policy_targets * np.log(policy_preds_clipped), axis=-1))
        
        # Compute total loss
        total_loss = (
            weights["value"] * value_loss +
            weights["reward"] * reward_loss +
            weights["policy"] * policy_loss
        )
        
        # Return loss components
        loss_dict = {
            "total": float(total_loss),
            "value": float(value_loss),
            "reward": float(reward_loss),
            "policy": float(policy_loss)
        }
        
        return total_loss, loss_dict
    
    def compute_n_step_returns(
        self,
        rewards: np.ndarray,  # Shape (batch_size, sequence_length)
        values: np.ndarray,   # Shape (batch_size, sequence_length + 1)
        dones: Optional[np.ndarray] = None,  # Shape (batch_size, sequence_length)
        discount: float = 0.99,
        n_steps: int = 5
    ) -> np.ndarray:
        """Compute n-step returns with bootstrapping.
        
        Args:
            rewards: Observed rewards
            values: Value estimates (one more than rewards for bootstrapping)
            dones: Done flags (1 means episode ended, 0 means continuing)
            discount: Discount factor
            n_steps: Number of steps for n-step returns
            
        Returns:
            N-step returns of shape (batch_size, sequence_length)
        """
        batch_size, seq_length = rewards.shape
        returns = np.zeros_like(rewards)
        
        # Create dones tensor if not provided
        if dones is None:
            dones = np.zeros_like(rewards)
        
        for b in range(batch_size):
            for t in range(seq_length):
                # Initialize return with immediate reward
                curr_return = rewards[b, t]
                
                # Add discounted rewards up to n steps or end of sequence
                gamma = discount
                done_mask = 1.0
                
                for i in range(1, min(n_steps, seq_length - t)):
                    # If we hit a terminal state, stop accumulating rewards
                    if dones[b, t + i - 1] > 0:
                        done_mask = 0.0
                        
                    curr_return += gamma * rewards[b, t + i] * done_mask
                    gamma *= discount
                
                # Add bootstrapped value if we haven't reached a terminal state
                if t + n_steps < seq_length + 1 and done_mask > 0:
                    curr_return += gamma * values[b, t + n_steps]
                
                returns[b, t] = curr_return
        
        return returns
    
    def _forward_dummy(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Generate dummy outputs for testing purposes.
        
        In a real implementation, this would be replaced with actual network forward passes.
        """
        # Define shapes
        state_shape = (batch_size, self.output_dim)
        value_shape = (batch_size, 1)
        reward_shape = (batch_size, 1)
        policy_shape = (batch_size, self.num_actions)
        
        # Generate random outputs scaled approximately to expected ranges
        states = self.rng.normal(0, 0.1, state_shape)
        values = self.rng.uniform(-0.1, 0.1, value_shape)
        rewards = self.rng.uniform(-0.1, 0.1, reward_shape)
        
        # For policy, generate logits and apply softmax
        policy_logits = self.rng.normal(0, 1, policy_shape)
        policy = np.exp(policy_logits) / np.sum(np.exp(policy_logits), axis=-1, keepdims=True)
        
        return {
            "states": states,
            "values": values,
            "rewards": rewards,
            "policies": policy
        }
    
    def initial_inference(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Initial inference for MuZero - representation + prediction.
        
        Args:
            obs: Batch of observations, shape (batch_size, *obs_shape)
            
        Returns:
            Dictionary with state representations, value and policy predictions
        """
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        
        # For testing purposes, we just return dummy outputs
        outputs = self._forward_dummy(batch_size)
        
        # Add value distribution
        value = outputs["values"].squeeze(-1)  # Remove last dimension
        value_categorical = self.scalar_to_categorical(value)
        outputs["value_categorical"] = value_categorical
        
        return outputs
    
    def recurrent_inference(
        self, 
        state: np.ndarray, 
        action: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Recurrent inference for MuZero - dynamics + prediction.
        
        Args:
            state: Batch of state representations, shape (batch_size, state_dim)
            action: Batch of actions, shape (batch_size,) or (batch_size, action_dim)
            
        Returns:
            Dictionary with next state, reward, value and policy predictions
        """
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        
        # For testing purposes, we just return dummy outputs
        outputs = self._forward_dummy(batch_size)
        
        # Add reward distribution (just for testing)
        reward = outputs["rewards"].squeeze(-1)  # Remove last dimension
        reward_categorical = self.scalar_to_categorical(reward)
        outputs["reward_categorical"] = reward_categorical
        
        # Add value distribution (as above)
        value = outputs["values"].squeeze(-1)
        value_categorical = self.scalar_to_categorical(value)
        outputs["value_categorical"] = value_categorical
        
        return outputs 