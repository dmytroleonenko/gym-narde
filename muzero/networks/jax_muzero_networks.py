"""
MuZero network implementation in JAX using Haiku.
This follows the JAX implementation style while incorporating ideas from the reference code.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Dict, Tuple, Any

# --- Network Architecture ---

class MuZeroNetworks(hk.Module):
    """MuZero networks implemented in Haiku."""
    
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, action_dim=30, num_bins=51, 
                 min_value=-1, max_value=1, name=None):
        """Initialize the MuZero network components.
        
        Args:
            input_dim: Dimension of the input observation (e.g., board state)
            hidden_dim: Size of hidden layers
            latent_dim: Size of the latent state representation
            action_dim: Number of possible actions
            num_bins: Number of bins for categorical value representation
            min_value: Minimum expected value
            max_value: Maximum expected value
            name: Optional name for the module
        """
        super().__init__(name=name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        
    def _mlp_block(self, x, output_sizes, activation=jax.nn.relu, name=None):
        """Helper to create an MLP block."""
        net = hk.nets.MLP(
            output_sizes=output_sizes,
            activation=activation,
            name=name
        )
        return net(x)
        
    def representation_network(self, obs):
        """Convert observation to latent state."""
        x = self._mlp_block(
            obs, 
            output_sizes=[self.hidden_dim, self.hidden_dim, self.latent_dim],
            name="representation"
        )
        return x
        
    def dynamics_network(self, latent_state, action):
        """Predict next latent state and reward given current state and action."""
        # Convert action to one-hot representation
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        
        # Concatenate latent state and action
        x = jnp.concatenate([latent_state, action_one_hot], axis=-1)
        
        # MLP to predict next state and reward
        x = self._mlp_block(
            x, 
            output_sizes=[self.hidden_dim, self.hidden_dim, self.latent_dim + self.num_bins],
            name="dynamics"
        )
        
        # Split into next state and reward
        next_latent_state = x[..., :self.latent_dim]
        reward_logits = x[..., self.latent_dim:]
        
        return next_latent_state, reward_logits
    
    def prediction_network(self, latent_state):
        """Predict policy and value from latent state."""
        # Policy head
        policy_logits = self._mlp_block(
            latent_state,
            output_sizes=[self.hidden_dim, self.action_dim],
            name="policy"
        )
        
        # Value head
        value_logits = self._mlp_block(
            latent_state,
            output_sizes=[self.hidden_dim, self.num_bins],
            name="value"
        )
        
        return policy_logits, value_logits
    
    def __call__(self, obs, actions=None):
        """Full forward pass through the network.
        
        Args:
            obs: Batch of observations
            actions: Optional batch of actions for dynamics network
            
        Returns:
            Dictionary containing network outputs
        """
        # Initial representation
        latent_state = self.representation_network(obs)
        
        # Get initial policy and value predictions
        policy_logits, value_logits = self.prediction_network(latent_state)
        
        # If actions are provided, use dynamics to predict next states and rewards
        predictions = {
            'state': latent_state,
            'policy_logits': policy_logits,
            'value_logits': value_logits,
        }
        
        # For training, we may want to unroll the dynamics network multiple steps
        if actions is not None:
            # Initialize lists to store unrolled predictions
            next_latent_states = []
            reward_logits_list = []
            policy_logits_list = []
            value_logits_list = []
            
            # Current state for unrolling
            current_state = latent_state
            
            # Support both single action and sequences of actions
            if actions.ndim == 1:
                actions = jnp.expand_dims(actions, axis=0)  # Add time dimension
            
            # Unroll for each action in the sequence
            for t in range(actions.shape[0]):
                # Get action at this timestep
                action_t = actions[t]
                
                # Apply dynamics
                next_state, reward_logits = self.dynamics_network(current_state, action_t)
                
                # Get policy and value for next state
                next_policy_logits, next_value_logits = self.prediction_network(next_state)
                
                # Store predictions
                next_latent_states.append(next_state)
                reward_logits_list.append(reward_logits)
                policy_logits_list.append(next_policy_logits)
                value_logits_list.append(next_value_logits)
                
                # Update current state for next iteration
                current_state = next_state
            
            # Add unrolled predictions to output
            predictions.update({
                'next_states': jnp.stack(next_latent_states),
                'reward_logits': jnp.stack(reward_logits_list),
                'next_policy_logits': jnp.stack(policy_logits_list),
                'next_value_logits': jnp.stack(value_logits_list),
            })
        
        return predictions


# --- Value Transformation ---

def scalar_to_categorical(x, num_bins, min_value, max_value):
    """Convert scalar values to categorical using two-hot encoding.
    
    This is a simplified version of the rlax.transform_to_2hot function.
    """
    x = jnp.clip(x, min_value, max_value)
    
    # Scale x to be in [0, num_bins - 1]
    x = (x - min_value) / (max_value - min_value) * (num_bins - 1)
    
    # Compute lower and upper indices
    lower_idx = jnp.floor(x).astype(jnp.int32)
    upper_idx = jnp.ceil(x).astype(jnp.int32)
    
    # Fast-track common case of integer x
    same_idx = lower_idx == upper_idx
    
    # Compute weights for lower and upper indices
    upper_weight = jnp.where(same_idx, 1.0, x - lower_idx)
    lower_weight = jnp.where(same_idx, 0.0, 1.0 - upper_weight)
    
    # Create one-hot encodings and combine them
    lower_one_hot = jax.nn.one_hot(lower_idx, num_bins)
    upper_one_hot = jax.nn.one_hot(upper_idx, num_bins)
    
    # Reshape weights for broadcasting
    lower_weight = jnp.expand_dims(lower_weight, axis=-1)
    upper_weight = jnp.expand_dims(upper_weight, axis=-1)
    
    # Combine with weights
    return lower_weight * lower_one_hot + upper_weight * upper_one_hot

def categorical_to_scalar(categorical, min_value, max_value):
    """Convert categorical distribution back to scalar."""
    support = jnp.linspace(min_value, max_value, categorical.shape[-1])
    return jnp.sum(categorical * support, axis=-1)


# --- MuZero Loss Functions ---

def compute_policy_entropy(policy_logits):
    """Compute the entropy of a policy distribution."""
    log_policy = jax.nn.log_softmax(policy_logits, axis=-1)
    policy = jax.nn.softmax(policy_logits, axis=-1)
    return -jnp.sum(policy * log_policy, axis=-1)

def categorical_cross_entropy(targets, logits):
    """Compute categorical cross entropy between targets and logits."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(targets * log_probs, axis=-1)

def compute_muzero_loss(predictions, targets, value_coef=1.0, policy_coef=1.0):
    """Compute the MuZero loss function.
    
    Args:
        predictions: Dictionary of network predictions including policy_logits, value_logits, reward_logits
        targets: Dictionary of target values including policy_target, value_target, reward_target
        value_coef: Weight for value loss
        policy_coef: Weight for policy loss
        
    Returns:
        total_loss: The combined loss
        metrics: Dictionary of individual loss components and metrics
    """
    # Unpack targets and predictions
    policy_logits = predictions['policy_logits']
    policy_target = targets['policy_target']
    
    value_logits = predictions['value_logits']
    value_target = targets['value_target']
    
    # Compute categorical cross-entropy losses
    policy_loss = jnp.mean(categorical_cross_entropy(policy_target, policy_logits))
    value_loss = jnp.mean(categorical_cross_entropy(value_target, value_logits))
    
    # Add reward loss if available
    reward_loss = 0.0
    if 'reward_logits' in predictions and 'reward_target' in targets:
        reward_logits = predictions['reward_logits']
        reward_target = targets['reward_target']
        reward_loss = jnp.mean(categorical_cross_entropy(reward_target, reward_logits))
    
    # Combine losses with weights
    total_loss = reward_loss + value_coef * value_loss + policy_coef * policy_loss
    
    # Compute additional metrics
    policy_entropy = jnp.mean(compute_policy_entropy(policy_logits))
    
    metrics = {
        'reward_loss': reward_loss,
        'value_loss': value_loss,
        'policy_loss': policy_loss,
        'policy_entropy': policy_entropy,
        'total_loss': total_loss,
    }
    
    return total_loss, metrics


# --- Helper Functions ---

def create_muzero_networks(input_dim, hidden_dim=128, latent_dim=64, action_dim=30, num_bins=51, 
                          min_value=-1, max_value=1):
    """Create MuZero networks with Haiku transform."""
    
    def forward_fn(obs, actions=None):
        networks = MuZeroNetworks(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_bins=num_bins,
            min_value=min_value,
            max_value=max_value
        )
        return networks(obs, actions)
    
    # Transform pure functions to stateful functions with parameters
    init, apply = hk.transform(forward_fn)
    
    return init, apply


def init_muzero_params(rng_key, input_dim, hidden_dim=128, latent_dim=64, action_dim=30, num_bins=51,
                      min_value=-1, max_value=1):
    """Initialize MuZero parameters."""
    init_fn, _ = create_muzero_networks(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        num_bins=num_bins,
        min_value=min_value,
        max_value=max_value
    )
    
    # Create dummy observation for initialization
    dummy_obs = jnp.zeros((1, input_dim))
    
    # Initialize parameters
    params = init_fn(rng_key, dummy_obs)
    
    return params


def configure_optimizer(learning_rate=0.001, weight_decay=1e-4, max_grad_norm=10.0):
    """Configure the optimizer for MuZero training."""
    # Create optimizer chain: gradient clipping + Adam with weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    
    return optimizer 