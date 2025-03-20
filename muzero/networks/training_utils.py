"""
Training utilities for MuZero implementation.
These functions implement n-step returns for bootstrapped targets
and model unrolling during training.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from typing import Dict, Tuple, List, Optional, Any

# --- Target Generation Functions ---

def compute_n_step_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    discount_factor: float,
    n_steps: int
) -> jnp.ndarray:
    """
    Compute n-step returns for value targets.
    
    Args:
        rewards: Rewards of shape [T]
        values: Bootstrap values of shape [T+1]
        dones: Done flags of shape [T]
        discount_factor: Discount factor γ
        n_steps: Number of steps for bootstrapping
        
    Returns:
        n_step_returns: N-step returns of shape [T]
    """
    # Input shapes validation
    assert rewards.shape[0] + 1 == values.shape[0], "Values should have one more entry than rewards"
    assert rewards.shape[0] == dones.shape[0], "Rewards and dones should have the same shape"
    
    sequence_length = rewards.shape[0]
    
    def compute_return_from_i(i):
        # Take n steps or until end of sequence, whichever comes first
        n_steps_from_i = min(n_steps, sequence_length - i)
        
        # Gather relevant rewards
        relevant_rewards = rewards[i:i+n_steps_from_i]
        
        # Compute discounts, taking into account done flags
        # We'll multiply subsequent discounts based on (1-dones)
        # If we hit a done state, the discount becomes 0 from that point
        not_done = 1.0 - dones[i:i+n_steps_from_i]
        cumulative_not_done = jnp.cumprod(not_done)
        discounts = cumulative_not_done * (discount_factor ** jnp.arange(1, n_steps_from_i + 1))
        
        # Add bootstrap value for the state after n steps, discounted appropriately
        bootstrap_value = values[i + n_steps_from_i]
        bootstrap_discount = discount_factor ** n_steps_from_i
        bootstrap_discount = bootstrap_discount * jnp.prod(not_done) if n_steps_from_i > 0 else bootstrap_discount
        
        # Compute the n-step return
        # R_t + γR_{t+1} + γ²R_{t+2} + ... + γⁿV_{t+n}
        return_i = jnp.sum(relevant_rewards * discounts) + bootstrap_discount * bootstrap_value
        
        return return_i
    
    # Compute return for each timestep in the sequence
    returns = jnp.array([compute_return_from_i(i) for i in range(sequence_length)])
    
    return returns


def generate_muzero_targets(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    next_observations: jnp.ndarray,
    search_policies: jnp.ndarray,
    bootstrap_values: jnp.ndarray,
    discount_factor: float,
    n_steps: int,
    num_bins: int,
    min_value: float,
    max_value: float
) -> Dict[str, jnp.ndarray]:
    """
    Generate targets for MuZero training.
    
    Args:
        observations: Observations of shape [T, ...]
        actions: Actions of shape [T]
        rewards: Rewards of shape [T]
        dones: Done flags of shape [T]
        next_observations: Next observations of shape [T, ...]
        search_policies: Policy targets from MCTS of shape [T, num_actions]
        bootstrap_values: Value estimates for bootstrapping of shape [T+1]
        discount_factor: Discount factor γ
        n_steps: Number of steps for bootstrapping
        num_bins: Number of bins for categorical representation
        min_value: Minimum value for categorical representation
        max_value: Maximum value for categorical representation
        
    Returns:
        targets: Dictionary containing policy, value, and reward targets
    """
    # Import scalar_to_categorical locally to avoid circular imports
    from muzero.networks.jax_muzero_networks import scalar_to_categorical
    
    # Compute n-step returns
    returns = compute_n_step_returns(
        rewards=rewards,
        values=bootstrap_values,
        dones=dones,
        discount_factor=discount_factor,
        n_steps=n_steps
    )
    
    # Convert scalar returns to categorical using two-hot encoding
    value_target = scalar_to_categorical(returns, num_bins, min_value, max_value)
    
    # Convert scalar rewards to categorical
    reward_target = scalar_to_categorical(rewards, num_bins, min_value, max_value)
    
    # Combine everything into targets dictionary
    targets = {
        'policy_target': search_policies,
        'value_target': value_target,
        'reward_target': reward_target,
        'returns': returns,  # Keep scalar returns for logging
    }
    
    return targets


# --- Training Step Functions ---

def muzero_train_step(
    params: Any,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    apply_fn: Any,
    batch: Dict[str, jnp.ndarray],
    rng_key: jnp.ndarray,
    discount_factor: float = 0.99,
    n_steps: int = 5,
    num_bins: int = 51,
    min_value: float = -1.0,
    max_value: float = 1.0,
    value_coef: float = 1.0,
    policy_coef: float = 1.0
) -> Tuple[Any, optax.OptState, Dict[str, float]]:
    """
    Single training step for MuZero.
    
    Args:
        params: Network parameters
        optimizer_state: Optimizer state
        optimizer: Optax optimizer
        apply_fn: Network apply function (from Haiku transform)
        batch: Batch of experience data
        rng_key: JAX PRNG key
        discount_factor: Discount factor γ
        n_steps: Number of steps for bootstrapping
        num_bins: Number of bins for categorical representation
        min_value: Minimum value for categorical representation
        max_value: Maximum value for categorical representation
        value_coef: Value loss coefficient
        policy_coef: Policy loss coefficient
        
    Returns:
        updated_params: Updated network parameters
        updated_optimizer_state: Updated optimizer state
        metrics: Training metrics
    """
    # Import locally to avoid circular imports
    from muzero.networks.jax_muzero_networks import compute_muzero_loss
    
    # Unpack batch
    observations = batch['observations']
    actions = batch['actions']
    rewards = batch['rewards']
    dones = batch['dones']
    next_observations = batch['next_observations']
    search_policies = batch['search_policies']
    bootstrap_values = batch['bootstrap_values']
    
    # Generate targets
    targets = generate_muzero_targets(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=next_observations,
        search_policies=search_policies,
        bootstrap_values=bootstrap_values,
        discount_factor=discount_factor,
        n_steps=n_steps,
        num_bins=num_bins,
        min_value=min_value,
        max_value=max_value
    )
    
    def loss_fn(params):
        # Forward pass
        predictions = apply_fn(params, rng_key, observations, actions)
        
        # Compute loss
        loss, metrics = compute_muzero_loss(
            predictions=predictions,
            targets=targets,
            value_coef=value_coef,
            policy_coef=policy_coef
        )
        
        return loss, metrics
    
    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Apply gradients
    updates, updated_optimizer_state = optimizer.update(grads, optimizer_state, params)
    updated_params = optax.apply_updates(params, updates)
    
    # Add gradient norm metrics
    metrics['grad_norm'] = optax.global_norm(grads)
    metrics['param_norm'] = optax.global_norm(updated_params)
    
    return updated_params, updated_optimizer_state, metrics


# --- Vectorized Operations ---

def vectorized_model_unroll(
    params: Any,
    apply_fn: Any,
    initial_states: jnp.ndarray,
    action_sequences: jnp.ndarray,
    rng_key: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Unroll the model for multiple trajectories in parallel.
    
    Args:
        params: Network parameters
        apply_fn: Network apply function (from Haiku transform)
        initial_states: Initial states of shape [B, ...]
        action_sequences: Sequences of actions of shape [B, T]
        rng_key: JAX PRNG key
        
    Returns:
        unrolled_predictions: Dictionary containing unrolled predictions
    """
    batch_size, seq_length = action_sequences.shape
    
    # Define a function to unroll a single trajectory
    def unroll_single_trajectory(initial_state, action_sequence):
        # Define a single trajectory unrolling function
        # This will be mapped over the batch
        def step_fn(carry, action):
            state = carry
            # Generate a subkey for this step
            subkey = rng_key  # In practice, you would split rng_key for each step
            
            # Apply dynamics network to get next state and reward
            predictions = apply_fn(params, subkey, state, action)
            
            # Extract next state
            next_state = predictions['next_states'][0]
            
            return next_state, predictions
        
        # Scan over the action sequence
        final_state, unrolled = jax.lax.scan(
            step_fn, 
            initial_state, 
            action_sequence
        )
        
        return unrolled
    
    # Map over batch dimension to unroll each trajectory
    batch_unrolled = jax.vmap(unroll_single_trajectory)(initial_states, action_sequences)
    
    return batch_unrolled 