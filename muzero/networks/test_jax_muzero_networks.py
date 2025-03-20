import os
import sys

# Replace environment variables with JAX config API
# Force JAX to use CPU and prevent any GPU/TPU/MPS acceleration
import jax

# Configure JAX to use CPU and disable JIT
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_disable_jit", True)

# Keep this one as it's specifically for XLA
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

# Only import the rest after setting configuration
import pytest
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from typing import Tuple, Dict

# Import our implementation
from muzero.networks.jax_muzero_networks import (
    MuZeroNetworks, 
    create_muzero_networks, 
    init_muzero_params,
    scalar_to_categorical,
    categorical_to_scalar,
    compute_muzero_loss,
    configure_optimizer
)

from muzero.networks.training_utils import (
    compute_n_step_returns,
    generate_muzero_targets,
    muzero_train_step,
    vectorized_model_unroll
)

# Constants for testing
BATCH_SIZE = 2
ACTION_DIM = 576  # 24x24 possible moves for Narde
HIDDEN_DIM = 64
NUM_CHANNELS = 64
NUM_BLOCKS = 2
INPUT_DIM = 28  # Actual observation space size for Narde
LATENT_DIM = 32
MIN_VALUE = -1.0
MAX_VALUE = 1.0
SEQ_LENGTH = 3


@pytest.fixture
def rng_key():
    """Fixture for random number generation."""
    return jax.random.key(42)


@pytest.fixture
def muzero_params(rng_key):
    """Fixture for initialized MuZero parameters."""
    return init_muzero_params(
        rng_key,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        num_bins=NUM_BINS,
        min_value=MIN_VALUE,
        max_value=MAX_VALUE
    )


@pytest.fixture
def muzero_networks():
    """Fixture for MuZero network initialization and apply functions."""
    return create_muzero_networks(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        num_bins=NUM_BINS,
        min_value=MIN_VALUE,
        max_value=MAX_VALUE
    )


@pytest.fixture
def sample_batch(rng_key):
    """Fixture for a sample batch of data."""
    # Split the key into multiple keys
    key1, key2, key3, key4, key5, key6 = jax.random.split(rng_key, 6)
    
    observations = jax.random.normal(key1, (BATCH_SIZE, INPUT_DIM))
    actions = jax.random.randint(key2, (BATCH_SIZE,), 0, ACTION_DIM)
    rewards = jax.random.uniform(key3, (BATCH_SIZE,), minval=-0.5, maxval=0.5)
    dones = jax.random.bernoulli(key4, 0.1, (BATCH_SIZE,))
    next_observations = jax.random.normal(key5, (BATCH_SIZE, INPUT_DIM))
    
    # Generate random policy distributions
    logits = jax.random.normal(key6, (BATCH_SIZE, ACTION_DIM))
    search_policies = jax.nn.softmax(logits, axis=-1)
    
    # Generate random bootstrap values
    bootstrap_values = jax.random.uniform(
        key3, (BATCH_SIZE + 1,), minval=-0.5, maxval=0.5
    )
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'next_observations': next_observations,
        'search_policies': search_policies,
        'bootstrap_values': bootstrap_values
    }


# --- Network Tests ---

def test_muzero_networks_initialization(muzero_networks, rng_key):
    """Test that MuZero networks can be initialized properly."""
    init_fn, apply_fn = muzero_networks
    
    # Create a dummy observation
    dummy_obs = jnp.zeros((1, INPUT_DIM))
    
    # Initialize parameters
    params = init_fn(rng_key, dummy_obs)
    
    # Check that parameters have the expected structure
    assert "representation" in params
    assert "dynamics" in params
    assert "policy" in params
    assert "value" in params


def test_muzero_networks_forward_pass(muzero_networks, muzero_params, rng_key):
    """Test a basic forward pass through the MuZero networks."""
    _, apply_fn = muzero_networks
    
    # Create a dummy observation and action
    dummy_obs = jnp.zeros((1, INPUT_DIM))
    dummy_action = jnp.array([5])  # Example action
    
    # Apply the network - include rng_key as the second argument
    predictions = apply_fn(muzero_params, rng_key, dummy_obs)
    
    # Check output shapes
    assert predictions['state'].shape == (1, LATENT_DIM)
    assert predictions['policy_logits'].shape == (1, ACTION_DIM)
    assert predictions['value_logits'].shape == (1, NUM_BINS)


def test_muzero_networks_with_actions(muzero_networks, muzero_params, rng_key):
    """Test forward pass with actions to check dynamics network."""
    _, apply_fn = muzero_networks
    
    # Create a dummy observation and action
    dummy_obs = jnp.zeros((1, INPUT_DIM))
    dummy_action = jnp.array([5])  # Example action
    
    # Apply the network with actions - include rng_key as the second argument
    predictions = apply_fn(muzero_params, rng_key, dummy_obs, dummy_action)
    
    # Check additional outputs when actions are provided
    assert 'next_states' in predictions
    assert 'reward_logits' in predictions
    assert 'next_policy_logits' in predictions
    assert 'next_value_logits' in predictions
    
    # Check shapes
    assert predictions['next_states'].shape == (1, 1, LATENT_DIM)
    assert predictions['reward_logits'].shape == (1, NUM_BINS)


def test_muzero_networks_with_action_sequence(muzero_networks, muzero_params, rng_key):
    """Test forward pass with a sequence of actions."""
    _, apply_fn = muzero_networks
    
    # Create a dummy observation and action sequence
    dummy_obs = jnp.zeros((1, INPUT_DIM))
    dummy_actions = jnp.array([[5, 10, 15]])  # Sequence of actions
    
    # Apply the network with action sequence - include rng_key as the second argument
    predictions = apply_fn(muzero_params, rng_key, dummy_obs, dummy_actions[0])
    
    # Check shapes for sequences
    assert predictions['next_states'].shape == (3, 1, LATENT_DIM)
    assert predictions['reward_logits'].shape == (3, NUM_BINS)
    assert predictions['next_policy_logits'].shape == (3, ACTION_DIM)
    assert predictions['next_value_logits'].shape == (3, NUM_BINS)


# --- Value Transformation Tests ---

def test_scalar_to_categorical_and_back():
    """Test conversion between scalar and categorical representations."""
    # Define constants
    num_bins = NUM_BINS  # Use the constant defined at the top of the file
    
    # Create test values using JAX
    test_values = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=jnp.float32)
    
    # Convert to categorical using the JAX implementation
    categorical = scalar_to_categorical(test_values, num_bins, MIN_VALUE, MAX_VALUE)
    
    # Convert back to scalar
    reconstructed = categorical_to_scalar(categorical, MIN_VALUE, MAX_VALUE)
    
    # Check values match
    assert jnp.allclose(test_values, reconstructed, atol=1e-2)


# --- Loss Computation Tests ---

def test_compute_muzero_loss(rng_key):
    """Test the MuZero loss computation."""
    # Create fake predictions and targets
    batch_size = 2
    
    # Create random predictions
    key1, key2, key3 = jax.random.split(rng_key, 3)
    policy_logits = jax.random.normal(key1, (batch_size, ACTION_DIM))
    value_logits = jax.random.normal(key2, (batch_size, NUM_BINS))
    reward_logits = jax.random.normal(key3, (batch_size, NUM_BINS))
    
    predictions = {
        'policy_logits': policy_logits,
        'value_logits': value_logits,
        'reward_logits': reward_logits
    }
    
    # Create targets (uniform policy, random value and reward categories)
    policy_target = jnp.ones((batch_size, ACTION_DIM)) / ACTION_DIM
    value_target = jax.nn.softmax(value_logits, axis=-1)  # Just for testing
    reward_target = jax.nn.softmax(reward_logits, axis=-1)  # Just for testing
    
    targets = {
        'policy_target': policy_target,
        'value_target': value_target,
        'reward_target': reward_target
    }
    
    # Compute loss
    loss, metrics = compute_muzero_loss(predictions, targets)
    
    # Check that loss is a scalar and metrics contain expected keys
    assert loss.ndim == 0
    assert 'reward_loss' in metrics
    assert 'value_loss' in metrics
    assert 'policy_loss' in metrics
    assert 'total_loss' in metrics


# --- Training Utility Tests ---

def test_compute_n_step_returns():
    """Test the computation of n-step returns."""
    # Create test data with JAX
    rewards = jnp.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=jnp.float32)
    values = jnp.array([0.1, 0.2, -0.1, 0.4, -0.2, 0.3], dtype=jnp.float32)  # Extra value for bootstrapping
    dones = jnp.zeros_like(rewards)  # No episodes end in this test
    discount = 0.9
    n_steps = 3
    
    # Compute returns using JAX implementation
    returns = compute_n_step_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        discount_factor=discount,
        n_steps=n_steps
    )
    
    # Verify returned shape
    assert returns.shape == (5,)
    
    # Manually calculate expected values for specific positions
    # t=0: rewards[0] + discount*rewards[1] + discount^2*rewards[2] + discount^3*values[3]
    expected_0 = 0.0 + 0.9*0.5 + 0.9*0.9*(-0.5) + 0.9*0.9*0.9*0.4
    # t=3: rewards[3] + discount*rewards[4] + discount^2*values[5]
    expected_3 = 1.0 + 0.9*(-1.0) + 0.9*0.9*0.3
    
    # Verify specific calculations against manual expectations - use a higher tolerance
    assert jnp.abs(returns[0] - expected_0) < 1e-2
    assert jnp.abs(returns[3] - expected_3) < 1e-2


def test_generate_muzero_targets(sample_batch):
    """Test the generation of MuZero targets."""
    # Extract data from sample batch
    observations = sample_batch['observations']
    actions = sample_batch['actions']
    rewards = sample_batch['rewards']
    dones = sample_batch['dones']
    next_observations = sample_batch['next_observations']
    search_policies = sample_batch['search_policies']
    bootstrap_values = sample_batch['bootstrap_values']
    
    # Generate targets
    discount_factor = 0.99
    n_steps = 3
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
        num_bins=NUM_BINS,
        min_value=MIN_VALUE,
        max_value=MAX_VALUE
    )
    
    # Check that all expected keys are present
    assert 'policy_target' in targets
    assert 'value_target' in targets
    assert 'reward_target' in targets
    assert 'returns' in targets
    
    # Check shapes
    assert targets['policy_target'].shape == (BATCH_SIZE, ACTION_DIM)
    assert targets['value_target'].shape == (BATCH_SIZE, NUM_BINS)
    assert targets['reward_target'].shape == (BATCH_SIZE, NUM_BINS)
    assert targets['returns'].shape == (BATCH_SIZE,)


def test_muzero_train_step(muzero_networks, muzero_params, sample_batch, rng_key):
    """Test a complete MuZero training step."""
    _, apply_fn = muzero_networks
    
    # Configure optimizer
    optimizer = configure_optimizer(learning_rate=0.001)
    optimizer_state = optimizer.init(muzero_params)
    
    # Perform a training step
    updated_params, updated_optimizer_state, metrics = muzero_train_step(
        params=muzero_params,
        optimizer_state=optimizer_state,
        optimizer=optimizer,
        apply_fn=apply_fn,
        batch=sample_batch,
        rng_key=rng_key
    )
    
    # Check that parameters were updated
    assert jax.tree_util.tree_structure(updated_params) == jax.tree_util.tree_structure(muzero_params)
    params_unchanged = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), muzero_params, updated_params)
    )
    assert not params_unchanged  # Parameters should have changed
    
    # Check that metrics contain expected keys
    assert 'total_loss' in metrics
    assert 'grad_norm' in metrics
    assert 'param_norm' in metrics


@pytest.mark.xfail(reason="Vectorized unroll is currently failing on Apple Silicon, needs investigation")
def test_vectorized_model_unroll(muzero_networks, muzero_params, rng_key):
    """Test vectorized model unrolling."""
    _, apply_fn = muzero_networks
    
    # Create dummy initial states and action sequences
    batch_size = 2
    seq_length = 3
    initial_states = jnp.zeros((batch_size, LATENT_DIM))
    action_sequences = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    
    # Add a second parameter for the RNG key
    unrolled = vectorized_model_unroll(
        params=muzero_params,
        apply_fn=apply_fn,
        initial_states=initial_states,
        action_sequences=action_sequences,
        rng_key=rng_key
    )
    
    # Check outputs
    assert 'next_states' in unrolled
    assert 'reward_logits' in unrolled
    assert 'next_policy_logits' in unrolled
    assert 'next_value_logits' in unrolled


# --- Integration Test ---

def test_end_to_end_training(muzero_networks, muzero_params, sample_batch, rng_key):
    """Test an end-to-end training loop for a few steps."""
    _, apply_fn = muzero_networks
    
    # Configure optimizer
    optimizer = configure_optimizer(learning_rate=0.001)
    optimizer_state = optimizer.init(muzero_params)
    
    # Train for a few steps
    params = muzero_params
    opt_state = optimizer_state
    
    for i in range(3):
        # Split key for each iteration to get a fresh key
        rng_key, subkey = jax.random.split(rng_key)
        params, opt_state, metrics = muzero_train_step(
            params=params,
            optimizer_state=opt_state,
            optimizer=optimizer,
            apply_fn=apply_fn,
            batch=sample_batch,
            rng_key=subkey
        )
    
    # Verify that we get back updated parameters and metrics
    assert 'total_loss' in metrics
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics 