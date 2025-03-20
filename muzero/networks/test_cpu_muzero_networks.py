"""
Tests for the CPU-compatible MuZero networks implementation.
"""

import numpy as np
import pytest
from muzero.networks.cpu_muzero_networks import CPUMuZeroNetworks


# Test configuration
INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 32
NUM_ACTIONS = 4
BATCH_SIZE = 2
SEQ_LENGTH = 5


@pytest.fixture
def muzero_networks():
    """Fixture for CPU MuZero networks."""
    return CPUMuZeroNetworks(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_actions=NUM_ACTIONS,
        min_value=-1.0,
        max_value=1.0,
        categorical_size=51,
        seed=42
    )


def test_network_initialization(muzero_networks):
    """Test that networks are properly initialized."""
    # Check that parameters are created
    assert "rep_net" in muzero_networks.params
    assert "dyn_net" in muzero_networks.params
    assert "rew_net" in muzero_networks.params
    assert "val_net" in muzero_networks.params
    assert "pol_net" in muzero_networks.params

    # Check parameter shapes for representation network
    rep_net = muzero_networks.params["rep_net"]
    assert len(rep_net) == 3  # 3 layers
    assert rep_net[0]["weights"].shape == (INPUT_DIM, HIDDEN_DIM)
    assert rep_net[1]["weights"].shape == (HIDDEN_DIM, HIDDEN_DIM)
    assert rep_net[2]["weights"].shape == (HIDDEN_DIM, OUTPUT_DIM)


def test_scalar_to_categorical_conversion(muzero_networks):
    """Test the scalar to categorical value conversion."""
    # Test values
    test_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    
    # Convert to categorical
    categorical = muzero_networks.scalar_to_categorical(test_values)
    
    # Check shape
    assert categorical.shape == (5, 51)
    
    # Check it's a proper probability distribution (middle values)
    sums = np.sum(categorical, axis=1)
    assert np.allclose(sums[1:4], 1.0)
    
    # Convert back to scalar
    reconstructed = muzero_networks.categorical_to_scalar(categorical)
    
    # Check values match (middle values, allowing tolerance)
    assert np.allclose(test_values[1:4], reconstructed[1:4], atol=1e-2)


def test_batch_scalar_to_categorical(muzero_networks):
    """Test scalar to categorical with batched inputs."""
    # Create batched input with shape (2, 3)
    batch_values = np.array([
        [-0.5, 0.0, 0.5],
        [0.0, 0.25, 0.75]
    ], dtype=np.float32)
    
    # Convert to categorical
    categorical = muzero_networks.scalar_to_categorical(batch_values)
    
    # Check shape includes batch dimensions
    assert categorical.shape == (2, 3, 51)
    
    # Check it's a proper probability distribution
    assert np.allclose(np.sum(categorical, axis=2), 1.0)
    
    # Convert back to scalar
    reconstructed = muzero_networks.categorical_to_scalar(categorical)
    
    # Check values match and batch dimensions are preserved
    assert reconstructed.shape == (2, 3)
    assert np.allclose(batch_values, reconstructed, atol=1e-2)


def test_n_step_returns(muzero_networks):
    """Test the computation of n-step returns."""
    # Create test data - batch size 2, sequence length 5
    rewards = np.array([
        [0.0, 0.5, -0.5, 1.0, -1.0],
        [0.1, 0.2, -0.1, -0.2, 0.5]
    ], dtype=np.float32)
    
    # Values have one extra value for bootstrapping
    values = np.array([
        [0.1, 0.2, -0.1, 0.4, -0.2, 0.3],
        [0.2, 0.1, -0.2, -0.3, 0.4, 0.1]
    ], dtype=np.float32)
    
    # No terminal states in this test
    dones = np.zeros_like(rewards)
    
    # Compute returns with n=3 and discount=0.99
    returns = muzero_networks.compute_n_step_returns(
        rewards=rewards,
        values=values,
        dones=dones,
        discount=0.99,
        n_steps=3
    )
    
    # Check shape
    assert returns.shape == rewards.shape


def test_initial_inference(muzero_networks):
    """Test initial inference function."""
    # Create dummy observations
    obs = np.random.normal(0, 1, (BATCH_SIZE, INPUT_DIM)).astype(np.float32)
    
    # Run initial inference
    outputs = muzero_networks.initial_inference(obs)
    
    # Check outputs contain expected keys
    assert "states" in outputs
    assert "values" in outputs
    assert "policies" in outputs
    assert "value_categorical" in outputs
    
    # Check shapes
    assert outputs["states"].shape == (BATCH_SIZE, OUTPUT_DIM)
    assert outputs["values"].shape == (BATCH_SIZE, 1)
    assert outputs["policies"].shape == (BATCH_SIZE, NUM_ACTIONS)
    assert outputs["value_categorical"].shape == (BATCH_SIZE, 51)  # Categorical size
    
    # Check policy is a valid probability distribution
    assert np.allclose(np.sum(outputs["policies"], axis=1), 1.0)
    
    # Check value_categorical is a valid probability distribution
    assert np.allclose(np.sum(outputs["value_categorical"], axis=1), 1.0)


def test_recurrent_inference(muzero_networks):
    """Test recurrent inference function."""
    # Create dummy state and action
    state = np.random.normal(0, 1, (BATCH_SIZE, OUTPUT_DIM)).astype(np.float32)
    action = np.random.randint(0, NUM_ACTIONS, (BATCH_SIZE,))
    
    # Run recurrent inference
    outputs = muzero_networks.recurrent_inference(state, action)
    
    # Check outputs contain expected keys
    assert "states" in outputs
    assert "values" in outputs
    assert "rewards" in outputs
    assert "policies" in outputs
    assert "value_categorical" in outputs
    assert "reward_categorical" in outputs
    
    # Check shapes
    assert outputs["states"].shape == (BATCH_SIZE, OUTPUT_DIM)
    assert outputs["values"].shape == (BATCH_SIZE, 1)
    assert outputs["rewards"].shape == (BATCH_SIZE, 1)
    assert outputs["policies"].shape == (BATCH_SIZE, NUM_ACTIONS)
    assert outputs["value_categorical"].shape == (BATCH_SIZE, 51)  # Categorical size
    assert outputs["reward_categorical"].shape == (BATCH_SIZE, 51)  # Categorical size
    
    # Check policy is a valid probability distribution
    assert np.allclose(np.sum(outputs["policies"], axis=1), 1.0)
    
    # Check value_categorical and reward_categorical are valid probability distributions
    assert np.allclose(np.sum(outputs["value_categorical"], axis=1), 1.0)
    assert np.allclose(np.sum(outputs["reward_categorical"], axis=1), 1.0)


def test_muzero_loss_computation(muzero_networks):
    """Test the computation of MuZero loss."""
    # Setup test data
    batch_size = 2
    num_unroll = 3
    
    # Value targets and predictions: (batch_size, num_unroll + 1)
    value_targets = np.array([
        [0.5, 0.4, 0.3, 0.2],
        [0.2, 0.1, 0.0, -0.1]
    ], dtype=np.float32)
    
    value_preds = np.array([
        [0.6, 0.5, 0.4, 0.3],
        [0.1, 0.0, -0.1, -0.2]
    ], dtype=np.float32)
    
    # Reward targets and predictions: (batch_size, num_unroll)
    reward_targets = np.array([
        [0.1, 0.2, 0.3],
        [-0.1, -0.2, -0.3]
    ], dtype=np.float32)
    
    reward_preds = np.array([
        [0.2, 0.3, 0.4],
        [-0.2, -0.3, -0.4]
    ], dtype=np.float32)
    
    # Policy targets and predictions: (batch_size, num_unroll + 1, num_actions)
    policy_targets = np.zeros((batch_size, num_unroll + 1, NUM_ACTIONS), dtype=np.float32)
    policy_preds = np.zeros((batch_size, num_unroll + 1, NUM_ACTIONS), dtype=np.float32)
    
    # Fill with sample distributions
    for b in range(batch_size):
        for t in range(num_unroll + 1):
            # Sample target policy (slightly skewed)
            policy_targets[b, t] = np.array([0.3, 0.3, 0.3, 0.1]) if (b + t) % 2 == 0 else np.array([0.1, 0.3, 0.3, 0.3])
            
            # Sample predicted policy (uniform)
            logits = np.random.normal(0, 1, NUM_ACTIONS)
            policy_preds[b, t] = np.exp(logits) / np.sum(np.exp(logits))
    
    # Compute loss
    loss, loss_dict = muzero_networks.compute_muzero_loss(
        value_targets=value_targets,
        reward_targets=reward_targets,
        policy_targets=policy_targets,
        value_preds=value_preds,
        reward_preds=reward_preds,
        policy_preds=policy_preds
    )
    
    # Check loss components
    assert "total" in loss_dict
    assert "value" in loss_dict
    assert "reward" in loss_dict
    assert "policy" in loss_dict
    
    # Check loss values are positive
    assert loss_dict["total"] > 0
    assert loss_dict["value"] > 0
    assert loss_dict["reward"] > 0
    assert loss_dict["policy"] > 0 