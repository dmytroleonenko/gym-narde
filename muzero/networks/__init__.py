"""
MuZero networks module.

This module provides implementations of MuZero networks for both JAX and CPU-only environments.
"""

from muzero.networks.cpu_muzero_networks import CPUMuZeroNetworks

__all__ = ["CPUMuZeroNetworks"]

# For specific hardware targets (JAX/PyTorch), these will be conditionally imported

# Import main components
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