import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad, devices, device_put, block_until_ready
import flax.linen as nn
from typing import Any, Tuple
import numpy as np
import os

# Set NCCL flags that might improve performance on GPU
# These flags don't impact CPU execution
os.environ.update({
    "NCCL_LL128_BUFFSIZE": "-2",
    "NCCL_LL_BUFFSIZE": "-2",
    "NCCL_PROTO": "SIMPLE,LL,LL128",
})

# Simple JAX implementation for MuZero components

class MuZeroNetworkJAX(nn.Module):
    """
    Full MuZero network implemented in JAX/Flax.
    """
    input_dim: int
    action_dim: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        """Forward pass - initial inference wrapper"""
        return self.initial_inference(x)
    
    @nn.compact
    def initial_inference(self, observation):
        """Initial inference for the root node"""
        # Representation network
        x = nn.Dense(128)(observation)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        hidden = nn.Dense(self.hidden_dim)(x)
        hidden = nn.LayerNorm()(hidden)
        
        # Prediction network (policy and value)
        x = nn.Dense(256)(hidden)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Value head
        value = nn.Dense(64)(x)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)  # Value normalized to [-1, 1]
        
        # Policy head
        policy_logits = nn.Dense(self.action_dim)(x)
        
        return hidden, value, policy_logits
    
    @nn.compact
    def recurrent_inference(self, hidden, action_onehot):
        """Recurrent inference for non-root nodes"""
        # Dynamics network
        x = jnp.concatenate([hidden, action_onehot], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        next_hidden = nn.Dense(self.hidden_dim)(x)
        next_hidden = nn.LayerNorm()(next_hidden)
        
        # Reward head
        reward = nn.Dense(64)(next_hidden)
        reward = nn.LayerNorm()(reward)
        reward = nn.relu(reward)
        reward = nn.Dense(1)(reward)
        
        # Prediction network on next hidden
        x = nn.Dense(256)(next_hidden)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Value head
        value = nn.Dense(64)(x)
        value = nn.LayerNorm()(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)  # Value normalized to [-1, 1]
        
        # Policy head
        policy_logits = nn.Dense(self.action_dim)(x)
        
        return next_hidden, reward, value, policy_logits


class MuZeroStateJAX:
    """Simplified JAX model state with JIT-compiled operations"""
    def __init__(self, input_dim, action_dim, hidden_dim=256, seed=0):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Get best available device - prefer accelerator over CPU
        self.devices = jax.devices()
        self.accelerator_types = [d.platform for d in self.devices]
        
        # Print device info
        print(f"Available JAX devices: {self.accelerator_types}")
        
        # Choose best device (prefer GPU/TPU over CPU)
        if "gpu" in self.accelerator_types:
            self.device = jax.devices("gpu")[0]
            print("Using CUDA GPU acceleration for JAX")
        elif "tpu" in self.accelerator_types:
            self.device = jax.devices("tpu")[0]
            print("Using TPU acceleration for JAX")
        elif "metal" in self.accelerator_types:
            self.device = jax.devices("metal")[0]
            print("Using Metal (MPS) acceleration for JAX")
        else:
            self.device = jax.devices()[0]  # Default to first device (CPU)
            print("No hardware accelerator available for JAX, using CPU")
        
        # Initialize model and parameters
        self.model = MuZeroNetworkJAX(input_dim, action_dim, hidden_dim)
        
        # Initialize random key
        self.rng = random.PRNGKey(seed)
        
        # Initialize parameters
        self.rng, init_rng = random.split(self.rng)
        dummy_input = jnp.ones((1, input_dim))
        self.params = self.model.init(init_rng, dummy_input)
        
        # Transfer parameters to device
        self.params = device_put(self.params, self.device)
        
        # JIT-compile the inference functions with strong optimizations
        backend = "gpu" if "gpu" in self.accelerator_types else jax.default_backend()
        print(f"Using JAX backend: {backend} for compilation")
        
        self._init_inference = jit(
            self._initial_inference_fn, 
            backend=backend
        )
        
        self._recur_inference = jit(
            self._recurrent_inference_fn,
            backend=backend
        )
        
        # Vectorized recurrent inference (jit + vmap)
        self._recur_inference_batch = jit(
            vmap(self._recurrent_inference_fn, in_axes=(None, 0, 0)),
            backend=backend
        )
        
        # Warm up the JIT compilation
        self._warmup()
    
    def _warmup(self):
        """Warm up the JIT compilation to avoid compilation time in benchmarks"""
        # Warm up initial inference
        dummy_obs = jnp.ones((1, self.input_dim))
        dummy_obs = device_put(dummy_obs, self.device)
        _ = self._init_inference(self.params, dummy_obs)
        
        # Warm up recurrent inference
        dummy_hidden = jnp.ones((1, self.hidden_dim))
        dummy_action = jnp.array(0)
        dummy_hidden = device_put(dummy_hidden, self.device)
        dummy_action = device_put(dummy_action, self.device)
        _ = self._recur_inference(self.params, dummy_hidden, dummy_action)
        
        # Warm up batch inference
        dummy_hidden_batch = jnp.ones((2, self.hidden_dim))
        dummy_action_batch = jnp.array([0, 1])
        dummy_hidden_batch = device_put(dummy_hidden_batch, self.device)
        dummy_action_batch = device_put(dummy_action_batch, self.device)
        _ = self._recur_inference_batch(self.params, dummy_hidden_batch, dummy_action_batch)
    
    def _initial_inference_fn(self, params, obs):
        """Function for JIT-compilation - initial inference"""
        output = self.model.apply(params, obs, method=self.model.initial_inference)
        return output
    
    def _recurrent_inference_fn(self, params, hidden, action_idx):
        """Function for JIT-compilation - recurrent inference"""
        # Create one-hot encoding for the action with proper batch dimension
        action_onehot = jax.nn.one_hot(action_idx, self.action_dim)
        # Ensure action_onehot has same batch dimension as hidden
        if len(hidden.shape) > 1 and len(action_onehot.shape) == 1:
            action_onehot = action_onehot.reshape(1, -1)
        
        output = self.model.apply(
            params, hidden, action_onehot, 
            method=self.model.recurrent_inference
        )
        return output
    
    def initial_inference(self, observation):
        """Initial inference with JIT acceleration"""
        # Ensure batch dimension
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        
        # Transfer observation to device
        observation = device_put(observation, self.device)
        
        # Run inference and block until ready
        output = self._init_inference(self.params, observation)
        output = tuple(block_until_ready(x) for x in output)
        
        return output
    
    def recurrent_inference(self, hidden, action_idx):
        """Recurrent inference with JIT acceleration"""
        # Ensure batch dimension
        if len(hidden.shape) == 1:
            hidden = hidden.reshape(1, -1)
        
        # Transfer inputs to device
        hidden = device_put(hidden, self.device)
        action_idx = device_put(action_idx, self.device)
        
        # Run inference and block until ready
        output = self._recur_inference(self.params, hidden, action_idx)
        output = tuple(block_until_ready(x) for x in output)
        
        return output
    
    def recurrent_inference_batch(self, hidden_batch, action_batch):
        """Vectorized batch recurrent inference"""
        # Transfer inputs to device
        hidden_batch = device_put(hidden_batch, self.device)
        action_batch = device_put(action_batch, self.device)
        
        # Run inference and block until ready
        output = self._recur_inference_batch(self.params, hidden_batch, action_batch)
        output = tuple(block_until_ready(x) for x in output)
        
        return output
    
    def to_torch_compatible(self):
        """
        Create a torch-compatible wrapper for the JAX model
        that can be used with the existing MCTS implementation
        """
        import torch
        
        class TorchWrapper:
            def __init__(self, jax_state):
                self.jax_state = jax_state
                self.action_dim = jax_state.action_dim
                self.hidden_dim = jax_state.hidden_dim
            
            def initial_inference(self, observation):
                # Convert torch tensor to numpy array
                if isinstance(observation, torch.Tensor):
                    observation = observation.detach().cpu().numpy()
                
                # JAX initial inference
                hidden, value, policy_logits = self.jax_state.initial_inference(observation)
                
                # Convert back to torch tensors
                hidden_torch = torch.from_numpy(np.array(hidden))
                value_torch = torch.from_numpy(np.array(value))
                policy_logits_torch = torch.from_numpy(np.array(policy_logits))
                
                return hidden_torch, value_torch, policy_logits_torch
            
            def recurrent_inference(self, hidden, action):
                # Convert torch tensor to numpy array
                if isinstance(hidden, torch.Tensor):
                    hidden = hidden.detach().cpu().numpy()
                
                if isinstance(action, torch.Tensor):
                    # Handle different action formats
                    if len(action.shape) > 1 and action.shape[1] > 1:
                        # One-hot format, find the index
                        action = action.detach().cpu().numpy()
                        action = np.argmax(action, axis=1)[0]
                    else:
                        action = action.detach().cpu().item()
                
                # JAX recurrent inference
                next_hidden, reward, value, policy_logits = self.jax_state.recurrent_inference(
                    hidden, action
                )
                
                # Convert back to torch tensors
                next_hidden_torch = torch.from_numpy(np.array(next_hidden))
                reward_torch = torch.from_numpy(np.array(reward))
                value_torch = torch.from_numpy(np.array(value))
                policy_logits_torch = torch.from_numpy(np.array(policy_logits))
                
                return next_hidden_torch, reward_torch, value_torch, policy_logits_torch
            
            def recurrent_inference_batch(self, hidden_batch, action_batch):
                # Convert torch tensors to numpy arrays
                if isinstance(hidden_batch, torch.Tensor):
                    hidden_batch = hidden_batch.detach().cpu().numpy()
                
                if isinstance(action_batch, torch.Tensor):
                    action_batch = action_batch.detach().cpu().numpy()
                
                # JAX batch recurrent inference
                results = self.jax_state.recurrent_inference_batch(
                    hidden_batch, action_batch
                )
                
                # Unpack results and convert back to torch tensors
                next_hidden_batch, reward_batch, value_batch, policy_logits_batch = results
                
                next_hidden_torch = torch.from_numpy(np.array(next_hidden_batch))
                reward_torch = torch.from_numpy(np.array(reward_batch))
                value_torch = torch.from_numpy(np.array(value_batch))
                policy_logits_torch = torch.from_numpy(np.array(policy_logits_batch))
                
                return next_hidden_torch, reward_torch, value_torch, policy_logits_torch
        
        return TorchWrapper(self)


# Helper function to create the JAX model
def create_muzero_jax(input_dim, action_dim, hidden_dim=256, seed=0):
    """Create a MuZeroStateJAX model with the given dimensions"""
    return MuZeroStateJAX(input_dim, action_dim, hidden_dim, seed) 