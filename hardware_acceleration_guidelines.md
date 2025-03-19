# Hardware Acceleration Guidelines for MuZero Narde

## Executive Summary

This document provides guidelines for implementing hardware acceleration in the Narde environment, based on comprehensive benchmarking of various operations using both JAX and PyTorch. Our testing on Apple Silicon (M-series) chips reveals significant acceleration opportunities for specific operations, particularly at larger batch sizes.

**Key Findings:**
- PyTorch with MPS (Metal Performance Shaders) shows up to 84x speedup for board operations at large batch sizes
- JAX with Metal backend has limited practical benefits due to compatibility issues
- The optimal acceleration strategy depends on operation type and batch size

## Implementation Recommendations

### 1. PyTorch Implementation

For optimal hardware acceleration, we recommend using the PyTorch implementation with conditional device selection based on operation and batch size:

```python
# Dynamic device selection based on batch size
def get_compute_device(batch_size, operation_type='default'):
    """
    Select the optimal device based on batch size and operation type.
    
    Args:
        batch_size: Size of the batch being processed
        operation_type: Type of operation ('board_rotation', 'feature_extraction', etc.)
        
    Returns:
        torch.device: The optimal device for this operation
    """
    # CPU device always available as fallback
    cpu_device = torch.device("cpu")
    
    # MPS device if available 
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        
        # Thresholds based on benchmarks
        thresholds = {
            'board_rotation': 1024,
            'feature_extraction': 128,
            'block_rule': 2048,
            'neural_network': 512,
            'default': 1024
        }
        
        # Use operation-specific threshold or default
        threshold = thresholds.get(operation_type, thresholds['default'])
        
        # Return appropriate device based on batch size
        return mps_device if batch_size >= threshold else cpu_device
    
    # CUDA device if available (for non-Mac systems)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    
    # Fallback to CPU
    return cpu_device
```

### 2. Operation-Specific Recommendations

Based on our benchmarks, here are operation-specific recommendations:

| Operation | Batch Size Threshold | Maximum Observed Speedup |
|-----------|----------------------|--------------------------|
| Board Rotation | ≥ 1024 | 8.17x at batch size 8192 |
| Board Operations | ≥ 128 | 84.00x at batch size 8192 |
| Neural Network | ≥ 512 | 1.97x at batch size 4096 |
| Block Rule Checking | ≥ 2048 | Not significant in benchmarks |

### 3. Model Architecture Considerations

For neural network operations, we recommend:

- **Larger hidden dimensions**: Models with hidden dimensions ≥ 512 show better acceleration
- **Batch normalization**: Including batch normalization layers improves MPS performance
- **Precision**: Consider using `torch.float16` (half-precision) for further acceleration

### 4. Training Strategy Adjustments

To maximize the benefits of hardware acceleration:

- **Larger batch sizes**: Use batch sizes ≥ 512 for training when possible
- **Gradient accumulation**: If memory constraints prevent large batches, use gradient accumulation
- **Mixed precision training**: Implement mixed precision training for better performance

## Implementation Examples

### Environment Wrapper

For a smooth transition, consider using an environment wrapper that dynamically selects the appropriate implementation:

```python
class AcceleratedNardeEnv:
    """
    Environment wrapper that selects the appropriate implementation
    based on hardware availability and operation batch sizes.
    """
    
    def __init__(self, use_acceleration=True):
        self.use_acceleration = use_acceleration
        self.has_mps = torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        
        # Initialize torch environment for accelerated operations
        if use_acceleration and (self.has_mps or self.has_cuda):
            self.torch_env = TorchNardeEnv(use_acceleration=True)
            self.using_torch = True
        else:
            # Fall back to optimized NumPy implementation
            self.numpy_env = OptimizedNardeEnv()
            self.using_torch = False
    
    # Delegate methods to appropriate implementation
    def reset(self, **kwargs):
        if self.using_torch:
            return self.torch_env.reset(**kwargs)
        else:
            return self.numpy_env.reset(**kwargs)
    
    def step(self, action):
        if self.using_torch:
            return self.torch_env.step(action)
        else:
            return self.numpy_env.step(action)
    
    # Example of operation with dynamic device selection
    def batch_board_rotation(self, boards, rotation=1):
        batch_size = boards.shape[0] if len(boards.shape) > 1 else 1
        
        if self.using_torch:
            # Convert input to torch tensor if it's numpy
            if isinstance(boards, np.ndarray):
                boards = torch.tensor(boards)
            
            # Select device based on batch size
            device = get_compute_device(batch_size, 'board_rotation')
            boards = boards.to(device)
            
            # Perform operation
            rotated = self.torch_env.rotate_board(boards, rotation)
            
            # Return as numpy for consistency
            return rotated.cpu().numpy()
        else:
            # Use numpy implementation
            if batch_size > 1:
                return np.array([self.numpy_env._rotate_board(board) for board in boards])
            else:
                return self.numpy_env._rotate_board(boards)
```

### Neural Network with Dynamic Device Selection

```python
class DynamicDeviceMuZeroNetwork(torch.nn.Module):
    """MuZero network that dynamically selects the appropriate device based on batch size."""
    
    def __init__(self, input_dim=24, hidden_dim=512, latent_dim=256, output_dim=30):
        super().__init__()
        self.representation_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            # ... more layers ...
        )
        # ... other networks ...
    
    def forward(self, obs, action=None, batch_size=None):
        """
        Full forward pass through the network with dynamic device selection.
        """
        # Determine batch size if not provided
        if batch_size is None:
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        
        # Select device based on batch size and operation
        device = get_compute_device(batch_size, 'neural_network')
        
        # Move networks to device as needed
        if next(self.parameters()).device != device:
            self.to(device)
        
        # Move input tensors to device
        obs = obs.to(device)
        if action is not None:
            action = action.to(device)
        
        # Forward pass
        latent_state = self.representation(obs)
        
        if action is not None:
            next_latent_state, reward = self.dynamics(latent_state, action)
            policy, value = self.prediction(next_latent_state)
            return policy, value, reward
        else:
            policy, value = self.prediction(latent_state)
            return policy, value
```

## Performance Tuning

For optimal performance, consider these additional tuning steps:

1. **Warm-up operations**: Pre-compile operations with example inputs to avoid compilation overhead
2. **Memory management**: Explicitly clear cache between large batch operations
3. **Synchronization**: Use `torch.mps.synchronize()` for proper timing of operations
4. **Batch segmentation**: For extremely large batches, consider splitting into sub-batches

## Conclusion

Hardware acceleration can significantly improve the performance of the Narde environment, especially for large batch operations typical in reinforcement learning training. By following the guidelines in this document, you can achieve speedups of up to 84x for certain operations compared to the baseline NumPy implementation.

The PyTorch-based implementation with dynamic device selection provides the most flexible and efficient approach for hardware acceleration across different systems, accommodating both Metal Performance Shaders on Apple Silicon and CUDA on NVIDIA GPUs. 