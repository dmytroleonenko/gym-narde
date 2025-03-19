#!/usr/bin/env python3
"""
Benchmark PyTorch with MPS acceleration for MuZero-like neural networks.
"""

import torch
import numpy as np
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")

# Determine the best available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# CPU device for comparison
cpu_device = torch.device("cpu")

# Define a MuZero-like network
class MuZeroNetwork(torch.nn.Module):
    """
    A simplified implementation of the MuZero network architecture
    with representation, dynamics, and prediction networks.
    """
    def __init__(self, input_dim=24, hidden_dim=128, latent_dim=64, output_dim=30):
        super().__init__()
        
        # Representation network (board state -> latent state)
        self.representation_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        
        # Dynamics network (latent state + action -> next latent state + reward)
        self.dynamics_network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 1, hidden_dim),  # +1 for action
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim + 1)  # +1 for reward
        )
        
        # Prediction network (latent state -> policy + value)
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def representation(self, obs):
        """Convert observation to latent state."""
        return self.representation_network(obs)
    
    def dynamics(self, latent_state, action):
        """Predict next latent state and reward given current state and action."""
        # Concatenate latent state and action
        action = action.unsqueeze(-1) if action.dim() == 1 else action
        x = torch.cat([latent_state, action], dim=-1)
        output = self.dynamics_network(x)
        
        # Split output into next state and reward
        next_latent_state = output[:, :-1]
        reward = output[:, -1:]
        
        return next_latent_state, reward
    
    def prediction(self, latent_state):
        """Predict policy and value from latent state."""
        policy = self.policy_network(latent_state)
        value = self.value_network(latent_state)
        return policy, value
    
    def forward(self, obs, action=None):
        """Full forward pass through the network."""
        latent_state = self.representation(obs)
        
        if action is not None:
            next_latent_state, reward = self.dynamics(latent_state, action)
            policy, value = self.prediction(next_latent_state)
            return policy, value, reward
        else:
            policy, value = self.prediction(latent_state)
            return policy, value

def run_muzero_benchmark(batch_sizes):
    """Benchmark MuZero-like network with PyTorch on CPU vs MPS."""
    print("\n=== MuZero Network Benchmark ===")
    
    # Network parameters
    input_dim = 24  # Board size for Narde
    hidden_dim = 256
    latent_dim = 128
    output_dim = 30  # Number of possible actions
    
    # Create models with same initialization for fair comparison
    torch.manual_seed(42)
    cpu_model = MuZeroNetwork(input_dim, hidden_dim, latent_dim, output_dim).to(cpu_device)
    torch.manual_seed(42)  # Reset seed for identical initialization
    device_model = MuZeroNetwork(input_dim, hidden_dim, latent_dim, output_dim).to(device)
    
    cpu_times = []
    mps_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random input
        np.random.seed(42)
        obs_np = np.random.random((batch_size, input_dim)).astype(np.float32)
        action_np = np.random.randint(0, output_dim, size=(batch_size)).astype(np.float32)
        
        # Convert to PyTorch tensors
        obs_cpu = torch.tensor(obs_np, device=cpu_device)
        action_cpu = torch.tensor(action_np, device=cpu_device)
        
        obs_device = torch.tensor(obs_np, device=device)
        action_device = torch.tensor(action_np, device=device)
        
        # Warm up
        _ = device_model(obs_device, action_device)
        if device.type in ["cuda", "mps"]:
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        
        # Benchmark full MuZero forward pass - CPU
        start_time = time.time()
        for _ in range(50):
            _ = cpu_model(obs_cpu, action_cpu)
        cpu_time = (time.time() - start_time) / 50
        print(f"PyTorch CPU MuZero forward: {cpu_time:.6f} seconds")
        
        # Benchmark full MuZero forward pass - device (MPS/CUDA/CPU)
        start_time = time.time()
        for _ in range(50):
            _ = device_model(obs_device, action_device)
            if device.type in ["cuda", "mps"]:
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        device_time = (time.time() - start_time) / 50
        print(f"PyTorch {device.type.upper()} MuZero forward: {device_time:.6f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / device_time if device_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        cpu_times.append(cpu_time)
        mps_times.append(device_time)
        
    return cpu_times, mps_times

def run_muzero_inference_benchmark(batch_sizes):
    """Benchmark MuZero inference (representation + prediction) with PyTorch on CPU vs MPS."""
    print("\n=== MuZero Inference Benchmark ===")
    
    # Network parameters
    input_dim = 24  # Board size for Narde
    hidden_dim = 256
    latent_dim = 128
    output_dim = 30  # Number of possible actions
    
    # Create models with same initialization for fair comparison
    torch.manual_seed(42)
    cpu_model = MuZeroNetwork(input_dim, hidden_dim, latent_dim, output_dim).to(cpu_device)
    torch.manual_seed(42)  # Reset seed for identical initialization
    device_model = MuZeroNetwork(input_dim, hidden_dim, latent_dim, output_dim).to(device)
    
    cpu_times = []
    mps_times = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Generate random input
        np.random.seed(42)
        obs_np = np.random.random((batch_size, input_dim)).astype(np.float32)
        
        # Convert to PyTorch tensors
        obs_cpu = torch.tensor(obs_np, device=cpu_device)
        obs_device = torch.tensor(obs_np, device=device)
        
        # Warm up
        _ = device_model.representation(obs_device)
        latent = device_model.representation(obs_device)
        _ = device_model.prediction(latent)
        if device.type in ["cuda", "mps"]:
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        
        # Benchmark inference (representation + prediction) - CPU
        start_time = time.time()
        for _ in range(50):
            latent = cpu_model.representation(obs_cpu)
            _ = cpu_model.prediction(latent)
        cpu_time = (time.time() - start_time) / 50
        print(f"PyTorch CPU MuZero inference: {cpu_time:.6f} seconds")
        
        # Benchmark inference (representation + prediction) - device (MPS/CUDA/CPU)
        start_time = time.time()
        for _ in range(50):
            latent = device_model.representation(obs_device)
            _ = device_model.prediction(latent)
            if device.type in ["cuda", "mps"]:
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        device_time = (time.time() - start_time) / 50
        print(f"PyTorch {device.type.upper()} MuZero inference: {device_time:.6f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / device_time if device_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        cpu_times.append(cpu_time)
        mps_times.append(device_time)
        
    return cpu_times, mps_times

def main():
    # Define batch sizes to test - include small batches and large batches
    small_batch_sizes = [1, 8, 32, 128]
    large_batch_sizes = [512, 1024, 2048]
    batch_sizes = small_batch_sizes + large_batch_sizes
    
    # Run benchmarks
    cpu_muzero_times, mps_muzero_times = run_muzero_benchmark(batch_sizes)
    cpu_inference_times, mps_inference_times = run_muzero_inference_benchmark(batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("\nMuZero Full Forward Pass:")
    for i, batch_size in enumerate(batch_sizes):
        speedup = cpu_muzero_times[i] / mps_muzero_times[i] if mps_muzero_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, CPU: {cpu_muzero_times[i]:.6f}s, {device.type.upper()}: {mps_muzero_times[i]:.6f}s, Speedup: {speedup:.2f}x")
    
    print("\nMuZero Inference (Representation + Prediction):")
    for i, batch_size in enumerate(batch_sizes):
        speedup = cpu_inference_times[i] / mps_inference_times[i] if mps_inference_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, CPU: {cpu_inference_times[i]:.6f}s, {device.type.upper()}: {mps_inference_times[i]:.6f}s, Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 