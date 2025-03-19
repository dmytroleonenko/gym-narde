#!/usr/bin/env python3
"""
Benchmark PyTorch with MPS acceleration for large MuZero-like neural networks and batch sizes.
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

# Define a larger MuZero-like network
class LargeMuZeroNetwork(torch.nn.Module):
    """
    A larger implementation of the MuZero network architecture
    with representation, dynamics, and prediction networks.
    """
    def __init__(self, input_dim=24, hidden_dim=512, latent_dim=256, output_dim=30):
        super().__init__()
        
        # Representation network (board state -> latent state)
        self.representation_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Dynamics network (latent state + action -> next latent state + reward)
        self.dynamics_network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 1, hidden_dim),  # +1 for action
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, latent_dim + 1)  # +1 for reward
        )
        
        # Prediction network (latent state -> policy + value)
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
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

def run_large_muzero_benchmark(batch_sizes):
    """Benchmark large MuZero-like network with PyTorch on CPU vs MPS."""
    print("\n=== Large MuZero Network Benchmark ===")
    
    # Network parameters
    input_dim = 24  # Board size for Narde
    hidden_dim = 512
    latent_dim = 256
    output_dim = 30  # Number of possible actions
    
    # Create models with same initialization for fair comparison
    torch.manual_seed(42)
    cpu_model = LargeMuZeroNetwork(input_dim, hidden_dim, latent_dim, output_dim).to(cpu_device)
    torch.manual_seed(42)  # Reset seed for identical initialization
    device_model = LargeMuZeroNetwork(input_dim, hidden_dim, latent_dim, output_dim).to(device)
    
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
        iterations = max(1, int(20 / batch_size))  # Fewer iterations for larger batches
        for _ in range(iterations):
            _ = cpu_model(obs_cpu, action_cpu)
        cpu_time = (time.time() - start_time) / iterations
        print(f"PyTorch CPU large MuZero forward: {cpu_time:.6f} seconds")
        
        # Benchmark full MuZero forward pass - device (MPS/CUDA/CPU)
        start_time = time.time()
        for _ in range(iterations):
            _ = device_model(obs_device, action_device)
            if device.type in ["cuda", "mps"]:
                torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        device_time = (time.time() - start_time) / iterations
        print(f"PyTorch {device.type.upper()} large MuZero forward: {device_time:.6f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / device_time if device_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        cpu_times.append(cpu_time)
        mps_times.append(device_time)
        
    return cpu_times, mps_times

def main():
    # Define batch sizes to test - focusing on larger batches for training scenarios
    large_batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # Run benchmark
    cpu_times, mps_times = run_large_muzero_benchmark(large_batch_sizes)
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("\nLarge MuZero Network (hidden_dim=512, latent_dim=256):")
    for i, batch_size in enumerate(large_batch_sizes):
        speedup = cpu_times[i] / mps_times[i] if mps_times[i] > 0 else 0
        print(f"Batch size: {batch_size}, CPU: {cpu_times[i]:.6f}s, {device.type.upper()}: {mps_times[i]:.6f}s, Speedup: {speedup:.2f}x")
    
    # Print optimal batch size recommendation
    if any(cpu_times[i] > mps_times[i] for i in range(len(cpu_times))):
        crossover_index = next((i for i, (cpu_t, mps_t) in enumerate(zip(cpu_times, mps_times)) if cpu_t > mps_t), -1)
        if crossover_index >= 0:
            print(f"\nRecommendation: Use MPS acceleration for batch sizes ≥ {large_batch_sizes[crossover_index]}")
        else:
            print("\nRecommendation: Use CPU for all tested batch sizes")
    else:
        print("\nRecommendation: Use CPU for all tested batch sizes")

if __name__ == "__main__":
    main() 