#!/usr/bin/env python3
"""
Benchmark script to compare PyTorch and JAX implementations of MuZero.
"""

import time
import torch
import numpy as np
import gymnasium as gym
import argparse
import gym_narde  # Import to register the environment
from collections import deque

# PyTorch implementation
from muzero.models import MuZeroNetwork
from muzero.mcts import MCTS
from muzero.training import train_muzero

# JAX implementation
import jax
from muzero.models_jax import create_muzero_jax


def benchmark_forward_passes(num_iterations=1000, batch_size=32):
    """
    Benchmark forward passes through the PyTorch and JAX models.
    """
    print("\n=== Benchmarking Forward Passes ===")
    
    # Create environment to get input dimensions
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    
    # Create random inputs
    observations = np.random.randn(batch_size, input_dim).astype(np.bfloat16)
    
    # Create PyTorch model with appropriate device
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        print("Using CUDA for PyTorch")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
        print("Using MPS for PyTorch")
    else:
        torch_device = torch.device("cpu")
        print("Using CPU for PyTorch (no GPU available)")
    
    torch_model = MuZeroNetwork(input_dim, action_dim).to(torch_device)
    torch_observations = torch.FloatTensor(observations).to(torch_device)
    
    # Create JAX model and ensure we use hardware acceleration if available
    print(f"JAX is using backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("Note: For optimal JAX performance on Mac, consider installing jax[metal]")
    
    jax_state = create_muzero_jax(input_dim, action_dim)
    
    # Warm-up
    for _ in range(10):
        # PyTorch warm-up
        with torch.no_grad():
            torch_model.initial_inference(torch_observations)
        
        # JAX warm-up
        jax_state.initial_inference(observations)
    
    # Benchmark PyTorch
    torch_times = []
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            result = torch_model.initial_inference(torch_observations)
            # Ensure computation is completed for GPU timing
            if torch_device.type != "cpu":
                torch.cuda.synchronize() if torch.cuda.is_available() else torch.mps.synchronize()
    torch_total_time = time.time() - start_time
    torch_avg_time = torch_total_time / num_iterations
    
    # Benchmark JAX
    jax_times = []
    start_time = time.time()
    for _ in range(num_iterations):
        # JAX's block_until_ready is used in the model implementation
        _, _, _ = jax_state.initial_inference(observations)
    jax_total_time = time.time() - start_time
    jax_avg_time = jax_total_time / num_iterations
    
    # Print results
    print(f"PyTorch: {torch_avg_time * 1000:.2f} ms per batch (batch size: {batch_size})")
    print(f"JAX    : {jax_avg_time * 1000:.2f} ms per batch (batch size: {batch_size})")
    print(f"Speedup: {torch_avg_time / jax_avg_time:.2f}x")
    
    return torch_avg_time, jax_avg_time


def benchmark_recurrent_inference(num_iterations=1000, batch_size=32):
    """
    Benchmark recurrent inference through the PyTorch and JAX models.
    """
    print("\n=== Benchmarking Recurrent Inference ===")
    
    # Create environment to get input dimensions
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    hidden_dim = 256
    
    # Create random inputs
    hidden_states = np.random.randn(batch_size, hidden_dim).astype(np.bfloat16)
    actions = np.random.randint(0, action_dim, size=batch_size)
    
    # Create PyTorch model with appropriate device
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    
    torch_model = MuZeroNetwork(input_dim, action_dim).to(torch_device)
    torch_hidden_states = torch.FloatTensor(hidden_states).to(torch_device)
    torch_actions = torch.LongTensor(actions).to(torch_device)
    
    # Create JAX model
    jax_state = create_muzero_jax(input_dim, action_dim)
    
    # Warm-up
    for _ in range(10):
        # PyTorch warm-up (using recurrent_inference for each item in batch)
        with torch.no_grad():
            for i in range(batch_size):
                # Create action one-hot for PyTorch model
                action_onehot = torch.zeros(1, action_dim, device=torch_device)
                action_onehot[0, torch_actions[i]] = 1.0
                torch_model.recurrent_inference(torch_hidden_states[i:i+1], action_onehot)
        
        # JAX warm-up (using vectorized recurrent_inference_batch)
        jax_state.recurrent_inference_batch(hidden_states, actions)
    
    # Benchmark PyTorch
    torch_times = []
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            for i in range(batch_size):
                # Create action one-hot for PyTorch model
                action_onehot = torch.zeros(1, action_dim, device=torch_device)
                action_onehot[0, torch_actions[i]] = 1.0
                result = torch_model.recurrent_inference(torch_hidden_states[i:i+1], action_onehot)
                # Ensure computation is completed for GPU timing
                if torch_device.type != "cpu":
                    torch.cuda.synchronize() if torch.cuda.is_available() else torch.mps.synchronize()
    torch_total_time = time.time() - start_time
    torch_avg_time = torch_total_time / num_iterations
    
    # Benchmark JAX with vectorization
    jax_times = []
    start_time = time.time()
    for _ in range(num_iterations):
        # JAX's block_until_ready is used in the model implementation
        _, _, _, _ = jax_state.recurrent_inference_batch(hidden_states, actions)
    jax_total_time = time.time() - start_time
    jax_avg_time = jax_total_time / num_iterations
    
    # Print results
    print(f"PyTorch (serial) : {torch_avg_time * 1000:.2f} ms per batch (batch size: {batch_size}, device: {torch_device.type})")
    print(f"JAX (vectorized) : {jax_avg_time * 1000:.2f} ms per batch (batch size: {batch_size}, backend: {jax.default_backend()})")
    print(f"Speedup         : {torch_avg_time / jax_avg_time:.2f}x")
    
    return torch_avg_time, jax_avg_time


def benchmark_mcts(num_simulations=50, num_runs=10):
    """
    Benchmark MCTS using the PyTorch and JAX models.
    """
    print("\n=== Benchmarking MCTS ===")
    
    # Create environment to get input dimensions
    env = gym.make('Narde-v0')
    input_dim = env.observation_space.shape[0]  # 28
    action_dim = 24 * 24  # 576 possible (from, to) combinations
    
    # Get a sample observation
    observation, _ = env.reset()
    
    # Get valid actions
    unwrapped_env = env.unwrapped
    valid_moves = unwrapped_env.game.get_valid_moves()
    valid_actions = []
    
    for move in valid_moves:
        # Convert move to action index
        if move[1] == 'off':
            # Bear off move: (from_pos, 'off')
            action_idx = move[0] * 24
        else:
            # Regular move: (from_pos, to_pos)
            action_idx = move[0] * 24 + move[1]
            
        valid_actions.append(action_idx)
    
    # Create PyTorch model with appropriate device
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    
    torch_model = MuZeroNetwork(input_dim, action_dim).to(torch_device)
    torch_mcts = MCTS(
        network=torch_model,
        num_simulations=num_simulations,
        discount=0.997,
        action_space_size=action_dim,
        device=torch_device
    )
    
    # Create JAX model and wrap it for MCTS
    jax_state = create_muzero_jax(input_dim, action_dim)
    jax_model_wrapped = jax_state.to_torch_compatible()
    jax_mcts = MCTS(
        network=jax_model_wrapped,
        num_simulations=num_simulations,
        discount=0.997,
        action_space_size=action_dim,
        device=torch_device
    )
    
    # Warm-up
    _ = torch_mcts.run(observation, valid_actions, add_exploration_noise=True)
    _ = jax_mcts.run(observation, valid_actions, add_exploration_noise=True)
    
    # Benchmark PyTorch MCTS
    torch_times = []
    for _ in range(num_runs):
        start_time = time.time()
        result = torch_mcts.run(observation, valid_actions, add_exploration_noise=True)
        # Ensure computation is completed for GPU timing
        if torch_device.type != "cpu":
            torch.cuda.synchronize() if torch.cuda.is_available() else torch.mps.synchronize()
        torch_times.append(time.time() - start_time)
    torch_avg_time = sum(torch_times) / len(torch_times)
    
    # Benchmark JAX MCTS
    jax_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = jax_mcts.run(observation, valid_actions, add_exploration_noise=True)
        jax_times.append(time.time() - start_time)
    jax_avg_time = sum(jax_times) / len(jax_times)
    
    # Print results
    print(f"PyTorch MCTS: {torch_avg_time * 1000:.2f} ms per run ({num_simulations} simulations, device: {torch_device.type})")
    print(f"JAX MCTS    : {jax_avg_time * 1000:.2f} ms per run ({num_simulations} simulations, backend: {jax.default_backend()})")
    print(f"Speedup     : {torch_avg_time / jax_avg_time:.2f}x")
    
    return torch_avg_time, jax_avg_time


def benchmark_training(num_episodes=5):
    """
    Benchmark a small training run using PyTorch.
    In the future, this could be extended to compare with a JAX implementation.
    """
    print("\n=== Benchmarking Training (PyTorch only) ===")
    
    # Determine device
    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    
    print(f"Using device: {device_str}")
    
    # Run PyTorch training
    start_time = time.time()
    train_muzero(
        num_episodes=num_episodes,
        replay_buffer_size=1000,
        batch_size=16,
        num_simulations=10,
        temperature_init=1.0,
        temperature_final=0.1,
        device_str=device_str,
        enable_profiling=False,
        save_interval=num_episodes
    )
    torch_total_time = time.time() - start_time
    
    print(f"PyTorch training: {torch_total_time:.2f} seconds for {num_episodes} episodes on {device_str}")
    
    return torch_total_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark MuZero implementations")
    parser.add_argument("--forward-iterations", type=int, default=1000, 
                       help="Number of iterations for forward pass benchmark")
    parser.add_argument("--recurrent-iterations", type=int, default=100, 
                       help="Number of iterations for recurrent inference benchmark")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for forward and recurrent benchmarks")
    parser.add_argument("--mcts-simulations", type=int, default=50, 
                       help="Number of MCTS simulations per run")
    parser.add_argument("--mcts-runs", type=int, default=10, 
                       help="Number of MCTS runs for benchmarking")
    parser.add_argument("--train-episodes", type=int, default=5, 
                       help="Number of episodes for training benchmark")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip the training benchmark")
    parser.add_argument("--forward-only", action="store_true",
                       help="Run only the forward pass benchmark")
    
    args = parser.parse_args()
    
    # Print JAX info
    print("JAX devices:", jax.devices())
    print("JAX default device:", jax.default_backend())
    
    # Run benchmarks
    forward_torch_time, forward_jax_time = benchmark_forward_passes(
        num_iterations=args.forward_iterations, 
        batch_size=args.batch_size
    )
    
    # Only run additional benchmarks if forward-only is not specified
    if not args.forward_only:
        recurrent_torch_time, recurrent_jax_time = benchmark_recurrent_inference(
            num_iterations=args.recurrent_iterations, 
            batch_size=args.batch_size
        )
        
        mcts_torch_time, mcts_jax_time = benchmark_mcts(
            num_simulations=args.mcts_simulations,
            num_runs=args.mcts_runs
        )
        
        if not args.skip_training:
            train_time = benchmark_training(num_episodes=args.train_episodes)
    
    # Summary
    print("\n=== Benchmark Summary ===")
    print(f"Forward Pass - PyTorch: {forward_torch_time * 1000:.2f} ms, JAX: {forward_jax_time * 1000:.2f} ms, Speedup: {forward_torch_time / forward_jax_time:.2f}x")
    
    if not args.forward_only:
        print(f"Recur. Infer - PyTorch: {recurrent_torch_time * 1000:.2f} ms, JAX: {recurrent_jax_time * 1000:.2f} ms, Speedup: {recurrent_torch_time / recurrent_jax_time:.2f}x")
        print(f"MCTS Sim     - PyTorch: {mcts_torch_time * 1000:.2f} ms, JAX: {mcts_jax_time * 1000:.2f} ms, Speedup: {mcts_torch_time / mcts_jax_time:.2f}x")
        
        if not args.skip_training:
            print(f"Training     - PyTorch: {train_time:.2f} seconds for {args.train_episodes} episodes")


if __name__ == "__main__":
    main() 