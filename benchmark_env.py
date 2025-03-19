#!/usr/bin/env python3
"""
Benchmark script to compare the performance of the original and PyTorch-accelerated Narde environments.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import timeit

# Import both environment implementations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'gym_narde', 'envs'))
from optimized_narde_env import OptimizedNardeEnv
from torch_narde_env import TorchNardeEnv

# Number of repetitions for timing
REPETITIONS = 10
# Number of episodes to run for testing
TEST_EPISODES = 5
# Maximum steps per episode
MAX_STEPS = 50

def time_function(func, *args, **kwargs) -> float:
    """
    Time a function execution using timeit.
    
    Args:
        func: Function to time
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Average execution time in seconds
    """
    # Create a timer function that calls our target function
    def timer():
        return func(*args, **kwargs)
    
    # Time the function with repetitions
    times = timeit.repeat(timer, repeat=REPETITIONS, number=1)
    
    # Return the average execution time
    return sum(times) / len(times)

def benchmark_reset() -> Dict[str, float]:
    """
    Benchmark the reset operation for both environments.
    
    Returns:
        Dictionary with timing results
    """
    print("Benchmarking environment reset...")
    
    # Create environments
    original_env = OptimizedNardeEnv()
    torch_env = TorchNardeEnv(use_acceleration=True)
    torch_cpu_env = TorchNardeEnv(use_acceleration=False)
    
    # Time reset operation
    original_time = time_function(original_env.reset)
    torch_time = time_function(torch_env.reset)
    torch_cpu_time = time_function(torch_cpu_env.reset)
    
    # Calculate speedup
    original_speedup = original_time / torch_time
    
    print(f"  Original: {original_time:.6f}s")
    print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s")
    print(f"  PyTorch (MPS): {torch_time:.6f}s")
    print(f"  Speedup (MPS vs Original): {original_speedup:.2f}x")
    
    return {
        'original': original_time,
        'torch_cpu': torch_cpu_time,
        'torch_mps': torch_time,
        'speedup': original_speedup
    }

def benchmark_get_valid_actions(batch_sizes: List[int] = [1, 10, 100, 1000]) -> Dict[str, List[float]]:
    """
    Benchmark the get_valid_actions operation for both environments with different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with timing results
    """
    print("\nBenchmarking get_valid_actions with different batch sizes...")
    
    # Create environments
    original_env = OptimizedNardeEnv()
    torch_env = TorchNardeEnv(use_acceleration=True)
    torch_cpu_env = TorchNardeEnv(use_acceleration=False)
    
    # Reset environments
    original_env.reset()
    torch_env.reset()
    torch_cpu_env.reset()
    
    original_times = []
    torch_times = []
    torch_cpu_times = []
    speedups = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Time get_valid_actions operation
        original_time = time_function(
            lambda: [original_env.valid_actions for _ in range(batch_size)]  # Use valid_actions attribute for original env
        )
        torch_time = time_function(lambda: torch_env.get_valid_actions(batch_size=batch_size))
        torch_cpu_time = time_function(lambda: torch_cpu_env.get_valid_actions(batch_size=batch_size))
        
        # Normalize by batch size
        original_time /= batch_size
        torch_time /= batch_size
        torch_cpu_time /= batch_size
        
        # Calculate speedup
        speedup = original_time / torch_time
        
        print(f"  Original: {original_time:.6f}s per call")
        print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s per call")
        print(f"  PyTorch (MPS): {torch_time:.6f}s per call")
        print(f"  Speedup (MPS vs Original): {speedup:.2f}x")
        
        original_times.append(original_time)
        torch_times.append(torch_time)
        torch_cpu_times.append(torch_cpu_time)
        speedups.append(speedup)
    
    return {
        'batch_sizes': batch_sizes,
        'original': original_times,
        'torch_cpu': torch_cpu_times,
        'torch_mps': torch_times,
        'speedup': speedups
    }

def benchmark_board_rotation(batch_sizes: List[int] = [1, 10, 100, 1000]) -> Dict[str, List[float]]:
    """
    Benchmark the board rotation operation for both environments with different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with timing results
    """
    print("\nBenchmarking board rotation with different batch sizes...")
    
    # Create environments
    original_env = OptimizedNardeEnv()
    torch_env = TorchNardeEnv(use_acceleration=True)
    torch_cpu_env = TorchNardeEnv(use_acceleration=False)
    
    # Reset environments
    original_env.reset()
    torch_env.reset()
    torch_cpu_env.reset()
    
    # Get boards
    original_board = original_env.board
    torch_board = torch_env.board
    
    # Ensure the original environment has a board that can be rotated for testing
    # Set a few pieces to make rotation visible
    original_board[0] = 1
    original_board[5] = -1
    original_board[10] = 2
    
    # Create similar board for torch environments
    torch_board[0] = 1
    torch_board[5] = -1
    torch_board[10] = 2
    
    original_times = []
    torch_times = []
    torch_cpu_times = []
    speedups = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create batched board for torch environment
        if batch_size > 1:
            batched_board = torch.stack([torch_board] * batch_size)
        else:
            batched_board = torch_board
        
        # Time board rotation operations - note that original env rotation is different
        original_time = time_function(
            lambda: [original_env._rotate_board(board=original_board) for _ in range(batch_size)]
        )
        torch_time = time_function(
            lambda: torch_env.rotate_board(batched_board, 3)
        )
        torch_cpu_time = time_function(
            lambda: torch_cpu_env.rotate_board(batched_board.to(torch.device("cpu")), 3)
        )
        
        # Normalize by batch size
        original_time /= batch_size
        torch_time /= batch_size
        torch_cpu_time /= batch_size
        
        # Calculate speedup
        speedup = original_time / torch_time
        
        print(f"  Original: {original_time:.6f}s per rotation")
        print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s per rotation")
        print(f"  PyTorch (MPS): {torch_time:.6f}s per rotation")
        print(f"  Speedup (MPS vs Original): {speedup:.2f}x")
        
        original_times.append(original_time)
        torch_times.append(torch_time)
        torch_cpu_times.append(torch_cpu_time)
        speedups.append(speedup)
    
    return {
        'batch_sizes': batch_sizes,
        'original': original_times,
        'torch_cpu': torch_cpu_times,
        'torch_mps': torch_times,
        'speedup': speedups
    }

def benchmark_check_block_rule(batch_sizes: List[int] = [1, 10, 100, 1000]) -> Dict[str, List[float]]:
    """
    Benchmark the check_block_rule operation for both environments with different batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with timing results
    """
    print("\nBenchmarking check_block_rule with different batch sizes...")
    
    # Create environments
    original_env = OptimizedNardeEnv()
    torch_env = TorchNardeEnv(use_acceleration=True)
    torch_cpu_env = TorchNardeEnv(use_acceleration=False)
    
    # Reset environments
    original_env.reset()
    torch_env.reset()
    torch_cpu_env.reset()
    
    # Get boards
    original_board = original_env.board
    torch_board = torch_env.board
    
    # Ensure the boards have some pieces for meaningful block rule checking
    original_board[0:6] = 1  # Create a potential block with 6 consecutive pieces
    
    # Create similar pattern for torch environments
    for i in range(6):
        torch_board[i] = 1
    
    original_times = []
    torch_times = []
    torch_cpu_times = []
    speedups = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create batched board for torch environment
        if batch_size > 1:
            batched_board = torch.stack([torch_board] * batch_size)
            batched_mode = True
        else:
            batched_board = torch_board
            batched_mode = False
        
        # Time check_block_rule operations
        original_time = time_function(
            lambda: [original_env.check_block_rule(board=original_board, player=1, use_jax=False) for _ in range(batch_size)]
        )
        torch_time = time_function(
            lambda: torch_env.check_block_rule(batched_board, batch_mode=batched_mode)
        )
        torch_cpu_time = time_function(
            lambda: torch_cpu_env.check_block_rule(batched_board.to(torch.device("cpu")), batch_mode=batched_mode)
        )
        
        # Normalize by batch size
        original_time /= batch_size
        torch_time /= batch_size
        torch_cpu_time /= batch_size
        
        # Calculate speedup
        speedup = original_time / torch_time
        
        print(f"  Original: {original_time:.6f}s per check")
        print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s per check")
        print(f"  PyTorch (MPS): {torch_time:.6f}s per check")
        print(f"  Speedup (MPS vs Original): {speedup:.2f}x")
        
        original_times.append(original_time)
        torch_times.append(torch_time)
        torch_cpu_times.append(torch_cpu_time)
        speedups.append(speedup)
    
    return {
        'batch_sizes': batch_sizes,
        'original': original_times,
        'torch_cpu': torch_cpu_times,
        'torch_mps': torch_times,
        'speedup': speedups
    }

def benchmark_episode() -> Dict[str, float]:
    """
    Benchmark a full episode run for both environments.
    
    Returns:
        Dictionary with timing results
    """
    print("\nBenchmarking full episode run...")
    
    # Define a function to run an episode with the original environment
    def run_original_episode():
        env = OptimizedNardeEnv()
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            valid_actions = env.valid_actions
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                break
            steps += 1
    
    # Define a function to run an episode with the PyTorch environment
    def run_torch_episode(use_acceleration=True):
        env = TorchNardeEnv(use_acceleration=use_acceleration)
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            valid_actions = env.get_valid_actions().cpu().numpy()
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                break
            steps += 1
    
    # Time episode runs
    original_time = time_function(run_original_episode)
    torch_time = time_function(lambda: run_torch_episode(use_acceleration=True))
    torch_cpu_time = time_function(lambda: run_torch_episode(use_acceleration=False))
    
    # Calculate speedup
    speedup = original_time / torch_time
    
    print(f"  Original: {original_time:.6f}s per episode")
    print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s per episode")
    print(f"  PyTorch (MPS): {torch_time:.6f}s per episode")
    print(f"  Speedup (MPS vs Original): {speedup:.2f}x")
    
    return {
        'original': original_time,
        'torch_cpu': torch_cpu_time,
        'torch_mps': torch_time,
        'speedup': speedup
    }

def plot_benchmark_results(results: Dict[str, Any], test_name: str, output_file: str = None):
    """
    Plot benchmark results.
    
    Args:
        results: Benchmark results
        test_name: Name of the benchmark test
        output_file: Optional file to save the plot
    """
    if 'batch_sizes' not in results:
        # Not a batch test, skip plotting
        return
    
    batch_sizes = results['batch_sizes']
    original_times = results['original']
    torch_cpu_times = results['torch_cpu']
    torch_mps_times = results['torch_mps']
    speedups = results['speedup']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot execution times
    ax1.plot(batch_sizes, original_times, 'o-', label='Original')
    ax1.plot(batch_sizes, torch_cpu_times, 's-', label='PyTorch (CPU)')
    ax1.plot(batch_sizes, torch_mps_times, '^-', label='PyTorch (MPS)')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (s)')
    ax1.set_title(f'{test_name} - Execution Time')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot speedup
    ax2.plot(batch_sizes, speedups, 'o-', color='green')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (vs Original)')
    ax2.set_title(f'{test_name} - Speedup')
    ax2.set_xscale('log')
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def generate_report(results: Dict[str, Dict[str, Any]]):
    """
    Generate a report with benchmark results.
    
    Args:
        results: Dictionary with benchmark results
    """
    # Create or overwrite report file
    with open('narde_env_benchmark_report.md', 'w') as f:
        f.write("# Narde Environment Benchmark Report\n\n")
        
        f.write("## Hardware Acceleration Results\n\n")
        f.write("This report compares the performance of the original NumPy-based Narde environment ")
        f.write("with the PyTorch-accelerated version using both CPU and MPS (Metal Performance Shaders).\n\n")
        
        # Environment reset
        f.write("### Environment Reset\n\n")
        f.write("| Implementation | Time (s) | Speedup vs Original |\n")
        f.write("|----------------|----------|-----------------|\n")
        
        reset_results = results['reset']
        f.write(f"| Original | {reset_results['original']:.6f} | 1.00x |\n")
        f.write(f"| PyTorch (CPU) | {reset_results['torch_cpu']:.6f} | {reset_results['original']/reset_results['torch_cpu']:.2f}x |\n")
        f.write(f"| PyTorch (MPS) | {reset_results['torch_mps']:.6f} | {reset_results['speedup']:.2f}x |\n\n")
        
        # Full episode
        f.write("### Full Episode Run\n\n")
        f.write("| Implementation | Time (s) | Speedup vs Original |\n")
        f.write("|----------------|----------|-----------------|\n")
        
        episode_results = results['episode']
        f.write(f"| Original | {episode_results['original']:.6f} | 1.00x |\n")
        f.write(f"| PyTorch (CPU) | {episode_results['torch_cpu']:.6f} | {episode_results['original']/episode_results['torch_cpu']:.2f}x |\n")
        f.write(f"| PyTorch (MPS) | {episode_results['torch_mps']:.6f} | {episode_results['speedup']:.2f}x |\n\n")
        
        # Operation benchmarks
        operations = {
            'valid_actions': 'Get Valid Actions',
            'board_rotation': 'Board Rotation',
            'block_rule': 'Check Block Rule'
        }
        
        for op_key, op_name in operations.items():
            if op_key in results:
                op_results = results[op_key]
                batch_sizes = op_results['batch_sizes']
                
                f.write(f"### {op_name}\n\n")
                f.write("| Batch Size | Original (s) | PyTorch CPU (s) | PyTorch MPS (s) | Speedup (MPS vs Original) |\n")
                f.write("|------------|--------------|-----------------|----------------|--------------------------|\n")
                
                for i, batch_size in enumerate(batch_sizes):
                    f.write(f"| {batch_size} | {op_results['original'][i]:.6f} | {op_results['torch_cpu'][i]:.6f} | ")
                    f.write(f"{op_results['torch_mps'][i]:.6f} | {op_results['speedup'][i]:.2f}x |\n")
                
                f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Find best speedups
        best_speedups = {}
        for op_key in operations.keys():
            if op_key in results:
                op_results = results[op_key]
                best_idx = np.argmax(op_results['speedup'])
                best_speedups[op_key] = {
                    'speedup': op_results['speedup'][best_idx],
                    'batch_size': op_results['batch_sizes'][best_idx]
                }
        
        f.write("The PyTorch-accelerated environment with MPS shows significant performance improvements for operations with larger batch sizes:\n\n")
        
        for op_key, op_name in operations.items():
            if op_key in best_speedups:
                best = best_speedups[op_key]
                f.write(f"- {op_name}: {best['speedup']:.2f}x speedup at batch size {best['batch_size']}\n")
        
        f.write(f"\nFor full episode runs, the PyTorch MPS implementation is {episode_results['speedup']:.2f}x faster than the original implementation.\n\n")
        
        # Recommendations
        f.write("### Recommendations\n\n")
        f.write("Based on the benchmark results, we recommend:\n\n")
        
        if any(best['speedup'] > 1.0 for best in best_speedups.values()):
            # Find crossover batch size
            crossover_batch_sizes = {}
            for op_key, op_name in operations.items():
                if op_key in results:
                    op_results = results[op_key]
                    # Find smallest batch size where MPS is faster than original
                    for i, batch_size in enumerate(op_results['batch_sizes']):
                        if op_results['speedup'][i] > 1.0:
                            crossover_batch_sizes[op_key] = batch_size
                            break
            
            f.write("1. Use the PyTorch MPS implementation for batch operations with the following batch size thresholds:\n\n")
            
            for op_key, op_name in operations.items():
                if op_key in crossover_batch_sizes:
                    f.write(f"   - {op_name}: Batch size ≥ {crossover_batch_sizes[op_key]}\n")
            
            f.write("\n2. For operations with small batch sizes, the CPU implementation may be more efficient.\n\n")
            f.write("3. Set a dynamic threshold system in the environment that chooses the appropriate device (CPU vs MPS) based on the operation and batch size.\n")
        else:
            f.write("1. Stick with the CPU implementation for all operations, as it provides better overall performance in this environment.\n")
            f.write("2. Consider optimizing the PyTorch implementation further to achieve better MPS acceleration.\n")
        
        print(f"Report generated: narde_env_benchmark_report.md")

def main():
    """Main function to run all benchmarks."""
    print("Starting Narde Environment Benchmarks")
    print("=====================================")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define batch sizes to test
    batch_sizes = [1, 8, 32, 128, 512, 1024]
    
    # Run benchmarks
    results = {}
    
    # Basic benchmarks
    results['reset'] = benchmark_reset()
    results['episode'] = benchmark_episode()
    
    # Operation benchmarks
    results['valid_actions'] = benchmark_get_valid_actions(batch_sizes)
    results['board_rotation'] = benchmark_board_rotation(batch_sizes)
    results['block_rule'] = benchmark_check_block_rule(batch_sizes)
    
    # Plot results
    plot_benchmark_results(results['valid_actions'], 'Get Valid Actions', 'valid_actions_benchmark.png')
    plot_benchmark_results(results['board_rotation'], 'Board Rotation', 'board_rotation_benchmark.png')
    plot_benchmark_results(results['block_rule'], 'Check Block Rule', 'block_rule_benchmark.png')
    
    # Generate report
    generate_report(results)
    
    print("\nBenchmarks completed!")

if __name__ == "__main__":
    main() 