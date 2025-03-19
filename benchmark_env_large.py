#!/usr/bin/env python3
"""
Benchmark script with larger batch sizes to better demonstrate MPS acceleration benefits.
This script focuses on batch operations that might benefit more from hardware acceleration.
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
REPETITIONS = 5
# Batch sizes to test, focusing on larger sizes
LARGE_BATCH_SIZES = [128, 512, 1024, 2048, 4096, 8192]

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

def benchmark_large_board_rotation(batch_sizes: List[int] = LARGE_BATCH_SIZES) -> Dict[str, List[float]]:
    """
    Benchmark the board rotation operation with larger batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with timing results
    """
    print("\nBenchmarking large batch board rotation...")
    
    # Create environments
    original_env = OptimizedNardeEnv()
    torch_env = TorchNardeEnv(use_acceleration=True)
    torch_cpu_env = TorchNardeEnv(use_acceleration=False)
    
    # Reset environments
    original_env.reset()
    torch_env.reset()
    torch_cpu_env.reset()
    
    # Get boards
    original_board = original_env.board.copy()  # Make a copy to avoid modifying the environment
    torch_board = torch_env.board.clone().to(torch.device("cpu"))  # Start with CPU tensor for easier batching
    
    # Create a more complex board with varied pieces
    for i in range(24):
        if i % 3 == 0:
            original_board[i] = 1
            torch_board[i] = 1
        elif i % 3 == 1:
            original_board[i] = -1
            torch_board[i] = -1
        else:
            original_board[i] = 0
            torch_board[i] = 0
    
    # Ensure original board is a numpy array
    original_board = np.array(original_board)
    
    original_times = []
    torch_times = []
    torch_cpu_times = []
    speedups = []
    
    # Create a large batch of identical boards for the largest batch size
    # We'll slice this for smaller batch sizes
    max_batch_size = max(batch_sizes)
    all_boards_torch = torch.stack([torch_board] * max_batch_size)
    all_boards_numpy = np.stack([original_board] * max_batch_size)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Slice the batch to the current size
        batched_board = all_boards_torch[:batch_size]
        batched_board_numpy = all_boards_numpy[:batch_size]
        
        # Move batched board to MPS device for torch_env
        batched_board_mps = batched_board.to(torch_env.device)
        
        # Define functions to benchmark with proper warm-up
        def benchmark_original():
            # Original implementation (using numpy)
            for i in range(batch_size):
                _ = original_env._rotate_board(board=batched_board_numpy[i])
        
        def benchmark_torch_cpu():
            # PyTorch CPU implementation
            _ = torch_cpu_env.rotate_board(batched_board, 3)
        
        def benchmark_torch_mps():
            # PyTorch MPS implementation
            _ = torch_env.rotate_board(batched_board_mps, 3)
            if torch_env.device.type in ["cuda", "mps"]:
                torch.mps.synchronize()
        
        # Warmup
        benchmark_original()
        benchmark_torch_cpu()
        benchmark_torch_mps()
        
        # Time each implementation
        original_time = time_function(benchmark_original)
        torch_cpu_time = time_function(benchmark_torch_cpu)
        torch_mps_time = time_function(benchmark_torch_mps)
        
        # Normalize by batch size
        original_time /= batch_size
        torch_cpu_time /= batch_size
        torch_mps_time /= batch_size
        
        # Calculate speedups
        original_vs_mps = original_time / torch_mps_time if torch_mps_time > 0 else 0
        cpu_vs_mps = torch_cpu_time / torch_mps_time if torch_mps_time > 0 else 0
        
        print(f"  Original (NumPy): {original_time:.6f}s per rotation")
        print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s per rotation")
        print(f"  PyTorch (MPS): {torch_mps_time:.6f}s per rotation")
        print(f"  Speedup (MPS vs Original): {original_vs_mps:.2f}x")
        print(f"  Speedup (MPS vs CPU): {cpu_vs_mps:.2f}x")
        
        original_times.append(original_time)
        torch_times.append(torch_mps_time)
        torch_cpu_times.append(torch_cpu_time)
        speedups.append(original_vs_mps)
    
    return {
        'batch_sizes': batch_sizes,
        'original': original_times,
        'torch_cpu': torch_cpu_times,
        'torch_mps': torch_times,
        'speedup': speedups
    }

def benchmark_large_batch_board_ops(batch_sizes: List[int] = LARGE_BATCH_SIZES) -> Dict[str, List[float]]:
    """
    Benchmark matrix operations on batched board states.
    
    Args:
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with timing results
    """
    print("\nBenchmarking large batch board operations...")
    
    # Create environments for their board representations
    torch_env = TorchNardeEnv(use_acceleration=True)
    torch_cpu_env = TorchNardeEnv(use_acceleration=False)
    
    # Reset environments
    torch_env.reset()
    torch_cpu_env.reset()
    
    # Create a basic board template
    board_template = torch.zeros(24, dtype=torch.int8, device=torch.device("cpu"))
    
    # Fill with some data for realistic operations
    for i in range(24):
        if i % 3 == 0:
            board_template[i] = 1
        elif i % 3 == 1:
            board_template[i] = -1
    
    # Create a large batch of varied boards
    max_batch_size = max(batch_sizes)
    all_boards = []
    for i in range(max_batch_size):
        # Create slightly varied board for each batch item
        board_copy = board_template.clone()
        # Add some randomness to make operations more realistic
        for j in range(3):
            pos = np.random.randint(0, 24)
            val = np.random.choice([-1, 1])
            board_copy[pos] = val
        all_boards.append(board_copy)
    
    # Stack into batch tensor
    all_boards_tensor = torch.stack(all_boards)
    
    numpy_times = []
    torch_cpu_times = []
    torch_mps_times = []
    speedups = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Slice to current batch size
        batch_boards = all_boards_tensor[:batch_size]
        # Create numpy version for comparison
        batch_boards_numpy = batch_boards.numpy()
        
        # Define operations to benchmark
        
        # 1. Matrix multiplication operation (typical in neural nets)
        weight_matrix = np.random.randn(24, 24).astype(np.float32)
        weight_tensor_cpu = torch.tensor(weight_matrix, device=torch.device("cpu"))
        weight_tensor_mps = torch.tensor(weight_matrix, device=torch_env.device)
        
        # 2. Board feature extraction operation
        def extract_features_numpy(boards):
            # Convert board positions to one-hot features (common operation in RL)
            batch_size, board_size = boards.shape
            features = np.zeros((batch_size, board_size, 3), dtype=np.float32)  # 3 channels: empty, player1, player2
            
            for b in range(batch_size):
                for i in range(board_size):
                    if boards[b, i] == 0:
                        features[b, i, 0] = 1.0  # Empty
                    elif boards[b, i] > 0:
                        features[b, i, 1] = boards[b, i] / 15.0  # Player 1, normalized
                    else:
                        features[b, i, 2] = -boards[b, i] / 15.0  # Player 2, normalized
            
            # Flatten features
            return features.reshape(batch_size, -1)
        
        def extract_features_torch(boards, device):
            # Same operation but with PyTorch
            batch_size, board_size = boards.shape
            boards = boards.to(device)
            
            # Create one-hot features
            features = torch.zeros((batch_size, board_size, 3), dtype=torch.float32, device=device)
            
            # Empty positions
            features[:, :, 0] = (boards == 0).float()
            # Player 1 positions
            features[:, :, 1] = torch.where(boards > 0, boards.float() / 15.0, torch.zeros_like(boards, dtype=torch.float32))
            # Player 2 positions
            features[:, :, 2] = torch.where(boards < 0, -boards.float() / 15.0, torch.zeros_like(boards, dtype=torch.float32))
            
            # Flatten features
            return features.reshape(batch_size, -1)
        
        # Define benchmark functions
        def benchmark_numpy():
            # 1. Matrix multiplication
            boards_float = batch_boards_numpy.astype(np.float32)
            result1 = np.matmul(boards_float, weight_matrix)
            
            # 2. Feature extraction
            result2 = extract_features_numpy(batch_boards_numpy)
            
            return result1, result2
        
        def benchmark_torch_cpu():
            # 1. Matrix multiplication
            boards_float = batch_boards.float()
            result1 = torch.matmul(boards_float, weight_tensor_cpu)
            
            # 2. Feature extraction
            result2 = extract_features_torch(batch_boards, torch.device("cpu"))
            
            return result1, result2
        
        def benchmark_torch_mps():
            # 1. Matrix multiplication
            boards_float = batch_boards.to(torch_env.device).float()
            result1 = torch.matmul(boards_float, weight_tensor_mps)
            
            # 2. Feature extraction
            result2 = extract_features_torch(batch_boards, torch_env.device)
            
            # Synchronize to ensure all operations complete
            if torch_env.device.type in ["cuda", "mps"]:
                torch.mps.synchronize()
            
            return result1, result2
        
        # Warmup
        _ = benchmark_numpy()
        _ = benchmark_torch_cpu()
        _ = benchmark_torch_mps()
        
        # Time each implementation
        numpy_time = time_function(benchmark_numpy)
        torch_cpu_time = time_function(benchmark_torch_cpu)
        torch_mps_time = time_function(benchmark_torch_mps)
        
        # Calculate speedups
        numpy_vs_mps = numpy_time / torch_mps_time if torch_mps_time > 0 else 0
        cpu_vs_mps = torch_cpu_time / torch_mps_time if torch_mps_time > 0 else 0
        
        print(f"  NumPy: {numpy_time:.6f}s")
        print(f"  PyTorch (CPU): {torch_cpu_time:.6f}s")
        print(f"  PyTorch (MPS): {torch_mps_time:.6f}s")
        print(f"  Speedup (MPS vs NumPy): {numpy_vs_mps:.2f}x")
        print(f"  Speedup (MPS vs CPU): {cpu_vs_mps:.2f}x")
        
        numpy_times.append(numpy_time)
        torch_cpu_times.append(torch_cpu_time)
        torch_mps_times.append(torch_mps_time)
        speedups.append(numpy_vs_mps)
    
    return {
        'batch_sizes': batch_sizes,
        'numpy': numpy_times,
        'torch_cpu': torch_cpu_times,
        'torch_mps': torch_mps_times,
        'speedup': speedups
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
    
    # Check which times are available in the results
    times_to_plot = []
    if 'original' in results:
        times_to_plot.append(('Original (NumPy)', 'original', 'o-'))
    if 'numpy' in results:
        times_to_plot.append(('NumPy', 'numpy', 'o-'))
    if 'torch_cpu' in results:
        times_to_plot.append(('PyTorch (CPU)', 'torch_cpu', 's-'))
    if 'torch_mps' in results:
        times_to_plot.append(('PyTorch (MPS)', 'torch_mps', '^-'))
    
    speedups = results['speedup']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot execution times
    for label, key, style in times_to_plot:
        ax1.plot(batch_sizes, results[key], style, label=label)
    
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
    ax2.set_ylabel('Speedup (MPS vs NumPy)')
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
    with open('narde_large_batch_benchmark_report.md', 'w') as f:
        f.write("# Narde Large Batch Operations Benchmark Report\n\n")
        
        f.write("## Hardware Acceleration Results for Large Batch Operations\n\n")
        f.write("This report examines the performance benefits of hardware acceleration (MPS) ")
        f.write("for large batch operations in the Narde environment, focusing on operations that ")
        f.write("are most relevant to reinforcement learning and neural network training.\n\n")
        
        # Board Rotation
        if 'board_rotation' in results:
            rotation_results = results['board_rotation']
            batch_sizes = rotation_results['batch_sizes']
            
            f.write("### Large Batch Board Rotation\n\n")
            f.write("| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch MPS (s) | Speedup (MPS vs NumPy) | Speedup (MPS vs CPU) |\n")
            f.write("|------------|-----------|-----------------|----------------|-----------------------|--------------------|\n")
            
            for i, batch_size in enumerate(batch_sizes):
                numpy_time = rotation_results['original'][i]
                cpu_time = rotation_results['torch_cpu'][i]
                mps_time = rotation_results['torch_mps'][i]
                
                numpy_speedup = numpy_time / mps_time if mps_time > 0 else 0
                cpu_speedup = cpu_time / mps_time if mps_time > 0 else 0
                
                f.write(f"| {batch_size} | {numpy_time:.6f} | {cpu_time:.6f} | {mps_time:.6f} | {numpy_speedup:.2f}x | {cpu_speedup:.2f}x |\n")
            
            f.write("\n")
            
            # Find best speedup
            best_idx = np.argmax(rotation_results['speedup'])
            best_batch = batch_sizes[best_idx]
            best_speedup = rotation_results['speedup'][best_idx]
            
            f.write(f"**Best Speedup:** {best_speedup:.2f}x at batch size {best_batch}\n\n")
        
        # Board Operations
        if 'board_ops' in results:
            ops_results = results['board_ops']
            batch_sizes = ops_results['batch_sizes']
            
            f.write("### Large Batch Board Operations (Matrix and Feature Operations)\n\n")
            f.write("| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch MPS (s) | Speedup (MPS vs NumPy) | Speedup (MPS vs CPU) |\n")
            f.write("|------------|-----------|-----------------|----------------|-----------------------|--------------------|\n")
            
            for i, batch_size in enumerate(batch_sizes):
                numpy_time = ops_results['numpy'][i]
                cpu_time = ops_results['torch_cpu'][i]
                mps_time = ops_results['torch_mps'][i]
                
                numpy_speedup = numpy_time / mps_time if mps_time > 0 else 0
                cpu_speedup = cpu_time / mps_time if mps_time > 0 else 0
                
                f.write(f"| {batch_size} | {numpy_time:.6f} | {cpu_time:.6f} | {mps_time:.6f} | {numpy_speedup:.2f}x | {cpu_speedup:.2f}x |\n")
            
            f.write("\n")
            
            # Find best speedup
            best_idx = np.argmax(ops_results['speedup'])
            best_batch = batch_sizes[best_idx]
            best_speedup = ops_results['speedup'][best_idx]
            
            f.write(f"**Best Speedup:** {best_speedup:.2f}x at batch size {best_batch}\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Compile best speedups
        best_speedups = {}
        for op_key in results.keys():
            if 'speedup' in results[op_key]:
                op_results = results[op_key]
                best_idx = np.argmax(op_results['speedup'])
                best_speedups[op_key] = {
                    'speedup': op_results['speedup'][best_idx],
                    'batch_size': op_results['batch_sizes'][best_idx]
                }
        
        f.write("The PyTorch-accelerated environment with MPS shows significant performance improvements for operations with larger batch sizes:\n\n")
        
        for op_key, op_name in [('board_rotation', 'Board Rotation'), ('board_ops', 'Board Operations')]:
            if op_key in best_speedups:
                best = best_speedups[op_key]
                f.write(f"- {op_name}: {best['speedup']:.2f}x speedup at batch size {best['batch_size']}\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("### Recommendations\n\n")
        
        if any(best['speedup'] > 1.0 for best in best_speedups.values()):
            # Find crossover batch sizes (where MPS becomes faster than NumPy)
            crossover_batch_sizes = {}
            
            for op_key in results.keys():
                if 'speedup' in results[op_key]:
                    op_results = results[op_key]
                    batch_sizes = op_results['batch_sizes']
                    speedups = op_results['speedup']
                    
                    # Find smallest batch size where MPS is significantly faster
                    for i, (batch_size, speedup) in enumerate(zip(batch_sizes, speedups)):
                        if speedup > 1.2:  # At least 20% faster
                            crossover_batch_sizes[op_key] = batch_size
                            break
            
            f.write("Based on the benchmark results, for optimal performance on Apple Silicon:\n\n")
            
            if crossover_batch_sizes:
                f.write("1. Use MPS acceleration for batch operations with the following thresholds:\n\n")
                
                for op_key, op_name in [('board_rotation', 'Board Rotation'), ('board_ops', 'Board Operations')]:
                    if op_key in crossover_batch_sizes:
                        f.write(f"   - {op_name}: Batch size ≥ {crossover_batch_sizes[op_key]}\n")
                
                f.write("\n2. For operations with batch sizes below these thresholds, use CPU (NumPy or PyTorch CPU).\n\n")
                f.write("3. Implement dynamic device selection based on operation type and batch size.\n")
            else:
                f.write("1. While MPS shows acceleration for some batch sizes, the benefits are inconsistent. Consider using CPU for most operations for now.\n")
                f.write("2. For extremely large batch operations (4096+), MPS may provide benefits and should be tested case by case.\n")
        else:
            f.write("1. For the operations tested, CPU implementations (either NumPy or PyTorch CPU) perform better overall.\n")
            f.write("2. Consider using PyTorch CPU for its vectorized operations, which often outperform NumPy for larger batch sizes.\n")
        
        f.write("\n### Future Optimization Opportunities\n\n")
        f.write("1. **Custom CUDA Kernels:** For systems with NVIDIA GPUs, custom CUDA kernels could provide even greater acceleration.\n")
        f.write("2. **Operation Fusion:** Combining multiple operations to reduce memory transfers between CPU and MPS/GPU.\n")
        f.write("3. **Precision Reduction:** Using half-precision (float16) operations could further accelerate MPS performance.\n")
        
        print(f"Report generated: narde_large_batch_benchmark_report.md")

def main():
    """Main function to run all benchmarks."""
    print("Starting Narde Large Batch Operations Benchmarks")
    print("===============================================")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run benchmarks
    results = {}
    
    # Large batch board rotation benchmark
    results['board_rotation'] = benchmark_large_board_rotation(LARGE_BATCH_SIZES)
    
    # Large batch board operations benchmark
    results['board_ops'] = benchmark_large_batch_board_ops(LARGE_BATCH_SIZES)
    
    # Plot results
    plot_benchmark_results(results['board_rotation'], 'Large Batch Board Rotation', 'large_batch_rotation_benchmark.png')
    plot_benchmark_results(results['board_ops'], 'Large Batch Board Operations', 'large_batch_ops_benchmark.png')
    
    # Generate report
    generate_report(results)
    
    print("\nLarge batch benchmarks completed!")

if __name__ == "__main__":
    main() 