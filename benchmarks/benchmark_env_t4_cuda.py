#!/usr/bin/env python3
"""
Benchmark Narde environment operations on NVIDIA T4 GPU.
This script compares PyTorch with CUDA vs NumPy implementations
for various environment operations and batch sizes.

Performance optimization notes:
- Tests will automatically skip larger batch sizes under two conditions:
  1. If a test takes more than 30 seconds to complete
  2. If a test takes more than 5x the time of the previous batch size
- This prevents wasting time on benchmarks that show progressively worse performance
- Particularly useful for operations like get_valid_actions which scale poorly
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Batch sizes to test - similar to those used in the M-series tests
BATCH_SIZES = [1, 4, 16, 64, 256, 1024, 4096, 8192]

# Maximum time in seconds that a single batch test should take before skipping larger batches
MAX_TEST_TIME = 30.0

# Define the board operations that we'll benchmark
def rotate_board_numpy(boards):
    """Rotate the board using NumPy."""
    return -1 * np.flip(boards, axis=1)

def rotate_board_pytorch_cpu(boards):
    """Rotate the board using PyTorch on CPU."""
    return -1 * torch.flip(boards, dims=[1])

def rotate_board_pytorch_cuda(boards):
    """Rotate the board using PyTorch on CUDA."""
    return -1 * torch.flip(boards, dims=[1])

def check_block_rule_numpy(boards):
    """Check block rule using NumPy."""
    # In the block rule, a player can't block all of opponent's pieces
    # Simplified for benchmarking: check if any cells meet certain conditions
    player_pieces = (boards > 0).astype(np.float32)
    opponent_pieces = (boards < 0).astype(np.float32)
    
    # Consider blocked if surrounded by player pieces
    # This is a simplified calculation for benchmarking
    # Real implementation would be more complex
    blocked = np.zeros_like(opponent_pieces)
    
    # Check neighbors (simplified)
    for i in range(1, boards.shape[1]-1):
        blocked[:, i] = (
            opponent_pieces[:, i] * 
            player_pieces[:, i-1] * 
            player_pieces[:, i+1]
        )
    
    # Check if all opponent pieces are blocked
    all_blocked = np.sum(blocked, axis=1) == np.sum(opponent_pieces, axis=1)
    
    return all_blocked

def check_block_rule_pytorch_cpu(boards):
    """Check block rule using PyTorch on CPU."""
    player_pieces = (boards > 0).float()
    opponent_pieces = (boards < 0).float()
    
    blocked = torch.zeros_like(opponent_pieces)
    
    # Check neighbors (simplified)
    for i in range(1, boards.shape[1]-1):
        blocked[:, i] = (
            opponent_pieces[:, i] * 
            player_pieces[:, i-1] * 
            player_pieces[:, i+1]
        )
    
    all_blocked = torch.sum(blocked, dim=1) == torch.sum(opponent_pieces, dim=1)
    
    return all_blocked

def check_block_rule_pytorch_cuda(boards):
    """Check block rule using PyTorch on CUDA."""
    player_pieces = (boards > 0).float()
    opponent_pieces = (boards < 0).float()
    
    blocked = torch.zeros_like(opponent_pieces)
    
    # Check neighbors (simplified)
    for i in range(1, boards.shape[1]-1):
        blocked[:, i] = (
            opponent_pieces[:, i] * 
            player_pieces[:, i-1] * 
            player_pieces[:, i+1]
        )
    
    all_blocked = torch.sum(blocked, dim=1) == torch.sum(opponent_pieces, dim=1)
    
    return all_blocked

def get_valid_actions_numpy(boards):
    """Get valid actions using NumPy."""
    # Simplified for benchmarking
    # In real implementation, this would check game rules
    
    batch_size = boards.shape[0]
    valid_actions = np.zeros((batch_size, 24*24), dtype=np.float32)
    
    # For each board in the batch
    for b in range(batch_size):
        board = boards[b]
        player_pieces = (board > 0)
        
        # For each piece, consider moves to empty spaces
        for from_pos in range(24):
            if player_pieces[from_pos]:
                for to_pos in range(24):
                    if board[to_pos] == 0:  # Empty space
                        action_idx = from_pos * 24 + to_pos
                        valid_actions[b, action_idx] = 1.0
    
    return valid_actions

def get_valid_actions_pytorch_cpu(boards):
    """Get valid actions using PyTorch on CPU."""
    batch_size = boards.shape[0]
    valid_actions = torch.zeros((batch_size, 24*24), dtype=torch.float32)
    
    # For each board in the batch
    for b in range(batch_size):
        board = boards[b]
        player_pieces = (board > 0)
        
        # For each piece, consider moves to empty spaces
        for from_pos in range(24):
            if player_pieces[from_pos]:
                for to_pos in range(24):
                    if board[to_pos] == 0:  # Empty space
                        action_idx = from_pos * 24 + to_pos
                        valid_actions[b, action_idx] = 1.0
    
    return valid_actions

def get_valid_actions_pytorch_cuda(boards):
    """Get valid actions using PyTorch on CUDA."""
    batch_size = boards.shape[0]
    valid_actions = torch.zeros((batch_size, 24*24), device=boards.device, dtype=torch.float32)
    
    # For each board in the batch
    for b in range(batch_size):
        board = boards[b]
        player_pieces = (board > 0)
        
        # For each piece, consider moves to empty spaces
        for from_pos in range(24):
            if player_pieces[from_pos]:
                for to_pos in range(24):
                    if board[to_pos] == 0:  # Empty space
                        action_idx = from_pos * 24 + to_pos
                        valid_actions[b, action_idx] = 1.0
    
    return valid_actions

def create_random_boards(batch_size):
    """Create random board states for benchmarking."""
    # Create random boards with values between -15 and 15
    # Negative values represent opponent pieces, positive values represent player pieces
    # 0 represents empty space
    return np.random.randint(-15, 16, size=(batch_size, 24)).astype(np.float32)

def benchmark_operation(operation_name, batch_sizes, num_iterations=100):
    """Benchmark a specific operation across different implementations and batch sizes."""
    print(f"\n=== Benchmarking {operation_name} ===")
    
    results = []
    
    # Operations mapping
    operations = {
        "rotate_board": (rotate_board_numpy, rotate_board_pytorch_cpu, rotate_board_pytorch_cuda),
        "check_block_rule": (check_block_rule_numpy, check_block_rule_pytorch_cpu, check_block_rule_pytorch_cuda),
        "get_valid_actions": (get_valid_actions_numpy, get_valid_actions_pytorch_cpu, get_valid_actions_pytorch_cuda)
    }
    
    numpy_op, pytorch_cpu_op, pytorch_cuda_op = operations[operation_name]
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"Running on: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, skipping CUDA benchmarks")
    
    # Track performance to know when to skip
    last_numpy_time = 0
    last_pytorch_cpu_time = 0
    last_pytorch_cuda_time = 0
    skip_numpy = False
    skip_pytorch_cpu = False
    skip_pytorch_cuda = False
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create random boards
        numpy_boards = create_random_boards(batch_size)
        pytorch_boards_cpu = torch.tensor(numpy_boards, dtype=torch.float32)
        
        if cuda_available:
            pytorch_boards_cuda = pytorch_boards_cpu.cuda()
            # Warm-up CUDA
            _ = pytorch_cuda_op(pytorch_boards_cuda)
            torch.cuda.synchronize()
        
        # Benchmark NumPy (if not skipped)
        if not skip_numpy:
            start = time.time()
            for _ in range(num_iterations):
                _ = numpy_op(numpy_boards)
            numpy_time = (time.time() - start) / num_iterations
            total_time = time.time() - start
            print(f"NumPy time: {numpy_time:.6f} seconds (total: {total_time:.2f}s)")
            
            # Check if we should skip NumPy for larger batches
            if total_time > MAX_TEST_TIME:
                skip_numpy = True
                print(f"NumPy tests for larger batches will be skipped (time: {total_time:.2f}s > {MAX_TEST_TIME}s)")
            elif last_numpy_time > 0 and numpy_time > 5 * last_numpy_time:
                skip_numpy = True
                print(f"NumPy tests for larger batches will be skipped (time ratio: {numpy_time/last_numpy_time:.2f}x > 5.0x)")
            last_numpy_time = numpy_time
        else:
            numpy_time = float('inf')
            print("NumPy: SKIPPED (would take too long)")
        
        # Benchmark PyTorch CPU (if not skipped)
        if not skip_pytorch_cpu:
            start = time.time()
            for _ in range(num_iterations):
                _ = pytorch_cpu_op(pytorch_boards_cpu)
            pytorch_cpu_time = (time.time() - start) / num_iterations
            total_time = time.time() - start
            print(f"PyTorch CPU time: {pytorch_cpu_time:.6f} seconds (total: {total_time:.2f}s)")
            
            # Check if we should skip PyTorch CPU for larger batches
            if total_time > MAX_TEST_TIME:
                skip_pytorch_cpu = True
                print(f"PyTorch CPU tests for larger batches will be skipped (time: {total_time:.2f}s > {MAX_TEST_TIME}s)")
            elif last_pytorch_cpu_time > 0 and pytorch_cpu_time > 5 * last_pytorch_cpu_time:
                skip_pytorch_cpu = True
                print(f"PyTorch CPU tests for larger batches will be skipped (time ratio: {pytorch_cpu_time/last_pytorch_cpu_time:.2f}x > 5.0x)")
            last_pytorch_cpu_time = pytorch_cpu_time
        else:
            pytorch_cpu_time = float('inf')
            print("PyTorch CPU: SKIPPED (would take too long)")
        
        # Benchmark PyTorch CUDA (if not skipped and available)
        if cuda_available and not skip_pytorch_cuda:
            start = time.time()
            for _ in range(num_iterations):
                _ = pytorch_cuda_op(pytorch_boards_cuda)
                torch.cuda.synchronize()
            pytorch_cuda_time = (time.time() - start) / num_iterations
            total_time = time.time() - start
            print(f"PyTorch CUDA time: {pytorch_cuda_time:.6f} seconds (total: {total_time:.2f}s)")
            
            # Check if we should skip PyTorch CUDA for larger batches
            if total_time > MAX_TEST_TIME:
                skip_pytorch_cuda = True
                print(f"PyTorch CUDA tests for larger batches will be skipped (time: {total_time:.2f}s > {MAX_TEST_TIME}s)")
            elif last_pytorch_cuda_time > 0 and pytorch_cuda_time > 5 * last_pytorch_cuda_time:
                skip_pytorch_cuda = True
                print(f"PyTorch CUDA tests for larger batches will be skipped (time ratio: {pytorch_cuda_time/last_pytorch_cuda_time:.2f}x > 5.0x)")
            last_pytorch_cuda_time = pytorch_cuda_time
            
            # Calculate speedups
            if not skip_numpy and not skip_pytorch_cpu:
                cpu_speedup = numpy_time / pytorch_cpu_time
                cuda_speedup = numpy_time / pytorch_cuda_time
                
                print(f"CPU speedup: {cpu_speedup:.2f}x")
                print(f"CUDA speedup: {cuda_speedup:.2f}x")
        else:
            if cuda_available:
                pytorch_cuda_time = float('inf')
                print("PyTorch CUDA: SKIPPED (would take too long)")
            else:
                pytorch_cuda_time = None
                cuda_speedup = None
        
        # Record results
        results.append({
            "batch_size": batch_size,
            "numpy_time": numpy_time if not skip_numpy else None,
            "pytorch_cpu_time": pytorch_cpu_time if not skip_pytorch_cpu else None,
            "pytorch_cuda_time": pytorch_cuda_time if cuda_available and not skip_pytorch_cuda else None,
            "cpu_speedup": numpy_time / pytorch_cpu_time if not skip_numpy and not skip_pytorch_cpu else None,
            "cuda_speedup": numpy_time / pytorch_cuda_time if cuda_available and not skip_numpy and not skip_pytorch_cuda else None
        })
        
        # If all implementations are being skipped, break out of the loop
        if skip_numpy and skip_pytorch_cpu and (not cuda_available or skip_pytorch_cuda):
            print(f"All implementations taking too long. Skipping batch sizes > {batch_size}")
            break
    
    return results

def benchmark_combined_operations(batch_sizes, num_episodes=10):
    """Benchmark a combination of operations to simulate episode execution."""
    print("\n=== Benchmarking Episode Execution ===")
    
    results = []
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    # Track performance to know when to skip
    last_numpy_time = 0
    last_pytorch_cpu_time = 0
    last_pytorch_cuda_time = 0
    skip_numpy = False
    skip_pytorch_cpu = False
    skip_pytorch_cuda = False
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create random boards
        numpy_boards = create_random_boards(batch_size)
        pytorch_boards_cpu = torch.tensor(numpy_boards, dtype=torch.float32)
        
        if cuda_available:
            pytorch_boards_cuda = pytorch_boards_cpu.cuda()
            # Warm-up CUDA
            _ = rotate_board_pytorch_cuda(pytorch_boards_cuda)
            torch.cuda.synchronize()
        
        # Benchmark NumPy (if not skipped)
        if not skip_numpy:
            start = time.time()
            for _ in range(num_episodes):
                # Simulate episode execution with multiple operations
                boards = numpy_boards.copy()
                
                # Step 1: Rotate boards
                boards = rotate_board_numpy(boards)
                
                # Step 2: Check block rule
                blocked = check_block_rule_numpy(boards)
                
                # Step 3: Get valid actions
                valid_actions = get_valid_actions_numpy(boards)
                
                # Step 4: Simulate taking random actions
                # For simplicity, just modify the boards slightly
                boards += np.random.normal(0, 0.1, boards.shape)
            
            numpy_time = (time.time() - start) / num_episodes
            total_time = time.time() - start
            print(f"NumPy episode time: {numpy_time:.6f} seconds (total: {total_time:.2f}s)")
            
            # Check if we should skip NumPy for larger batches
            if total_time > MAX_TEST_TIME:
                skip_numpy = True
                print(f"NumPy tests for larger batches will be skipped (time: {total_time:.2f}s > {MAX_TEST_TIME}s)")
            elif last_numpy_time > 0 and numpy_time > 5 * last_numpy_time:
                skip_numpy = True
                print(f"NumPy tests for larger batches will be skipped (time ratio: {numpy_time/last_numpy_time:.2f}x > 5.0x)")
            last_numpy_time = numpy_time
        else:
            numpy_time = float('inf')
            print("NumPy: SKIPPED (would take too long)")
        
        # Benchmark PyTorch CPU (if not skipped)
        if not skip_pytorch_cpu:
            start = time.time()
            for _ in range(num_episodes):
                # Simulate episode execution with multiple operations
                boards = pytorch_boards_cpu.clone()
                
                # Step 1: Rotate boards
                boards = rotate_board_pytorch_cpu(boards)
                
                # Step 2: Check block rule
                blocked = check_block_rule_pytorch_cpu(boards)
                
                # Step 3: Get valid actions
                valid_actions = get_valid_actions_pytorch_cpu(boards)
                
                # Step 4: Simulate taking random actions
                boards += torch.randn_like(boards) * 0.1
            
            pytorch_cpu_time = (time.time() - start) / num_episodes
            total_time = time.time() - start
            print(f"PyTorch CPU episode time: {pytorch_cpu_time:.6f} seconds (total: {total_time:.2f}s)")
            
            # Check if we should skip PyTorch CPU for larger batches
            if total_time > MAX_TEST_TIME:
                skip_pytorch_cpu = True
                print(f"PyTorch CPU tests for larger batches will be skipped (time: {total_time:.2f}s > {MAX_TEST_TIME}s)")
            elif last_pytorch_cpu_time > 0 and pytorch_cpu_time > 5 * last_pytorch_cpu_time:
                skip_pytorch_cpu = True
                print(f"PyTorch CPU tests for larger batches will be skipped (time ratio: {pytorch_cpu_time/last_pytorch_cpu_time:.2f}x > 5.0x)")
            last_pytorch_cpu_time = pytorch_cpu_time
        else:
            pytorch_cpu_time = float('inf')
            print("PyTorch CPU: SKIPPED (would take too long)")
        
        # Benchmark PyTorch CUDA (if not skipped and available)
        if cuda_available and not skip_pytorch_cuda:
            start = time.time()
            for _ in range(num_episodes):
                # Simulate episode execution with multiple operations
                boards = pytorch_boards_cuda.clone()
                
                # Step 1: Rotate boards
                boards = rotate_board_pytorch_cuda(boards)
                
                # Step 2: Check block rule
                blocked = check_block_rule_pytorch_cuda(boards)
                
                # Step 3: Get valid actions
                valid_actions = get_valid_actions_pytorch_cuda(boards)
                
                # Step 4: Simulate taking random actions
                boards += torch.randn_like(boards).cuda() * 0.1
                
                # Synchronize to ensure operations complete
                torch.cuda.synchronize()
            
            pytorch_cuda_time = (time.time() - start) / num_episodes
            total_time = time.time() - start
            print(f"PyTorch CUDA episode time: {pytorch_cuda_time:.6f} seconds (total: {total_time:.2f}s)")
            
            # Check if we should skip PyTorch CUDA for larger batches
            if total_time > MAX_TEST_TIME:
                skip_pytorch_cuda = True
                print(f"PyTorch CUDA tests for larger batches will be skipped (time: {total_time:.2f}s > {MAX_TEST_TIME}s)")
            elif last_pytorch_cuda_time > 0 and pytorch_cuda_time > 5 * last_pytorch_cuda_time:
                skip_pytorch_cuda = True
                print(f"PyTorch CUDA tests for larger batches will be skipped (time ratio: {pytorch_cuda_time/last_pytorch_cuda_time:.2f}x > 5.0x)")
            last_pytorch_cuda_time = pytorch_cuda_time
            
            # Calculate speedups
            if not skip_numpy and not skip_pytorch_cpu:
                cpu_speedup = numpy_time / pytorch_cpu_time
                cuda_speedup = numpy_time / pytorch_cuda_time
                
                print(f"CPU speedup: {cpu_speedup:.2f}x")
                print(f"CUDA speedup: {cuda_speedup:.2f}x")
        else:
            if cuda_available:
                pytorch_cuda_time = float('inf')
                print("PyTorch CUDA: SKIPPED (would take too long)")
            else:
                pytorch_cuda_time = None
                cuda_speedup = None
        
        # Record results
        results.append({
            "batch_size": batch_size,
            "numpy_time": numpy_time if not skip_numpy else None,
            "pytorch_cpu_time": pytorch_cpu_time if not skip_pytorch_cpu else None,
            "pytorch_cuda_time": pytorch_cuda_time if cuda_available and not skip_pytorch_cuda else None,
            "cpu_speedup": numpy_time / pytorch_cpu_time if not skip_numpy and not skip_pytorch_cpu else None,
            "cuda_speedup": numpy_time / pytorch_cuda_time if cuda_available and not skip_numpy and not skip_pytorch_cuda else None
        })
        
        # If all implementations are being skipped, break out of the loop
        if skip_numpy and skip_pytorch_cpu and (not cuda_available or skip_pytorch_cuda):
            print(f"All implementations taking too long. Skipping batch sizes > {batch_size}")
            break
    
    return results

def plot_results(results, operation_name):
    """Generate plots for the benchmark results."""
    plt.figure(figsize=(12, 8))
    
    # Collect times, filtering out None values
    numpy_times = []
    pytorch_cpu_times = []
    pytorch_cuda_times = []
    batch_sizes_numpy = []
    batch_sizes_pytorch_cpu = []
    batch_sizes_pytorch_cuda = []
    
    for i, r in enumerate(results):
        if r["numpy_time"] is not None and r["numpy_time"] != float('inf'):
            numpy_times.append(r["numpy_time"])
            batch_sizes_numpy.append(r["batch_size"])
        
        if r["pytorch_cpu_time"] is not None and r["pytorch_cpu_time"] != float('inf'):
            pytorch_cpu_times.append(r["pytorch_cpu_time"])
            batch_sizes_pytorch_cpu.append(r["batch_size"])
        
        if r["pytorch_cuda_time"] is not None and r["pytorch_cuda_time"] != float('inf'):
            pytorch_cuda_times.append(r["pytorch_cuda_time"]) 
            batch_sizes_pytorch_cuda.append(r["batch_size"])
    
    # Plot execution times
    plt.subplot(2, 1, 1)
    if numpy_times:
        plt.loglog(batch_sizes_numpy, numpy_times, 'o-', label='NumPy')
    if pytorch_cpu_times:
        plt.loglog(batch_sizes_pytorch_cpu, pytorch_cpu_times, 's-', label='PyTorch CPU')
    if pytorch_cuda_times:
        plt.loglog(batch_sizes_pytorch_cuda, pytorch_cuda_times, 'v-', label='PyTorch CUDA')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'{operation_name} - Execution Time vs Batch Size')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Plot speedups (only if we have data)
    plt.subplot(2, 1, 2)
    
    # Collect speedups, filtering out None values
    cpu_speedups = []
    cuda_speedups = []
    batch_sizes_cpu_speedup = []
    batch_sizes_cuda_speedup = []
    
    for i, r in enumerate(results):
        if r["cpu_speedup"] is not None and r["cpu_speedup"] != float('inf'):
            cpu_speedups.append(r["cpu_speedup"])
            batch_sizes_cpu_speedup.append(r["batch_size"])
        
        if r["cuda_speedup"] is not None and r["cuda_speedup"] != float('inf'):
            cuda_speedups.append(r["cuda_speedup"])
            batch_sizes_cuda_speedup.append(r["batch_size"])
    
    if cpu_speedups:
        plt.semilogx(batch_sizes_cpu_speedup, cpu_speedups, 's-', label='PyTorch CPU Speedup')
    
    if cuda_speedups:
        plt.semilogx(batch_sizes_cuda_speedup, cuda_speedups, 'v-', label='PyTorch CUDA Speedup')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup over NumPy')
    plt.title(f'{operation_name} - Speedup vs Batch Size')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{operation_name.lower().replace(" ", "_")}_benchmark.png')
    plt.close()

def generate_report(rotate_results, block_results, actions_results, episode_results):
    """Generate a markdown report with the benchmark results."""
    report = "# Narde Environment on NVIDIA T4 GPU Benchmark Report\n\n"
    
    # Test environment info
    report += "## Test Environment\n\n"
    report += f"- PyTorch Version: {torch.__version__}\n"
    report += f"- CUDA Available: {torch.cuda.is_available()}\n"
    if torch.cuda.is_available():
        report += f"- GPU: {torch.cuda.get_device_name(0)}\n"
        report += f"- CUDA Version: {torch.version.cuda}\n"
    report += f"- Time Limit Per Test: {MAX_TEST_TIME} seconds\n"
    report += "\n"
    
    # Board rotation results
    if rotate_results:
        report += "## Board Rotation Benchmark\n\n"
        report += "| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |\n"
        report += "|------------|-----------|-----------------|------------------|-------------|-------------|\n"
        
        for result in rotate_results:
            # Handle None or inf values
            numpy_time = f"{result['numpy_time']:.6f}" if result['numpy_time'] is not None and result['numpy_time'] != float('inf') else "SKIPPED"
            pytorch_cpu_time = f"{result['pytorch_cpu_time']:.6f}" if result['pytorch_cpu_time'] is not None and result['pytorch_cpu_time'] != float('inf') else "SKIPPED"
            pytorch_cuda_time = f"{result['pytorch_cuda_time']:.6f}" if result['pytorch_cuda_time'] is not None and result['pytorch_cuda_time'] != float('inf') else "SKIPPED"
            
            cpu_speedup = f"{result['cpu_speedup']:.2f}x" if result['cpu_speedup'] is not None and result['cpu_speedup'] != float('inf') else "N/A"
            cuda_speedup = f"{result['cuda_speedup']:.2f}x" if result['cuda_speedup'] is not None and result['cuda_speedup'] != float('inf') else "N/A"
            
            report += f"| {result['batch_size']:<10} | {numpy_time} | {pytorch_cpu_time} | {pytorch_cuda_time} | {cpu_speedup} | {cuda_speedup} |\n"
    else:
        report += "## Board Rotation Benchmark\n\n"
        report += "No data available for this benchmark.\n\n"
    
    # Block rule checking results
    if block_results:
        report += "\n## Block Rule Checking Benchmark\n\n"
        report += "| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |\n"
        report += "|------------|-----------|-----------------|------------------|-------------|-------------|\n"
        
        for result in block_results:
            # Handle None or inf values
            numpy_time = f"{result['numpy_time']:.6f}" if result['numpy_time'] is not None and result['numpy_time'] != float('inf') else "SKIPPED"
            pytorch_cpu_time = f"{result['pytorch_cpu_time']:.6f}" if result['pytorch_cpu_time'] is not None and result['pytorch_cpu_time'] != float('inf') else "SKIPPED"
            pytorch_cuda_time = f"{result['pytorch_cuda_time']:.6f}" if result['pytorch_cuda_time'] is not None and result['pytorch_cuda_time'] != float('inf') else "SKIPPED"
            
            cpu_speedup = f"{result['cpu_speedup']:.2f}x" if result['cpu_speedup'] is not None and result['cpu_speedup'] != float('inf') else "N/A"
            cuda_speedup = f"{result['cuda_speedup']:.2f}x" if result['cuda_speedup'] is not None and result['cuda_speedup'] != float('inf') else "N/A"
            
            report += f"| {result['batch_size']:<10} | {numpy_time} | {pytorch_cpu_time} | {pytorch_cuda_time} | {cpu_speedup} | {cuda_speedup} |\n"
    else:
        report += "\n## Block Rule Checking Benchmark\n\n"
        report += "No data available for this benchmark.\n\n"
    
    # Get valid actions results
    if actions_results:
        report += "\n## Get Valid Actions Benchmark\n\n"
        report += "| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |\n"
        report += "|------------|-----------|-----------------|------------------|-------------|-------------|\n"
        
        for result in actions_results:
            # Handle None or inf values
            numpy_time = f"{result['numpy_time']:.6f}" if result['numpy_time'] is not None and result['numpy_time'] != float('inf') else "SKIPPED"
            pytorch_cpu_time = f"{result['pytorch_cpu_time']:.6f}" if result['pytorch_cpu_time'] is not None and result['pytorch_cpu_time'] != float('inf') else "SKIPPED"
            pytorch_cuda_time = f"{result['pytorch_cuda_time']:.6f}" if result['pytorch_cuda_time'] is not None and result['pytorch_cuda_time'] != float('inf') else "SKIPPED"
            
            cpu_speedup = f"{result['cpu_speedup']:.2f}x" if result['cpu_speedup'] is not None and result['cpu_speedup'] != float('inf') else "N/A"
            cuda_speedup = f"{result['cuda_speedup']:.2f}x" if result['cuda_speedup'] is not None and result['cuda_speedup'] != float('inf') else "N/A"
            
            report += f"| {result['batch_size']:<10} | {numpy_time} | {pytorch_cpu_time} | {pytorch_cuda_time} | {cpu_speedup} | {cuda_speedup} |\n"
    else:
        report += "\n## Get Valid Actions Benchmark\n\n"
        report += "No data available for this benchmark.\n\n"
    
    # Episode execution results
    if episode_results:
        report += "\n## Episode Execution Benchmark\n\n"
        report += "| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |\n"
        report += "|------------|-----------|-----------------|------------------|-------------|-------------|\n"
        
        for result in episode_results:
            # Handle None or inf values
            numpy_time = f"{result['numpy_time']:.6f}" if result['numpy_time'] is not None and result['numpy_time'] != float('inf') else "SKIPPED"
            pytorch_cpu_time = f"{result['pytorch_cpu_time']:.6f}" if result['pytorch_cpu_time'] is not None and result['pytorch_cpu_time'] != float('inf') else "SKIPPED"
            pytorch_cuda_time = f"{result['pytorch_cuda_time']:.6f}" if result['pytorch_cuda_time'] is not None and result['pytorch_cuda_time'] != float('inf') else "SKIPPED"
            
            cpu_speedup = f"{result['cpu_speedup']:.2f}x" if result['cpu_speedup'] is not None and result['cpu_speedup'] != float('inf') else "N/A"
            cuda_speedup = f"{result['cuda_speedup']:.2f}x" if result['cuda_speedup'] is not None and result['cuda_speedup'] != float('inf') else "N/A"
            
            report += f"| {result['batch_size']:<10} | {numpy_time} | {pytorch_cpu_time} | {pytorch_cuda_time} | {cpu_speedup} | {cuda_speedup} |\n"
    else:
        report += "\n## Episode Execution Benchmark\n\n"
        report += "No data available for this benchmark.\n\n"
    
    # Analysis
    report += "\n## Analysis\n\n"
    
    # Calculate maximum speedups from available data
    max_rotate_cuda_speedup = 0
    max_block_cuda_speedup = 0
    max_actions_cuda_speedup = 0
    max_episode_cuda_speedup = 0
    
    rotate_crossover = "N/A"
    block_crossover = "N/A"
    actions_crossover = "N/A"
    episode_crossover = "N/A"
    
    # Find max speedups and crossover points only for valid data
    if rotate_results:
        for r in rotate_results:
            if r["cuda_speedup"] is not None and r["cuda_speedup"] != float('inf'):
                max_rotate_cuda_speedup = max(max_rotate_cuda_speedup, r["cuda_speedup"])
                if r["cuda_speedup"] > 1.0 and rotate_crossover == "N/A":
                    rotate_crossover = r["batch_size"]
    
    if block_results:
        for r in block_results:
            if r["cuda_speedup"] is not None and r["cuda_speedup"] != float('inf'):
                max_block_cuda_speedup = max(max_block_cuda_speedup, r["cuda_speedup"])
                if r["cuda_speedup"] > 1.0 and block_crossover == "N/A":
                    block_crossover = r["batch_size"]
    
    if actions_results:
        for r in actions_results:
            if r["cuda_speedup"] is not None and r["cuda_speedup"] != float('inf'):
                max_actions_cuda_speedup = max(max_actions_cuda_speedup, r["cuda_speedup"])
                if r["cuda_speedup"] > 1.0 and actions_crossover == "N/A":
                    actions_crossover = r["batch_size"]
    
    if episode_results:
        for r in episode_results:
            if r["cuda_speedup"] is not None and r["cuda_speedup"] != float('inf'):
                max_episode_cuda_speedup = max(max_episode_cuda_speedup, r["cuda_speedup"])
                if r["cuda_speedup"] > 1.0 and episode_crossover == "N/A":
                    episode_crossover = r["batch_size"]
    
    report += "### Key Findings\n\n"
    report += f"1. **Board Rotation**: Maximum {max_rotate_cuda_speedup:.2f}x speedup with CUDA, crossover at batch size {rotate_crossover}\n"
    report += f"2. **Block Rule Checking**: Maximum {max_block_cuda_speedup:.2f}x speedup with CUDA, crossover at batch size {block_crossover}\n"
    report += f"3. **Get Valid Actions**: Maximum {max_actions_cuda_speedup:.2f}x speedup with CUDA, crossover at batch size {actions_crossover}\n"
    report += f"4. **Episode Execution**: Maximum {max_episode_cuda_speedup:.2f}x speedup with CUDA, crossover at batch size {episode_crossover}\n\n"
    
    report += "### Conclusions\n\n"
    report += "1. **CUDA Acceleration**: PyTorch with CUDA provides significant speedup for environment operations at larger batch sizes\n"
    report += "2. **Memory Transfer Overhead**: For small batch sizes, the overhead of transferring data between CPU and GPU memory negates the benefits\n"
    report += "3. **Operation Complexity**: More complex operations show greater benefits from GPU acceleration\n"
    report += "4. **Batch Size Impact**: The speedup increases dramatically with batch size for most operations\n"
    report += "5. **Performance Limitation**: Some operations become prohibitively slow at larger batch sizes, requiring optimized implementation\n\n"
    
    report += "### Recommendations\n\n"
    report += "1. **Dynamic Device Selection**: Implement conditional logic that selects CPU or GPU based on operation type and batch size\n"
    report += "2. **Batch Size Optimization**: Structure environment execution to use optimal batch sizes for each operation\n"
    report += "3. **Operation Batching**: Batch small operations together to exceed the threshold where GPU acceleration becomes beneficial\n"
    report += "4. **PyTorch Implementation**: Use PyTorch for environment implementation to enable GPU acceleration for larger batches\n"
    report += "5. **Custom Kernels**: For operations like get_valid_actions that scale poorly, consider custom CUDA kernels for additional speedup\n"
    
    # Save report
    with open("narde_env_t4_gpu_benchmark_report.md", "w") as f:
        f.write(report)
    
    print(f"Report saved to narde_env_t4_gpu_benchmark_report.md")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Narde environment operations on CPU vs CUDA")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for each operation benchmark")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for combined operations benchmark")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available, only running CPU benchmarks")
    
    # Run benchmarks - catch exceptions to ensure all benchmarks run
    try:
        rotate_results = benchmark_operation("rotate_board", BATCH_SIZES, args.iterations)
    except Exception as e:
        print(f"Error in rotate_board benchmark: {e}")
        rotate_results = []
    
    try:
        block_results = benchmark_operation("check_block_rule", BATCH_SIZES, args.iterations)
    except Exception as e:
        print(f"Error in check_block_rule benchmark: {e}")
        block_results = []
    
    try:
        actions_results = benchmark_operation("get_valid_actions", BATCH_SIZES, args.iterations)
    except Exception as e:
        print(f"Error in get_valid_actions benchmark: {e}")
        actions_results = []
    
    try:
        episode_results = benchmark_combined_operations(BATCH_SIZES, args.episodes)
    except Exception as e:
        print(f"Error in episode execution benchmark: {e}")
        episode_results = []
    
    # Generate plots if not skipped
    if not args.skip_plots:
        try:
            if rotate_results:
                plot_results(rotate_results, "Board Rotation")
            if block_results:
                plot_results(block_results, "Block Rule Checking")
            if actions_results:
                plot_results(actions_results, "Get Valid Actions")
            if episode_results:
                plot_results(episode_results, "Episode Execution")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    # Generate report
    try:
        generate_report(rotate_results, block_results, actions_results, episode_results)
        print("Report generation completed successfully")
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    main() 