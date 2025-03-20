#!/usr/bin/env python3
"""
Benchmark Game Generation: CPU (NumPy) vs MPS (Metal Performance Shaders)

This script benchmarks the performance of game generation using:
1. Pure NumPy on CPU
2. PyTorch on CPU
3. PyTorch with MPS acceleration (Apple Silicon)

The benchmark measures games per second across different simulation counts
and worker configurations.
"""

import os
import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Benchmark")

# Import MuZero modules
from muzero.models import MuZeroNetwork
from muzero.training_optimized import optimized_self_play
from muzero.parallel_self_play import generate_games_parallel

def run_cpu_numpy_benchmark(num_games: int, num_simulations: int, num_workers: int) -> Tuple[float, float]:
    """
    Run benchmark using the standard parallel implementation with NumPy on CPU.
    
    Args:
        num_games: Number of games to generate
        num_simulations: Number of MCTS simulations per move
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (seconds taken, games per second)
    """
    logger.info(f"Running CPU (NumPy) benchmark: {num_games} games, {num_simulations} simulations, {num_workers} workers")
    
    # Create network on CPU
    network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
    
    # Create temporary directory for games
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Run benchmark
    start_time = time.time()
    
    generate_games_parallel(
        network=network,
        num_games=num_games,
        num_simulations=num_simulations,
        temperature=1.0,
        save_dir=temp_dir,
        num_workers=num_workers,
        temperature_drop=None
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    games_per_second = num_games / elapsed
    
    logger.info(f"CPU (NumPy) completed in {elapsed:.2f}s ({games_per_second:.2f} games/s)")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return elapsed, games_per_second

def run_cpu_pytorch_benchmark(num_games: int, num_simulations: int, num_workers: int) -> Tuple[float, float]:
    """
    Run benchmark using PyTorch on CPU.
    
    Args:
        num_games: Number of games to generate
        num_simulations: Number of MCTS simulations per move
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (seconds taken, games per second)
    """
    logger.info(f"Running CPU (PyTorch) benchmark: {num_games} games, {num_simulations} simulations, {num_workers} workers")
    
    # Create network on CPU with PyTorch
    network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
    network = network.to("cpu")
    
    # Create temporary directory for games
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Run benchmark
    start_time = time.time()
    
    # Use optimized_self_play but with CPU device
    batch_game_histories = optimized_self_play(
        network=network,
        num_games=num_games,
        num_simulations=num_simulations,
        mcts_batch_size=num_games,  # Use one big batch for fair comparison
        env_batch_size=min(num_games, 64),  # Limit env batch size to avoid OOM
        temperature=1.0,
        temperature_drop=None,
        device="cpu"
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    games_per_second = num_games / elapsed
    
    logger.info(f"CPU (PyTorch) completed in {elapsed:.2f}s ({games_per_second:.2f} games/s)")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return elapsed, games_per_second

def run_mps_benchmark(num_games: int, num_simulations: int, num_workers: int) -> Tuple[float, float]:
    """
    Run benchmark using PyTorch with MPS acceleration.
    
    Args:
        num_games: Number of games to generate
        num_simulations: Number of MCTS simulations per move
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (seconds taken, games per second)
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS not available, skipping MPS benchmark")
        return float('nan'), float('nan')
    
    logger.info(f"Running MPS benchmark: {num_games} games, {num_simulations} simulations, {num_workers} workers")
    
    # Create network on MPS
    network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
    network = network.to("mps")
    
    # Create temporary directory for games
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Run benchmark
    start_time = time.time()
    
    batch_game_histories = optimized_self_play(
        network=network,
        num_games=num_games,
        num_simulations=num_simulations,
        mcts_batch_size=32,  # Use MPS-optimal batch size based on benchmarks
        env_batch_size=min(num_games, 64),  # Limit env batch size to avoid OOM
        temperature=1.0,
        temperature_drop=None,
        device="mps"
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    games_per_second = num_games / elapsed
    
    logger.info(f"MPS completed in {elapsed:.2f}s ({games_per_second:.2f} games/s)")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return elapsed, games_per_second

def run_parallel_mps_benchmark(num_games: int, num_simulations: int, num_workers: int) -> Tuple[float, float]:
    """
    Run benchmark using PyTorch with MPS acceleration in parallel worker mode.
    
    Args:
        num_games: Number of games to generate
        num_simulations: Number of MCTS simulations per move
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (seconds taken, games per second)
    """
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS not available, skipping MPS benchmark")
        return float('nan'), float('nan')
    
    logger.info(f"Running Parallel MPS benchmark: {num_games} games, {num_simulations} simulations, {num_workers} workers")
    
    # Create temporary directory for games
    import tempfile, concurrent.futures, multiprocessing
    temp_dir = tempfile.mkdtemp()
    
    # Import the worker function
    from muzero.worker_functions import optimized_self_play_worker
    
    # Create network on CPU for pickling
    network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
    network_weights = network.state_dict()
    
    # Calculate games per worker
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers
    
    # Run benchmark
    start_time = time.time()
    
    # Use spawn method for multiprocessing (required for MPS)
    ctx = multiprocessing.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        mp_context=ctx
    ) as executor:
        futures = []
        
        for worker_id in range(num_workers):
            worker_games = games_per_worker + (1 if worker_id < remainder else 0)
            if worker_games > 0:
                future = executor.submit(
                    optimized_self_play_worker,
                    worker_id,
                    network_weights,
                    worker_games,
                    temp_dir,
                    num_simulations,
                    1.0,  # temperature
                    None,  # temperature_drop
                    32,    # mcts_batch_size
                    min(worker_games, 64),  # env_batch_size
                    512,   # hidden_dim
                    "mps"  # device
                )
                futures.append(future)
        
        # Wait for all to complete
        completed_games = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                worker_paths = future.result()
                completed_games += len(worker_paths)
            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    games_per_second = num_games / elapsed
    
    logger.info(f"Parallel MPS completed in {elapsed:.2f}s ({games_per_second:.2f} games/s)")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return elapsed, games_per_second

def run_benchmarks(games_list: List[int], sim_list: List[int], worker_list: List[int]) -> Dict:
    """
    Run all benchmarks for various combinations of parameters.
    
    Args:
        games_list: List of game counts to benchmark
        sim_list: List of simulation counts to benchmark
        worker_list: List of worker counts to benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    # First benchmark: vary simulations with fixed games and workers
    num_games = 20  # Fixed number of games for this test
    num_workers = 4  # Fixed number of workers
    
    sim_results = {
        "simulation_counts": sim_list,
        "cpu_numpy_time": [],
        "cpu_numpy_gps": [],
        "cpu_pytorch_time": [],
        "cpu_pytorch_gps": [],
        "mps_time": [],
        "mps_gps": [],
        "parallel_mps_time": [],
        "parallel_mps_gps": []
    }
    
    for num_simulations in sim_list:
        # CPU NumPy benchmark
        cpu_numpy_time, cpu_numpy_gps = run_cpu_numpy_benchmark(num_games, num_simulations, num_workers)
        sim_results["cpu_numpy_time"].append(cpu_numpy_time)
        sim_results["cpu_numpy_gps"].append(cpu_numpy_gps)
        
        # CPU PyTorch benchmark
        cpu_pytorch_time, cpu_pytorch_gps = run_cpu_pytorch_benchmark(num_games, num_simulations, num_workers)
        sim_results["cpu_pytorch_time"].append(cpu_pytorch_time)
        sim_results["cpu_pytorch_gps"].append(cpu_pytorch_gps)
        
        # MPS benchmark
        mps_time, mps_gps = run_mps_benchmark(num_games, num_simulations, num_workers)
        sim_results["mps_time"].append(mps_time)
        sim_results["mps_gps"].append(mps_gps)
        
        # Parallel MPS benchmark
        parallel_mps_time, parallel_mps_gps = run_parallel_mps_benchmark(num_games, num_simulations, num_workers)
        sim_results["parallel_mps_time"].append(parallel_mps_time)
        sim_results["parallel_mps_gps"].append(parallel_mps_gps)
    
    results["simulation_benchmark"] = sim_results
    
    # Second benchmark: vary workers with fixed games and simulations
    num_games = 20  # Fixed number of games
    num_simulations = 50  # Fixed number of simulations
    
    worker_results = {
        "worker_counts": worker_list,
        "cpu_numpy_time": [],
        "cpu_numpy_gps": [],
        "cpu_pytorch_time": [],
        "cpu_pytorch_gps": [],
        "mps_time": [],
        "mps_gps": [],
        "parallel_mps_time": [],
        "parallel_mps_gps": []
    }
    
    for num_workers in worker_list:
        # Only run CPU NumPy and Parallel MPS for worker benchmark
        # (single CPU/MPS benchmarks don't use workers)
        
        # CPU NumPy benchmark
        cpu_numpy_time, cpu_numpy_gps = run_cpu_numpy_benchmark(num_games, num_simulations, num_workers)
        worker_results["cpu_numpy_time"].append(cpu_numpy_time)
        worker_results["cpu_numpy_gps"].append(cpu_numpy_gps)
        
        # Skip single-threaded benchmarks for worker comparison
        worker_results["cpu_pytorch_time"].append(float('nan'))
        worker_results["cpu_pytorch_gps"].append(float('nan'))
        worker_results["mps_time"].append(float('nan'))
        worker_results["mps_gps"].append(float('nan'))
        
        # Parallel MPS benchmark
        parallel_mps_time, parallel_mps_gps = run_parallel_mps_benchmark(num_games, num_simulations, num_workers)
        worker_results["parallel_mps_time"].append(parallel_mps_time)
        worker_results["parallel_mps_gps"].append(parallel_mps_gps)
    
    results["worker_benchmark"] = worker_results
    
    # Third benchmark: vary game count with fixed simulations and workers
    num_simulations = 50  # Fixed number of simulations
    num_workers = 4  # Fixed number of workers
    
    game_results = {
        "game_counts": games_list,
        "cpu_numpy_time": [],
        "cpu_numpy_gps": [],
        "cpu_pytorch_time": [],
        "cpu_pytorch_gps": [],
        "mps_time": [],
        "mps_gps": [],
        "parallel_mps_time": [],
        "parallel_mps_gps": []
    }
    
    for num_games in games_list:
        # CPU NumPy benchmark
        cpu_numpy_time, cpu_numpy_gps = run_cpu_numpy_benchmark(num_games, num_simulations, num_workers)
        game_results["cpu_numpy_time"].append(cpu_numpy_time)
        game_results["cpu_numpy_gps"].append(cpu_numpy_gps)
        
        # CPU PyTorch benchmark
        cpu_pytorch_time, cpu_pytorch_gps = run_cpu_pytorch_benchmark(num_games, num_simulations, num_workers)
        game_results["cpu_pytorch_time"].append(cpu_pytorch_time)
        game_results["cpu_pytorch_gps"].append(cpu_pytorch_gps)
        
        # MPS benchmark
        mps_time, mps_gps = run_mps_benchmark(num_games, num_simulations, num_workers)
        game_results["mps_time"].append(mps_time)
        game_results["mps_gps"].append(mps_gps)
        
        # Parallel MPS benchmark
        parallel_mps_time, parallel_mps_gps = run_parallel_mps_benchmark(num_games, num_simulations, num_workers)
        game_results["parallel_mps_time"].append(parallel_mps_time)
        game_results["parallel_mps_gps"].append(parallel_mps_gps)
    
    results["game_benchmark"] = game_results
    
    return results

def generate_report(results: Dict, output_file: str = "game_generation_benchmark_report.md"):
    """
    Generate a detailed Markdown report from benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Path to save the report
    """
    with open(output_file, "w") as f:
        f.write("# Game Generation Benchmark: CPU vs MPS\n\n")
        f.write("This report compares the performance of game generation using different hardware acceleration options:\n\n")
        f.write("1. **CPU (NumPy)**: Standard implementation using NumPy on CPU with multiprocessing\n")
        f.write("2. **CPU (PyTorch)**: PyTorch implementation on CPU\n")
        f.write("3. **MPS**: PyTorch with Metal Performance Shaders on Apple Silicon\n")
        f.write("4. **Parallel MPS**: Multiple worker processes each using MPS\n\n")
        
        # Plot for simulation benchmark
        sim_data = results["simulation_benchmark"]
        sim_counts = sim_data["simulation_counts"]
        
        f.write("## Benchmark 1: Varying Simulation Count\n\n")
        f.write(f"- Fixed parameters: {len(sim_data['cpu_numpy_time'])} games, 4 workers\n\n")
        
        # Create table
        headers = ["Simulations", "CPU (NumPy) Time", "CPU (NumPy) Games/s", 
                  "CPU (PyTorch) Time", "CPU (PyTorch) Games/s",
                  "MPS Time", "MPS Games/s",
                  "Parallel MPS Time", "Parallel MPS Games/s"]
        
        rows = []
        for i, sim_count in enumerate(sim_counts):
            row = [
                sim_count,
                f"{sim_data['cpu_numpy_time'][i]:.2f}s",
                f"{sim_data['cpu_numpy_gps'][i]:.2f}",
                f"{sim_data['cpu_pytorch_time'][i]:.2f}s",
                f"{sim_data['cpu_pytorch_gps'][i]:.2f}",
                f"{sim_data['mps_time'][i]:.2f}s" if not np.isnan(sim_data['mps_time'][i]) else "N/A",
                f"{sim_data['mps_gps'][i]:.2f}" if not np.isnan(sim_data['mps_gps'][i]) else "N/A",
                f"{sim_data['parallel_mps_time'][i]:.2f}s" if not np.isnan(sim_data['parallel_mps_time'][i]) else "N/A",
                f"{sim_data['parallel_mps_gps'][i]:.2f}" if not np.isnan(sim_data['parallel_mps_gps'][i]) else "N/A"
            ]
            rows.append(row)
        
        f.write(tabulate(rows, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Worker benchmark
        worker_data = results["worker_benchmark"]
        worker_counts = worker_data["worker_counts"]
        
        f.write("## Benchmark 2: Varying Worker Count\n\n")
        f.write(f"- Fixed parameters: 20 games, 50 simulations\n\n")
        
        # Create table
        headers = ["Workers", "CPU (NumPy) Time", "CPU (NumPy) Games/s", 
                  "Parallel MPS Time", "Parallel MPS Games/s"]
        
        rows = []
        for i, worker_count in enumerate(worker_counts):
            row = [
                worker_count,
                f"{worker_data['cpu_numpy_time'][i]:.2f}s",
                f"{worker_data['cpu_numpy_gps'][i]:.2f}",
                f"{worker_data['parallel_mps_time'][i]:.2f}s" if not np.isnan(worker_data['parallel_mps_time'][i]) else "N/A",
                f"{worker_data['parallel_mps_gps'][i]:.2f}" if not np.isnan(worker_data['parallel_mps_gps'][i]) else "N/A"
            ]
            rows.append(row)
        
        f.write(tabulate(rows, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Game count benchmark
        game_data = results["game_benchmark"]
        game_counts = game_data["game_counts"]
        
        f.write("## Benchmark 3: Varying Game Count\n\n")
        f.write(f"- Fixed parameters: 50 simulations, 4 workers\n\n")
        
        # Create table
        headers = ["Games", "CPU (NumPy) Time", "CPU (NumPy) Games/s", 
                  "CPU (PyTorch) Time", "CPU (PyTorch) Games/s",
                  "MPS Time", "MPS Games/s",
                  "Parallel MPS Time", "Parallel MPS Games/s"]
        
        rows = []
        for i, game_count in enumerate(game_counts):
            row = [
                game_count,
                f"{game_data['cpu_numpy_time'][i]:.2f}s",
                f"{game_data['cpu_numpy_gps'][i]:.2f}",
                f"{game_data['cpu_pytorch_time'][i]:.2f}s",
                f"{game_data['cpu_pytorch_gps'][i]:.2f}",
                f"{game_data['mps_time'][i]:.2f}s" if not np.isnan(game_data['mps_time'][i]) else "N/A",
                f"{game_data['mps_gps'][i]:.2f}" if not np.isnan(game_data['mps_gps'][i]) else "N/A",
                f"{game_data['parallel_mps_time'][i]:.2f}s" if not np.isnan(game_data['parallel_mps_time'][i]) else "N/A",
                f"{game_data['parallel_mps_gps'][i]:.2f}" if not np.isnan(game_data['parallel_mps_gps'][i]) else "N/A"
            ]
            rows.append(row)
        
        f.write(tabulate(rows, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
        
        # Results analysis
        f.write("## Analysis\n\n")
        
        # Find the fastest method for simulation benchmark
        sim_methods = ["CPU (NumPy)", "CPU (PyTorch)", "MPS", "Parallel MPS"]
        sim_speeds = [
            np.nanmean(sim_data["cpu_numpy_gps"]),
            np.nanmean(sim_data["cpu_pytorch_gps"]),
            np.nanmean(sim_data["mps_gps"]),
            np.nanmean(sim_data["parallel_mps_gps"])
        ]
        
        fastest_sim_method = sim_methods[np.nanargmax(sim_speeds)]
        fastest_sim_speed = np.nanmax(sim_speeds)
        
        f.write(f"### Summary\n\n")
        f.write(f"- Fastest method overall: **{fastest_sim_method}** with an average of {fastest_sim_speed:.2f} games/s\n")
        
        # Calculate speedup ratios
        try:
            cpu_numpy_baseline = np.nanmean(sim_data["cpu_numpy_gps"])
            cpu_pytorch_speedup = np.nanmean(sim_data["cpu_pytorch_gps"]) / cpu_numpy_baseline
            mps_speedup = np.nanmean(sim_data["mps_gps"]) / cpu_numpy_baseline
            parallel_mps_speedup = np.nanmean(sim_data["parallel_mps_gps"]) / cpu_numpy_baseline
            
            f.write(f"- Speedup compared to CPU (NumPy):\n")
            f.write(f"  - CPU (PyTorch): {cpu_pytorch_speedup:.2f}x\n")
            f.write(f"  - MPS: {mps_speedup:.2f}x\n")
            f.write(f"  - Parallel MPS: {parallel_mps_speedup:.2f}x\n\n")
        except:
            f.write("- Could not calculate speedup ratios due to missing data\n\n")
        
        # Recommendations
        f.write("### Recommendations\n\n")
        
        # Determine best for different simulation counts
        f.write("#### By Simulation Count\n\n")
        for i, sim_count in enumerate(sim_counts):
            speeds = [
                sim_data["cpu_numpy_gps"][i] if not np.isnan(sim_data["cpu_numpy_gps"][i]) else 0,
                sim_data["cpu_pytorch_gps"][i] if not np.isnan(sim_data["cpu_pytorch_gps"][i]) else 0,
                sim_data["mps_gps"][i] if not np.isnan(sim_data["mps_gps"][i]) else 0,
                sim_data["parallel_mps_gps"][i] if not np.isnan(sim_data["parallel_mps_gps"][i]) else 0
            ]
            best_method = sim_methods[np.argmax(speeds)]
            f.write(f"- For {sim_count} simulations: **{best_method}** is fastest\n")
        
        f.write("\n")
        
        # Hardware specifications
        f.write("## Hardware Information\n\n")
        import platform
        import psutil
        
        f.write(f"- OS: {platform.system()} {platform.release()}\n")
        f.write(f"- Python: {platform.python_version()}\n")
        f.write(f"- CPU: {platform.processor()}\n")
        f.write(f"- CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n")
        f.write(f"- RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            f.write("- GPU: Apple Silicon with MPS support\n")
        else:
            f.write("- GPU: No MPS support detected\n")
        
        f.write(f"- PyTorch: {torch.__version__}\n")
        f.write(f"- NumPy: {np.__version__}\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("Based on the benchmark results, we can draw the following conclusions:\n\n")
        
        if fastest_sim_method == "CPU (NumPy)":
            f.write("1. The standard CPU implementation using NumPy provides the best overall performance for game generation.\n")
            f.write("2. Hardware acceleration with MPS does not significantly improve game generation speed in this specific workload.\n")
        elif fastest_sim_method == "CPU (PyTorch)":
            f.write("1. The PyTorch CPU implementation outperforms the standard NumPy implementation.\n")
            f.write("2. Hardware acceleration with MPS does not significantly improve game generation speed compared to optimized CPU code.\n")
        elif fastest_sim_method == "MPS":
            f.write("1. Hardware acceleration with MPS provides significant performance benefits for game generation.\n")
            f.write("2. The single-process MPS implementation is most efficient for this workload.\n")
        else:  # Parallel MPS
            f.write("1. Hardware acceleration with MPS in a multi-process configuration provides the best performance.\n")
            f.write("2. Utilizing both multiple CPU cores and MPS acceleration yields the highest throughput.\n")
        
        f.write("\nThe benchmark demonstrates that [CONCLUSION BASED ON ACTUAL RESULTS].\n")
    
    logger.info(f"Report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark game generation performance: CPU vs MPS")
    
    parser.add_argument("--games", nargs="+", type=int, default=[10, 20, 50],
                        help="List of game counts to benchmark")
    parser.add_argument("--simulations", nargs="+", type=int, default=[10, 50, 100],
                        help="List of simulation counts to benchmark")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="List of worker counts to benchmark")
    parser.add_argument("--output", type=str, default="game_generation_benchmark_report.md",
                        help="Output file for the report")
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(args.games, args.simulations, args.workers)
    
    # Generate report
    generate_report(results, args.output)

if __name__ == "__main__":
    main() 