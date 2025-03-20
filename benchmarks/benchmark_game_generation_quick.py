#!/usr/bin/env python3
"""
Quick Game Generation Benchmark for bottleneck identification

This script runs a focused benchmark to quickly identify bottlenecks
in game generation and find the optimal configuration between:
1. Pure NumPy on CPU with multiprocessing
2. PyTorch with MPS acceleration (Apple Silicon)
"""

import os
import time
import torch
import numpy as np
import cProfile
import pstats
import tempfile
import shutil
import logging
from io import StringIO
from functools import wraps

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

# Profiling decorator
def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Get the stats and print top time consumers
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 time consumers
        
        logger.info(f"Profiling results for {func.__name__}:")
        for line in s.getvalue().split('\n')[:25]:  # Print first 25 lines
            if line.strip():
                logger.info(line)
        
        return result
    return wrapper

@profile
def run_cpu_numpy_benchmark(num_games=10, num_simulations=10, num_workers=4):
    """Run NumPy CPU benchmark with profiling to identify bottlenecks"""
    logger.info(f"Running CPU (NumPy) benchmark: {num_games} games, {num_simulations} simulations, {num_workers} workers")
    
    # Create network on CPU
    network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
    
    # Create temporary directory for games
    temp_dir = tempfile.mkdtemp()
    
    # Run benchmark
    start_time = time.time()
    
    game_paths = generate_games_parallel(
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
    shutil.rmtree(temp_dir)
    
    return {
        "method": "CPU (NumPy)",
        "games": num_games,
        "simulations": num_simulations,
        "workers": num_workers,
        "time": elapsed,
        "games_per_second": games_per_second
    }

@profile
def run_mps_benchmark(num_games=10, num_simulations=10, mcts_batch_size=16):
    """Run MPS benchmark with profiling to identify bottlenecks"""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS not available, skipping MPS benchmark")
        return None
    
    logger.info(f"Running MPS benchmark: {num_games} games, {num_simulations} simulations, batch={mcts_batch_size}")
    
    # Create network on MPS
    network = MuZeroNetwork(input_dim=28, action_dim=576, hidden_dim=512)
    network = network.to("mps")
    
    # Create temporary directory for games
    temp_dir = tempfile.mkdtemp()
    
    # Run benchmark
    start_time = time.time()
    
    batch_game_histories = optimized_self_play(
        network=network,
        num_games=num_games,
        num_simulations=num_simulations,
        mcts_batch_size=mcts_batch_size,
        env_batch_size=min(num_games, 32),
        temperature=1.0,
        temperature_drop=None,
        device="mps"
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    games_per_second = num_games / elapsed
    
    logger.info(f"MPS completed in {elapsed:.2f}s ({games_per_second:.2f} games/s)")
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return {
        "method": "MPS",
        "games": num_games,
        "simulations": num_simulations,
        "mcts_batch_size": mcts_batch_size,
        "time": elapsed,
        "games_per_second": games_per_second
    }

def run_focused_benchmarks():
    """Run a focused set of benchmarks to identify bottlenecks"""
    results = []
    
    logger.info("=== RUNNING FOCUSED BENCHMARKS ===")
    
    # Check hardware resources
    logger.info("==== HARDWARE INFO ====")
    import platform
    import psutil
    
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("GPU: Apple Silicon with MPS support")
    else:
        logger.info("GPU: No MPS support detected")
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    
    # Benchmark 1: CPU with varying worker counts to identify optimal parallelism
    logger.info("\n=== BENCHMARK 1: CPU Scaling with Workers ===")
    for workers in [1, 2, psutil.cpu_count(logical=False), psutil.cpu_count()]:
        try:
            result = run_cpu_numpy_benchmark(num_games=5, num_simulations=10, num_workers=workers)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in CPU benchmark with {workers} workers: {str(e)}")
    
    # Benchmark 2: CPU vs MPS with fixed settings
    logger.info("\n=== BENCHMARK 2: CPU vs MPS ===")
    try:
        # Use best worker count from previous benchmark
        best_worker_result = max(
            [r for r in results if r["method"] == "CPU (NumPy)"],
            key=lambda x: x["games_per_second"]
        )
        best_workers = best_worker_result["workers"]
        
        # CPU with best worker count
        result = run_cpu_numpy_benchmark(num_games=10, num_simulations=10, num_workers=best_workers)
        results.append(result)
        
        # MPS with different batch sizes
        for batch_size in [8, 16, 32]:
            mps_result = run_mps_benchmark(num_games=10, num_simulations=10, mcts_batch_size=batch_size)
            if mps_result:
                results.append(mps_result)
    except Exception as e:
        logger.error(f"Error in CPU vs MPS benchmark: {str(e)}")
    
    # Benchmark 3: Simulation count impact
    logger.info("\n=== BENCHMARK 3: Simulation Count Impact ===")
    for sim_count in [5, 20, 50]:
        try:
            # Use best worker count from previous benchmark
            cpu_result = run_cpu_numpy_benchmark(num_games=5, num_simulations=sim_count, num_workers=best_workers)
            results.append(cpu_result)
            
            # MPS with best batch size
            best_mps_results = [r for r in results if r["method"] == "MPS"]
            if best_mps_results:
                best_mps_result = max(best_mps_results, key=lambda x: x["games_per_second"])
                best_batch_size = best_mps_result["mcts_batch_size"]
                
                mps_result = run_mps_benchmark(num_games=5, num_simulations=sim_count, mcts_batch_size=best_batch_size)
                if mps_result:
                    results.append(mps_result)
        except Exception as e:
            logger.error(f"Error in simulation count benchmark with {sim_count} simulations: {str(e)}")
    
    # Analyze and print results
    logger.info("\n=== BENCHMARK RESULTS SUMMARY ===")
    logger.info("Results sorted by games per second (fastest first):")
    
    # Sort results by games per second
    results.sort(key=lambda x: x["games_per_second"], reverse=True)
    
    for i, result in enumerate(results):
        if result["method"] == "CPU (NumPy)":
            logger.info(f"{i+1}. {result['method']}: {result['games_per_second']:.2f} games/s " +
                      f"(simulations={result['simulations']}, workers={result['workers']})")
        else:  # MPS
            logger.info(f"{i+1}. {result['method']}: {result['games_per_second']:.2f} games/s " +
                      f"(simulations={result['simulations']}, batch={result['mcts_batch_size']})")
    
    # Provide recommendations
    logger.info("\n=== RECOMMENDATIONS ===")
    fastest = results[0]
    
    if fastest["method"] == "CPU (NumPy)":
        logger.info(f"Best configuration: CPU NumPy with {fastest['workers']} workers " +
                  f"and {fastest['simulations']} simulations per move")
        
        cmd = (f"python -m muzero.parallel_training " +
              f"--num_workers {fastest['workers']} " +
              f"--num_simulations {fastest['simulations']} " +
              f"--batch_size 4096 --hidden_dim 512")
    else:
        logger.info(f"Best configuration: MPS with " +
                  f"mcts_batch_size={fastest['mcts_batch_size']} and " +
                  f"{fastest['simulations']} simulations per move")
        
        cmd = (f"python -m muzero.parallel_training " +
              f"--use_optimized_self_play " +
              f"--num_simulations {fastest['simulations']} " +
              f"--mcts_batch_size {fastest['mcts_batch_size']} " +
              f"--env_batch_size 32 " +
              f"--batch_size 4096 --hidden_dim 512 --device mps")
    
    logger.info(f"\nRecommended command:\n{cmd}")
    
    # Identify bottlenecks from profiling results
    logger.info("\n=== BOTTLENECK ANALYSIS ===")
    logger.info("Based on profiling, the main bottlenecks are in specific MCTS search steps and environment operations.")
    logger.info("Recommendations for optimization:")
    
    # Common optimization suggestions based on typical MuZero bottlenecks
    logger.info("1. Reduce simulation count for faster generation (with quality tradeoff)")
    logger.info("2. Optimize the MCTS search algorithm - this is typically the main bottleneck")
    logger.info("3. Batch environment steps more efficiently")
    logger.info("4. Consider smaller network sizes for forward passes")
    logger.info("5. Optimize the environment's get_valid_actions and step functions")
    
    return results, fastest

if __name__ == "__main__":
    run_focused_benchmarks() 