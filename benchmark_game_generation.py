#!/usr/bin/env python3
"""
Benchmark script to measure MuZero game generation performance.
Tests different configurations to find the optimal settings.
"""

import time
import torch
import argparse
import logging
from pathlib import Path
import os
import tempfile

from muzero.models import MuZeroNetwork
from muzero.parallel_self_play import generate_games_parallel, get_optimal_worker_count
from muzero.training_optimized import optimized_self_play
from muzero.mcts_batched import BatchedMCTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MuZero-Benchmark")

def benchmark_standard_generation(network, num_games=50, num_simulations=20, num_workers=None):
    """Benchmark standard parallel game generation using CPU."""
    logger.info(f"Benchmarking standard parallel game generation with {num_simulations} simulations")
    
    # Create temp directory for saving games
    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()
        
        # Generate games
        game_paths = generate_games_parallel(
            network=network,
            num_games=num_games,
            num_simulations=num_simulations,
            save_dir=temp_dir,
            num_workers=num_workers
        )
        
        duration = time.time() - start_time
        games_per_second = num_games / duration
        
        logger.info(f"Generated {len(game_paths)} games in {duration:.2f}s")
        logger.info(f"Performance: {games_per_second:.2f} games/s")
        
        return {
            "method": "standard_parallel",
            "num_games": num_games,
            "num_simulations": num_simulations,
            "num_workers": num_workers,
            "duration": duration,
            "games_per_second": games_per_second
        }

def benchmark_optimized_generation(network, num_games=50, num_simulations=20, 
                                  mcts_batch_size=8, env_batch_size=16):
    """Benchmark optimized game generation using GPU/MPS."""
    logger.info(f"Benchmarking optimized game generation with {num_simulations} simulations")
    
    # Check for GPU/MPS
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA for optimized game generation")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon) for optimized game generation")
    else:
        logger.warning("No GPU/MPS available, skipping optimized benchmark")
        return None
    
    # Move network to device
    network = network.to(device)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()
        
        # Generate games
        game_histories = optimized_self_play(
            network=network,
            num_games=num_games,
            num_simulations=num_simulations,
            mcts_batch_size=mcts_batch_size,
            env_batch_size=env_batch_size,
            temperature=1.0,
            temperature_drop=None,
            device=device
        )
        
        duration = time.time() - start_time
        games_per_second = num_games / duration
        
        logger.info(f"Generated {len(game_histories)} games in {duration:.2f}s")
        logger.info(f"Performance: {games_per_second:.2f} games/s")
        
        # Move network back to CPU
        network = network.to("cpu")
        
        return {
            "method": "optimized",
            "device": device,
            "num_games": num_games,
            "num_simulations": num_simulations,
            "mcts_batch_size": mcts_batch_size,
            "env_batch_size": env_batch_size,
            "duration": duration,
            "games_per_second": games_per_second
        }

def run_benchmarks():
    """Run all benchmarks with different configurations."""
    # Set up network
    input_dim = 28
    action_dim = 576
    hidden_dim = 128
    network = MuZeroNetwork(input_dim, action_dim, hidden_dim)
    
    # Get optimal worker count 
    default_workers = get_optimal_worker_count(None)
    
    # Store results
    results = []
    
    # Benchmark standard parallel generation with different simulation counts
    for num_simulations in [10, 20, 50]:
        result = benchmark_standard_generation(
            network=network,
            num_games=50,
            num_simulations=num_simulations,
            num_workers=default_workers
        )
        results.append(result)
    
    # Benchmark standard parallel generation with different worker counts
    for worker_ratio in [0.5, 0.75, 1.0]:
        workers = max(1, int(default_workers * worker_ratio))
        if workers != default_workers:  # Skip if we already tested this
            result = benchmark_standard_generation(
                network=network,
                num_games=50,
                num_simulations=20,
                num_workers=workers
            )
            results.append(result)
    
    # Benchmark optimized generation if available
    try:
        # Test different batch sizes
        for mcts_batch in [4, 8, 16]:
            for env_batch in [8, 16, 32]:
                result = benchmark_optimized_generation(
                    network=network,
                    num_games=50,
                    num_simulations=20,
                    mcts_batch_size=mcts_batch,
                    env_batch_size=env_batch
                )
                if result:
                    results.append(result)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not run optimized benchmarks: {str(e)}")
    
    # Print summary of results
    logger.info("\n===== BENCHMARK RESULTS =====")
    logger.info("Sorted by games per second (fastest first):")
    
    # Sort results by games per second
    results.sort(key=lambda x: x["games_per_second"], reverse=True)
    
    for i, result in enumerate(results):
        if result["method"] == "standard_parallel":
            logger.info(f"{i+1}. Standard Parallel: {result['games_per_second']:.2f} games/s " +
                        f"(simulations={result['num_simulations']}, workers={result['num_workers']})")
        else:
            logger.info(f"{i+1}. Optimized ({result['device']}): {result['games_per_second']:.2f} games/s " +
                        f"(simulations={result['num_simulations']}, mcts_batch={result['mcts_batch_size']}, " +
                        f"env_batch={result['env_batch_size']})")
    
    # Provide recommendations
    fastest = results[0]
    logger.info("\n===== RECOMMENDATIONS =====")
    
    if fastest["method"] == "standard_parallel":
        logger.info(f"Best configuration: Standard parallel with {fastest['num_workers']} workers " +
                    f"and {fastest['num_simulations']} simulations per move")
        
        cmd = (f"python -m muzero.parallel_training " +
               f"--num_workers {fastest['num_workers']} " +
               f"--num_simulations {fastest['num_simulations']} " +
               f"--batch_size 128 --hidden_dim 512")
    else:
        logger.info(f"Best configuration: Optimized self-play on {fastest['device']} with " +
                    f"mcts_batch_size={fastest['mcts_batch_size']} and env_batch_size={fastest['env_batch_size']}")
        
        cmd = (f"python -m muzero.parallel_training " +
               f"--use_optimized_self_play " +
               f"--num_simulations {fastest['num_simulations']} " +
               f"--mcts_batch_size {fastest['mcts_batch_size']} " +
               f"--env_batch_size {fastest['env_batch_size']} " +
               f"--batch_size 128 --hidden_dim 512 --device {fastest['device']}")
    
    logger.info(f"\nRecommended command:\n{cmd}")
    
    return results

if __name__ == "__main__":
    logger.info("Starting MuZero game generation benchmark")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available: True")
    
    run_benchmarks() 