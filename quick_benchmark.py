#!/usr/bin/env python3
"""
Quick benchmark script to measure MuZero game generation performance.
"""

import time
import torch
import logging
import tempfile
import os
from datetime import datetime

from muzero.models import MuZeroNetwork
from muzero.parallel_self_play import generate_games_parallel, get_optimal_worker_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MuZero-QuickBench")

def benchmark_simulation_counts(network, num_workers):
    """Benchmark different simulation counts."""
    results = []
    
    for num_simulations in [10, 20, 30]:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Testing with {num_simulations} simulations and {num_workers} workers")
            
            start_time = time.time()
            game_paths = generate_games_parallel(
                network=network,
                num_games=30,
                num_simulations=num_simulations,
                save_dir=temp_dir,
                num_workers=num_workers
            )
            
            duration = time.time() - start_time
            games_per_second = len(game_paths) / duration
            
            logger.info(f"Generated {len(game_paths)} games in {duration:.2f}s")
            logger.info(f"Performance: {games_per_second:.2f} games/s")
            
            results.append({
                "simulations": num_simulations,
                "workers": num_workers,
                "games_per_second": games_per_second
            })
    
    return results

if __name__ == "__main__":
    # Set up network
    logger.info("Starting quick benchmark")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    input_dim = 28
    action_dim = 576
    hidden_dim = 128
    network = MuZeroNetwork(input_dim, action_dim, hidden_dim)
    
    # Get optimal worker count 
    default_workers = get_optimal_worker_count(None)
    logger.info(f"Detected {default_workers} optimal workers")
    
    # Run benchmarks
    results = benchmark_simulation_counts(network, default_workers)
    
    # Print results
    logger.info("\n===== BENCHMARK RESULTS =====")
    for result in results:
        logger.info(f"Simulations: {result['simulations']}, Workers: {result['workers']}, Speed: {result['games_per_second']:.2f} games/s")
    
    # Find best configuration
    fastest = max(results, key=lambda x: x["games_per_second"])
    
    logger.info("\n===== RECOMMENDATION =====")
    logger.info(f"Best configuration: {fastest['simulations']} simulations with {fastest['workers']} workers")
    logger.info(f"Estimated speed: {fastest['games_per_second']:.2f} games/s")
    logger.info(f"For 2000 games, estimated time: {2000 / fastest['games_per_second'] / 60:.1f} minutes")
    
    cmd = (f"python -m muzero.parallel_training " +
           f"--num_workers {fastest['workers']} " +
           f"--num_simulations {fastest['simulations']} " +
           f"--batch_size 32768 --hidden_dim 512 " + 
           f"--device auto")
    
    logger.info(f"\nRecommended command:\n{cmd}") 