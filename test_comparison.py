#!/usr/bin/env python3
"""
Test script for MuZero implementation comparison.
Runs a simplified version of the comparison with fewer games and epochs.
"""

# Only import the training step comparison, avoiding the self_play comparison
# as it may have more complex dependencies
from compare_muzero_implementations import compare_training_step

if __name__ == "__main__":
    print("Running simplified training comparison test...")
    # Run with small batch size and fewer epochs
    metrics = compare_training_step(batch_size=4, num_epochs=2)
    
    print("\n--- Test Results ---")
    print(f"Original implementation time: {metrics['original_time']:.4f}s")
    print(f"Optimized implementation time: {metrics['optimized_time']:.4f}s")
    print(f"Speedup: {metrics['speedup']:.2f}x")
    print(f"Memory reduction: {metrics['memory_reduction']:.2f}%") 