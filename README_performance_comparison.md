# MuZero Performance Comparison

This directory contains scripts to compare the performance of the original and optimized MuZero implementations.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- psutil (`pip install psutil`)

## Scripts

### Main Comparison Script

The `compare_muzero_implementations.py` script runs a full comparison between the original and optimized implementations of MuZero. It measures:

- Self-play performance (time and memory usage)
- Training step performance (time and memory usage)

Run the full comparison (takes a while):

```bash
./compare_muzero_implementations.py
```

### Test Script

The `test_comparison.py` script runs a smaller comparison to verify that everything is working correctly:

```bash
./test_comparison.py
```

## Results

The results are saved in the `benchmark_results` directory:
- Performance graphs are saved as PNG files
- Summary metrics are saved in a text file

## Performance Metrics

The scripts measure the following performance metrics:

1. **Execution time**: How long it takes to run self-play and training steps
2. **Memory usage**: How much memory is consumed during execution
3. **Speedup**: The ratio of original implementation time to optimized implementation time
4. **Memory reduction**: The percentage reduction in memory usage

## Hardware Configuration

The comparison automatically detects and uses the best available hardware:
- CUDA if available
- MPS for Apple Silicon
- CPU as fallback 