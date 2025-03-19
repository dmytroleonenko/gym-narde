# PyTorch on NVIDIA T4 GPU Benchmark Report

## Test Environment

- PyTorch Version: 2.6.0+cu124
- CUDA Available: True
- GPU: Tesla T4
- CUDA Version: 12.4

## Matrix Operations Benchmark

| Batch Size | NumPy (seconds) | PyTorch CPU (seconds) | PyTorch CUDA (seconds) | CPU vs NumPy | CUDA vs NumPy |
|------------|-----------------|------------------------|--------------------------|--------------|---------------|
| 1          | 0.000081 | 0.000478 | 0.000130 | 0.17x | 0.63x |
| 4          | 0.000039 | 0.000072 | 0.000132 | 0.55x | 0.30x |
| 16         | 0.000080 | 0.000106 | 0.000133 | 0.75x | 0.60x |
| 64         | 0.000247 | 0.000224 | 0.000133 | 1.10x | 1.86x |
| 256        | 0.000831 | 0.000702 | 0.000143 | 1.18x | 5.82x |
| 1024       | 0.003212 | 0.002394 | 0.000301 | 1.34x | 10.67x |
| 4096       | 0.012294 | 0.012638 | 0.000961 | 0.97x | 12.80x |

## Neural Network Benchmark

| Batch Size | PyTorch CPU (seconds) | PyTorch CUDA (seconds) | CUDA vs CPU |
|------------|------------------------|--------------------------|------------|
| 1          | 0.000195 | 0.000624 | 0.31x |
| 4          | 0.001543 | 0.004286 | 0.36x |
| 16         | 0.000650 | 0.000625 | 1.04x |
| 64         | 0.000684 | 0.000450 | 1.52x |
| 256        | 0.001505 | 0.000427 | 3.52x |
| 1024       | 0.004571 | 0.000529 | 8.64x |
| 4096       | 0.017170 | 0.000646 | 26.57x |

## Comparison with JAX (XLA) Results

This benchmark evaluates PyTorch's performance on the NVIDIA T4 GPU compared to CPU, performing similar operations to those in the JAX benchmark. From these results, we can make the following observations:

1. For matrix operations, PyTorch with CUDA achieves a maximum speedup of 12.80x over NumPy on CPU
2. For neural network operations, PyTorch with CUDA achieves a maximum speedup of 26.57x over PyTorch on CPU
3. PyTorch with CUDA shows excellent scaling with batch size, especially for larger batches
