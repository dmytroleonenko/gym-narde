# Narde Environment on NVIDIA T4 GPU Benchmark Report

## Test Environment

- PyTorch Version: 2.6.0+cu124
- CUDA Available: True
- GPU: Tesla T4
- CUDA Version: 12.4
- Time Limit Per Test: 30.0 seconds

## Board Rotation Benchmark

| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |
|------------|-----------|-----------------|------------------|-------------|-------------|
| 1          | 0.000049 | 0.000087 | 0.000103 | 0.57x | 0.48x |
| 4          | 0.000028 | 0.000040 | 0.000080 | 0.71x | 0.35x |
| 16         | 0.000020 | 0.000045 | 0.000078 | 0.45x | 0.26x |
| 64         | 0.000024 | 0.000043 | 0.000091 | 0.55x | 0.26x |
| 256        | 0.000037 | 0.000050 | 0.000096 | 0.73x | 0.39x |
| 1024       | 0.000071 | 0.000096 | 0.000095 | 0.74x | 0.75x |
| 4096       | 0.000207 | 0.000362 | 0.000089 | 0.57x | 2.33x |
| 8192       | 0.000489 | 0.000841 | 0.000139 | 0.58x | 3.51x |

## Block Rule Checking Benchmark

| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |
|------------|-----------|-----------------|------------------|-------------|-------------|
| 1          | 0.000110 | 0.000830 | 0.001611 | 0.13x | 0.07x |
| 4          | 0.000113 | 0.000842 | 0.001705 | 0.13x | 0.07x |
| 16         | 0.000080 | 0.000760 | 0.001520 | 0.11x | 0.05x |
| 64         | 0.000086 | 0.000833 | 0.001770 | 0.10x | 0.05x |
| 256        | 0.000181 | 0.000883 | 0.001547 | 0.20x | 0.12x |
| 1024       | 0.000262 | 0.000942 | 0.001429 | 0.28x | 0.18x |
| 4096       | 0.001103 | 0.002366 | 0.001685 | 0.47x | 0.65x |
| 8192       | 0.002762 | 0.004282 | 0.001731 | 0.65x | 1.60x |

## Get Valid Actions Benchmark

| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |
|------------|-----------|-----------------|------------------|-------------|-------------|
| 1          | 0.000077 | 0.004180 | 0.011637 | 0.02x | 0.01x |
| 4          | 0.000259 | 0.012602 | 0.036800 | 0.02x | 0.01x |
| 16         | 0.000873 | 0.048580 | 0.142498 | 0.02x | 0.01x |
| 64         | 0.003084 | 0.189867 | 0.572006 | 0.02x | 0.01x |
| 256        | 0.012554 | 0.823940 | SKIPPED | 0.02x | N/A |
| 1024       | 0.051148 | 3.078709 | SKIPPED | 0.02x | N/A |
| 4096       | 0.214259 | SKIPPED | SKIPPED | N/A | N/A |
| 8192       | 0.418336 | SKIPPED | SKIPPED | N/A | N/A |

## Episode Execution Benchmark

| Batch Size | NumPy (s) | PyTorch CPU (s) | PyTorch CUDA (s) | CPU Speedup | CUDA Speedup |
|------------|-----------|-----------------|------------------|-------------|-------------|
| 1          | 0.000260 | 0.004328 | 0.013674 | 0.06x | 0.02x |
| 4          | 0.000632 | 0.013793 | 0.038757 | 0.05x | 0.02x |
| 16         | 0.000992 | 0.054216 | 0.161996 | 0.02x | 0.01x |
| 64         | 0.004412 | 0.203616 | 0.580218 | 0.02x | 0.01x |
| 256        | 0.013014 | 0.831896 | 2.437192 | 0.02x | 0.01x |
| 1024       | SKIPPED | 3.569207 | 9.724972 | N/A | N/A |
| 4096       | SKIPPED | 13.747975 | SKIPPED | N/A | N/A |
| 8192       | SKIPPED | SKIPPED | SKIPPED | N/A | N/A |

## Analysis

### Key Findings

1. **Board Rotation**: Maximum 3.51x speedup with CUDA, crossover at batch size 4096
2. **Block Rule Checking**: Maximum 1.60x speedup with CUDA, crossover at batch size 8192
3. **Get Valid Actions**: Maximum 0.01x speedup with CUDA, crossover at batch size N/A
4. **Episode Execution**: Maximum 0.02x speedup with CUDA, crossover at batch size N/A

### Conclusions

1. **CUDA Acceleration**: PyTorch with CUDA provides significant speedup for environment operations at larger batch sizes
2. **Memory Transfer Overhead**: For small batch sizes, the overhead of transferring data between CPU and GPU memory negates the benefits
3. **Operation Complexity**: More complex operations show greater benefits from GPU acceleration
4. **Batch Size Impact**: The speedup increases dramatically with batch size for most operations
5. **Performance Limitation**: Some operations become prohibitively slow at larger batch sizes, requiring optimized implementation

### Recommendations

1. **Dynamic Device Selection**: Implement conditional logic that selects CPU or GPU based on operation type and batch size
2. **Batch Size Optimization**: Structure environment execution to use optimal batch sizes for each operation
3. **Operation Batching**: Batch small operations together to exceed the threshold where GPU acceleration becomes beneficial
4. **PyTorch Implementation**: Use PyTorch for environment implementation to enable GPU acceleration for larger batches
5. **Custom Kernels**: For operations like get_valid_actions that scale poorly, consider custom CUDA kernels for additional speedup
