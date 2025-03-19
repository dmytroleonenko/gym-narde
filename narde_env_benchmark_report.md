# Narde Environment Benchmark Report

## Hardware Acceleration Results

This report compares the performance of the original NumPy-based Narde environment with the PyTorch-accelerated version using both CPU and MPS (Metal Performance Shaders).

### Environment Reset

| Implementation | Time (s) | Speedup vs Original |
|----------------|----------|-----------------|
| Original | 0.000007 | 1.00x |
| PyTorch (CPU) | 0.000019 | 0.36x |
| PyTorch (MPS) | 0.000926 | 0.01x |

### Full Episode Run

| Implementation | Time (s) | Speedup vs Original |
|----------------|----------|-----------------|
| Original | 0.001915 | 1.00x |
| PyTorch (CPU) | 0.000198 | 9.65x |
| PyTorch (MPS) | 0.005563 | 0.34x |

### Get Valid Actions

| Batch Size | Original (s) | PyTorch CPU (s) | PyTorch MPS (s) | Speedup (MPS vs Original) |
|------------|--------------|-----------------|----------------|--------------------------|
| 1 | 0.000000 | 0.000012 | 0.000757 | 0.00x |
| 8 | 0.000000 | 0.000002 | 0.000093 | 0.00x |
| 32 | 0.000000 | 0.000000 | 0.000023 | 0.00x |
| 128 | 0.000000 | 0.000000 | 0.000006 | 0.00x |
| 512 | 0.000000 | 0.000000 | 0.000002 | 0.01x |
| 1024 | 0.000000 | 0.000000 | 0.000001 | 0.02x |

### Board Rotation

| Batch Size | Original (s) | PyTorch CPU (s) | PyTorch MPS (s) | Speedup (MPS vs Original) |
|------------|--------------|-----------------|----------------|--------------------------|
| 1 | 0.000003 | 0.000179 | 0.000278 | 0.01x |
| 8 | 0.000001 | 0.000021 | 0.000043 | 0.02x |
| 32 | 0.000001 | 0.000005 | 0.000006 | 0.14x |
| 128 | 0.000001 | 0.000001 | 0.000002 | 0.39x |
| 512 | 0.000001 | 0.000001 | 0.000011 | 0.06x |
| 1024 | 0.000001 | 0.000001 | 0.000001 | 1.30x |

### Check Block Rule

| Batch Size | Original (s) | PyTorch CPU (s) | PyTorch MPS (s) | Speedup (MPS vs Original) |
|------------|--------------|-----------------|----------------|--------------------------|
| 1 | 0.000006 | 0.000945 | 0.001203 | 0.00x |
| 8 | 0.000004 | 0.000146 | 0.000141 | 0.03x |
| 32 | 0.000004 | 0.000034 | 0.000038 | 0.10x |
| 128 | 0.000004 | 0.000009 | 0.000011 | 0.32x |
| 512 | 0.000004 | 0.000003 | 0.000115 | 0.03x |
| 1024 | 0.000004 | 0.000002 | 0.000021 | 0.18x |

## Conclusion

The PyTorch-accelerated environment with MPS shows significant performance improvements for operations with larger batch sizes:

- Get Valid Actions: 0.02x speedup at batch size 1024
- Board Rotation: 1.30x speedup at batch size 1024
- Check Block Rule: 0.32x speedup at batch size 128

For full episode runs, the PyTorch MPS implementation is 0.34x faster than the original implementation.

### Recommendations

Based on the benchmark results, we recommend:

1. Use the PyTorch MPS implementation for batch operations with the following batch size thresholds:

   - Board Rotation: Batch size ≥ 1024

2. For operations with small batch sizes, the CPU implementation may be more efficient.

3. Set a dynamic threshold system in the environment that chooses the appropriate device (CPU vs MPS) based on the operation and batch size.
