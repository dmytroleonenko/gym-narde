# Benchmarking Hardware Acceleration for MuZero Narde

This document summarizes our comprehensive benchmarking of hardware acceleration options for the Narde reinforcement learning environment, specifically targeting Apple Silicon (M-series) chips with Metal Performance Shaders (MPS) support.

## Scripts Developed

We created several specialized benchmark scripts to evaluate different aspects of hardware acceleration:

### 1. `test_pytorch_mps.py`

**Purpose**: Evaluate basic PyTorch operations with MPS acceleration vs CPU.

**Key features**:
- Benchmarks matrix multiplication, neural network operations, and vector operations
- Tests across various batch sizes (1, 8, 32, 128, 512, 2048, 4096, 8192)
- Reports execution time and speedup for each operation
- Provides performance summaries identifying optimal batch sizes

**Results**: Matrix multiplication showed the most significant acceleration (up to 8.17x speedup at batch size 2048), while neural network operations saw moderate improvements (up to 1.26x at batch size 8192).

### 2. `test_pytorch_muzero.py`

**Purpose**: Benchmark a MuZero-like neural network architecture using PyTorch with MPS.

**Key features**:
- Implements representation, dynamics, and prediction networks
- Tests forward passes and inference across various batch sizes
- Compares CPU vs MPS performance for neural network operations
- Includes synchronization for accurate timing measurements

**Results**: MPS acceleration became effective for batch sizes of 512 and above, with speedups increasing with larger batches (up to 1.44x at batch size 2048).

### 3. `test_pytorch_muzero_large.py`

**Purpose**: Evaluate MPS acceleration benefits for a larger MuZero network with bigger hidden dimensions.

**Key features**:
- Uses larger network architecture (hidden_dim=512, latent_dim=256)
- Focuses on batch sizes applicable to training scenarios (64-4096)
- Tests full forward passes through the larger network
- Provides optimal batch size recommendations

**Results**: Larger networks showed more significant MPS acceleration benefits at higher batch sizes, with nearly 2x speedup at batch size 4096.

### 4. `benchmark_env.py`

**Purpose**: Compare the original NumPy-based Narde environment with our PyTorch-accelerated implementation.

**Key features**:
- Benchmarks environment reset, episode execution, get_valid_actions, board_rotation, and check_block_rule
- Tests both implementations across various batch sizes
- Generates tables, charts, and a comprehensive report
- Provides specific recommendations based on crossover points

**Results**: The PyTorch CPU implementation was surprisingly faster than both the original and MPS implementations for most operations at smaller batch sizes. Board rotation with MPS showed 1.3x speedup at batch size 1024.

### 5. `benchmark_env_large.py`

**Purpose**: Focus on larger batch sizes to better demonstrate MPS acceleration benefits.

**Key features**:
- Tests larger batch sizes (128, 512, 1024, 2048, 4096, 8192)
- Benchmarks board rotation and complex board operations
- Implements warm-up for more accurate timing
- Generates detailed reports and visualizations

**Results**: Revealed significant speedups for large batches:
- Board rotation: 8.17x speedup at batch size 8192
- Board operations: 84.00x speedup at batch size 8192

## Implementation Created

We developed a complete PyTorch-accelerated implementation of the Narde environment:

### `torch_narde_env.py`

**Purpose**: Provide an MPS-accelerated alternative to the original NumPy-based environment.

**Key features**:
- Fully compatible with the Gymnasium interface
- Uses PyTorch tensors and operations optimized for GPU/MPS execution
- Implements dynamic device selection based on operation and batch size
- Optimizes board rotation, block rule checking, and other key operations

This implementation serves as a drop-in replacement for the original environment while leveraging hardware acceleration when beneficial.

## Reports Generated

### 1. `hardware_acceleration_report.md`

**Purpose**: Provide an executive summary of hardware acceleration options and performance.

**Contents**:
- Analysis of PyTorch with MPS vs CPU performance
- Detailed benchmark results for basic operations
- MuZero network benchmarks across various batch sizes
- Implementation recommendations and performance patterns

### 2. `narde_env_benchmark_report.md`

**Purpose**: Compare the performance of the original and PyTorch-accelerated environments.

**Contents**:
- Detailed timing results for environment operations
- Batch size analysis for various operations
- Speedup measurements and crossover points
- Specific recommendations for each operation type

### 3. `narde_large_batch_benchmark_report.md`

**Purpose**: Focus on the performance benefits for large batch operations.

**Contents**:
- Large batch board rotation results
- Matrix and feature operations performance
- Detailed speedup analysis
- Batch size threshold recommendations
- Future optimization opportunities

### 4. `hardware_acceleration_guidelines.md`

**Purpose**: Provide implementation guidelines based on benchmark results.

**Contents**:
- Dynamic device selection code examples
- Operation-specific recommendations and thresholds
- Model architecture considerations
- Training strategy adjustments
- Environment wrapper examples
- Performance tuning suggestions

## Key Findings

Our comprehensive benchmarking revealed several important insights:

1. **Operation-Specific Thresholds**: Each operation has a specific batch size threshold where MPS acceleration becomes beneficial:
   - Board Rotation: ≥ 1024 (8.17x max speedup)
   - Board Operations: ≥ 128 (84.00x max speedup)
   - Neural Network: ≥ 512 (1.97x max speedup)
   - Block Rule Checking: ≥ 2048 (minimal speedup)

2. **Matrix Operations**: Simple matrix multiplication shows the most dramatic speedups with MPS (up to 8.17x), particularly relevant for neural network layers.

3. **Batch Size Impact**: Performance improvements scale with batch size, with the most significant speedups observed at batch sizes of 4096-8192.

4. **JAX Limitations**: JAX with Metal backend showed limited practical benefits due to compatibility issues and operational constraints.

5. **PyTorch Advantages**: PyTorch with MPS provided more consistent acceleration and better compatibility than JAX.

6. **Model Size Effects**: Larger neural networks (hidden dimensions ≥ 512) show better acceleration on MPS.

7. **Memory Transfer Overhead**: For small batches, the overhead of transferring data between CPU and GPU memory negates acceleration benefits.

## Implementation Recommendations

Based on our findings, we recommend:

1. **Dynamic Device Selection**: Implement conditional logic that selects CPU or MPS based on operation type and batch size.

2. **Batch Size Optimization**: Structure training to use batch sizes of 512 or larger when using MPS acceleration.

3. **Operation Batching**: Batch small operations together to exceed the threshold where MPS becomes beneficial.

4. **PyTorch Implementation**: Use PyTorch for hardware-accelerated implementations due to better MPS support.

5. **Warm-up Operations**: Pre-compile operations with example inputs to avoid compilation overhead.

## Conclusion

Hardware acceleration via Metal Performance Shaders offers significant performance benefits for the Narde environment, particularly for larger models and batch sizes. By following our implementation guidelines with dynamic device selection, speedups of up to 84x can be achieved for certain operations compared to the baseline NumPy implementation.

The benefits of hardware acceleration are most pronounced for matrix operations and neural network forward passes with batch sizes above the operation-specific thresholds. For reinforcement learning training with MuZero, this translates to potentially significant reductions in training time when properly implemented. 