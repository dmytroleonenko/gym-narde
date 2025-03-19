# Benchmarking Hardware Acceleration for MuZero Narde

This document summarizes our comprehensive benchmarking of hardware acceleration options for the Narde reinforcement learning environment, targeting both Apple Silicon (M-series) chips with Metal Performance Shaders (MPS) support and NVIDIA GPUs with CUDA support.

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

### 6. `benchmark_jax_cuda.py`

**Purpose**: Evaluate JAX performance with XLA on NVIDIA T4 GPU vs CPU.

**Key features**:
- Tests matrix operations and neural network performance
- Compares JAX on GPU against NumPy on CPU
- Uses batched operations across various sizes
- Tests compilation and execution time separately

**Results**:
- Matrix operations: Up to 28.88x speedup at batch size 4096
- Neural networks: Inconsistent performance, with best speedup at batch size 1 (8.53x)
- Crossover point at batch size 256 for matrix operations

### 7. `pytorch_t4_benchmark.py`

**Purpose**: Benchmark PyTorch on NVIDIA T4 GPU vs CPU for basic operations.

**Key features**:
- Compares PyTorch with CUDA vs NumPy for matrix operations
- Evaluates neural network forward passes
- Measures memory transfer overhead
- Tests various batch sizes

**Results**:
- Matrix operations: Up to 12.80x speedup with CUDA at batch size 4096
- Neural network: Up to 26.57x speedup at batch size 4096
- CUDA acceleration becomes effective at batch size 64 for matrices and 16 for neural networks

### 8. `benchmark_env_t4_cuda.py`

**Purpose**: Compare environment operations on NVIDIA T4 GPU vs CPU.

**Key features**:
- Benchmarks board rotation, block rule checking, and action validation
- Implements automatic test skipping for time-intensive operations
- Tests both PyTorch CPU and CUDA implementations
- Includes reporting and visualization functions

**Results**:
- Board rotation: Up to 3.51x speedup with CUDA at batch size 8192
- Block rule checking: Up to 1.60x speedup at batch size 8192
- Get valid actions: CPU outperformed CUDA for all batch sizes
- Some operations became prohibitively slow at larger batch sizes

### 9. `benchmark_muzero_t4_cuda.py`

**Purpose**: Evaluate MuZero network performance on NVIDIA T4 GPU.

**Key features**:
- Tests standard (128 hidden dim) and large (512 hidden dim) networks
- Benchmarks forward passes and recurrent inference
- Simulates MCTS-like operations
- Compares model sizes and batch size effects

**Results**:
- Standard network: Up to 28.84x forward pass speedup, 32.09x recurrent speedup
- Large network: Up to 42.19x forward pass speedup, 42.61x recurrent speedup
- Larger model showed greater benefit from GPU acceleration
- GPU acceleration effective from batch size 8 for large models

## Implementations Created

We developed accelerated implementations of the Narde environment:

### 1. `torch_narde_env.py`

**Purpose**: Provide an MPS-accelerated alternative to the original NumPy-based environment.

**Key features**:
- Fully compatible with the Gymnasium interface
- Uses PyTorch tensors and operations optimized for GPU/MPS execution
- Implements dynamic device selection based on operation and batch size
- Optimizes board rotation, block rule checking, and other key operations

### 2. `cuda_narde_env.py`

**Purpose**: Provide a CUDA-accelerated implementation for NVIDIA GPUs.

**Key features**:
- Compatible with the Gymnasium interface
- Optimized for CUDA execution on NVIDIA GPUs
- Implements efficient batch processing
- Uses dynamic device selection based on operation characteristics

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

### 5. `narde_env_t4_gpu_benchmark_report.md`

**Purpose**: Document benchmarking results for the Narde environment on NVIDIA T4 GPU.

**Contents**:
- Performance comparison between NumPy, PyTorch CPU, and PyTorch CUDA
- Tables with execution times and speedup metrics for all operations
- Analysis of memory transfer overhead
- Recommendations for batch size thresholds and operation strategies

### 6. `muzero_t4_gpu_benchmark_report.md`

**Purpose**: Analyze MuZero network performance on NVIDIA T4 GPU.

**Contents**:
- Comparison between standard and large network architectures
- Forward pass and recurrent inference benchmarks
- MCTS simulation performance analysis
- Recommendations for model size and device selection

### 7. `jax_t4_gpu_xla_benchmark_report.md`

**Purpose**: Evaluate JAX with XLA performance on NVIDIA T4 GPU.

**Contents**:
- Matrix operations and neural network benchmarks
- Comparison between JAX GPU and NumPy CPU
- Analysis of compilation and execution time
- Batch size threshold recommendations

## Key Findings

Our comprehensive benchmarking revealed several important insights:

### Apple Silicon (M-series) with MPS

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

### NVIDIA T4 GPU with CUDA

1. **Operation-Specific Performance**:
   - Board Rotation: Effective acceleration (3.51x) at batch sizes ≥ 4096
   - Block Rule Checking: Modest acceleration (1.60x) at batch sizes ≥ 8192
   - Get Valid Actions: CPU outperformed GPU across all tested batch sizes
   - MuZero Networks: Excellent acceleration (up to 42.61x) for large networks

2. **Exceptional Neural Network Performance**: MuZero networks showed much better acceleration on the T4 GPU compared to Apple Silicon, with speedups up to 42x.

3. **Large Model Advantage**: Larger MuZero networks (512 hidden dim) showed significantly better GPU utilization than standard models (128 hidden dim).

4. **JAX vs PyTorch**: Both frameworks showed strong performance, with JAX showing higher maximum speedups for matrix operations but more inconsistent performance for neural networks.

5. **GPU Memory Limitations**: Operations with large batch sizes can be memory-limited on GPUs with less VRAM (16GB on T4).

6. **Kernel Launch Overhead**: Small batch operations are often slower on GPU due to kernel launch overhead.

7. **Skipping Prohibitively Slow Tests**: Some operations like `get_valid_actions` scale so poorly that time limits are needed to skip tests that would take too long.

## Implementation Recommendations

Based on our findings, we recommend:

### For Apple Silicon (M-series)

1. **Dynamic Device Selection**: Implement conditional logic that selects CPU or MPS based on operation type and batch size.

2. **Batch Size Optimization**: Structure training to use batch sizes of 512 or larger when using MPS acceleration.

3. **Operation Batching**: Batch small operations together to exceed the threshold where MPS becomes beneficial.

4. **PyTorch Implementation**: Use PyTorch for hardware-accelerated implementations due to better MPS support.

5. **Warm-up Operations**: Pre-compile operations with example inputs to avoid compilation overhead.

### For NVIDIA GPUs

1. **Model-Centric Acceleration**: Focus on accelerating neural network operations, which showed the most significant speedups.

2. **CPU Fallback for Environment**: Some environment operations are better on CPU, especially at smaller batch sizes.

3. **Custom CUDA Kernels**: For operations like `get_valid_actions`, consider specialized CUDA kernels for better performance.

4. **Large Batch Processing**: Structure training to use the largest practical batch sizes for maximum GPU utilization.

5. **Mixed Precision Training**: Use FP16/BF16 for further performance improvements on GPU.

## Guidelines for Future Hardware Benchmarking

When benchmarking new hardware accelerators (e.g., NVIDIA A100, Google TPUs), follow these structured guidelines:

### 1. Benchmark Setup and Preparation

**Hardware Configuration**:
- Document exact hardware specifications (model, memory, connectivity)
- Record software environment (driver versions, runtime libraries)
- Note any thermal or power constraints

**Baseline Establishment**:
- Run CPU-only benchmarks as a consistent baseline
- Use identical operations and batch sizes across all platforms
- Ensure timing methods are consistent and accurate

**Warm-up Considerations**:
- Include warm-up iterations to ensure JIT compilation is complete
- Separate compilation time from execution time when using XLA
- Account for frequency scaling and thermal throttling

### 2. Core Benchmarks to Run

**Matrix Operations Benchmark**:
- Test simple matrix multiplication with various sizes
- Measure memory bandwidth with large array operations
- Compare framework-specific optimizations (e.g., torch.matmul vs JAX lax.dot)

**Neural Network Benchmark**:
- Test both standard (128 hidden dim) and large (512 hidden dim) MuZero networks
- Measure forward passes, recurrent inference, and gradient computation
- Test mixed precision performance when available (FP16, BF16)

**Environment Operations Benchmark**:
- Benchmark core operations like board rotation and state validation
- Measure batch processing efficiency at various sizes
- Time complete episode execution with different batch sizes

**MCTS Simulation Benchmark**:
- Simulate complete MCTS search steps (forward pass + recurrent steps)
- Measure end-to-end training iteration performance
- Test different tree sizes and exploration parameters

### 3. Advanced Testing Techniques

**Memory Profiling**:
- Measure peak memory usage during operations
- Test memory throughput with varying batch sizes
- Profile memory allocation and deallocation patterns

**Multi-Device Scaling**:
- For multi-GPU or TPU pod setups, test scaling efficiency
- Measure communication overhead between devices
- Compare different parallelism strategies (data, model, pipeline)

**Compiler Optimization**:
- For XLA/TPU, experiment with different compilation options
- Test specialized graph optimizations
- Measure compilation cache effectiveness

**Custom Kernel Evaluation**:
- Implement and test specialized kernels for critical operations
- Compare vendor-specific libraries (cuDNN, CUTLASS, etc.)
- Benchmark different implementation strategies for complex operations

### 4. Analysis and Reporting

**Comprehensive Metrics**:
- Record execution time, throughput, and latency for all operations
- Calculate speedups relative to CPU baseline
- Determine crossover points where acceleration becomes beneficial

**Scaling Analysis**:
- Plot performance vs. batch size on log-log scales
- Identify bottlenecks and saturation points
- Determine optimal batch sizes for different operations

**Practical Recommendations**:
- Document optimal framework choices for the hardware
- Provide code examples for dynamic device selection
- Suggest architecture modifications to better utilize the hardware

**Report Generation**:
- Create standardized markdown reports with consistent formatting
- Include raw data tables for all benchmarks
- Generate visualization plots for key metrics

### 5. Platform-Specific Considerations

**For NVIDIA A100**:
- Test Multi-Instance GPU (MIG) configurations
- Evaluate TensorFloat-32 (TF32) precision benefits
- Benchmark NVLink performance for multi-GPU setups

**For Google TPUs**:
- Focus on JAX with XLA performance
- Benchmark performance on various TPU versions (v2, v3, v4)
- Test TPU pod slice scaling efficiency

**For AMD GPUs**:
- Benchmark ROCm framework support and performance
- Compare HIP backend with CUDA performance
- Test pipeline optimization for GCN/RDNA architectures

**For Custom/Specialized Hardware**:
- Develop adapter layers for framework compatibility
- Benchmark precision vs performance tradeoffs
- Evaluate energy efficiency and cost-performance ratio

## Conclusion

Hardware acceleration offers significant performance benefits for the Narde environment and MuZero training, with different accelerators showing various strengths and weaknesses. By following our testing methodology and implementation recommendations, you can achieve optimal performance across a variety of hardware platforms.

Our benchmarks have shown that:

1. Apple Silicon with MPS provides excellent matrix operation acceleration, particularly at large batch sizes.

2. NVIDIA GPUs with CUDA deliver exceptional neural network performance, especially for larger models.

3. Operation-specific thresholds determine when hardware acceleration becomes beneficial.

4. Framework choice matters significantly, with PyTorch generally providing more consistent acceleration.

For future hardware platforms, thorough benchmarking using our established methodology will help determine the optimal implementation strategy and identify the most cost-effective hardware solutions for MuZero training and deployment. 