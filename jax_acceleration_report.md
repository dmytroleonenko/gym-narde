# JAX Acceleration Performance Report

## Summary

This report summarizes the performance benchmarks of JAX acceleration compared to NumPy and PyTorch for various operations in the Narde game implementation. We tested different operations with varying batch sizes from very small (1) to extra large (16384) to understand when JAX provides acceleration benefits and when it might not be the optimal choice.

## Test Environment

- Hardware: Mac with Apple Silicon
- JAX Version: 0.5.2
- PyTorch: Using MPS acceleration (Metal Performance Shaders)
- Test operations:
  1. Board rotation (flip and negate board)
  2. Block rule checking (game rule validation)
  3. Neural network forward pass (similar to MuZero)

## Key Findings

### 1. Simple Operations (Board Rotation)

For simple operations like board rotation (flipping a board and negating values):

| Batch Size | JAX (FP32) | NumPy | Speedup |
|------------|------------|-------|---------|
| 1          | 0.000424s  | 0.000198s | 0.47x (slower) |
| 8          | 0.005185s  | 0.000227s | 0.04x (slower) |
| 32         | 0.004936s  | 0.000274s | 0.06x (slower) |
| 128        | 0.005173s  | 0.000419s | 0.08x (slower) |
| 512        | 0.005244s  | 0.001135s | 0.22x (slower) |
| 2048       | 0.007071s  | 0.003721s | 0.53x (slower) |
| 4096       | 0.007449s  | 0.007227s | 0.97x (similar) |
| 8192       | 0.008669s  | 0.014128s | 1.63x (faster) |
| 16384      | 0.011903s  | 0.028397s | 2.39x (faster) |

**Conclusion:** While JAX is slower than NumPy for small and medium batch sizes, we observe a crossover point at around batch size 4096, after which JAX becomes increasingly more efficient. At the largest batch size tested (16384), JAX is 2.4x faster than NumPy.

### 2. Complex Game Logic (Block Rule Checking)

For more complex game logic like checking the block rule (testing if all opponent pieces are in player's home):

| Batch Size | JAX | NumPy | Speedup |
|------------|-----|-------|---------|
| 1          | 0.014983s | 0.000437s | 0.03x (slower) |
| 8          | 0.023892s | 0.003139s | 0.13x (slower) |
| 32         | 0.022197s | 0.012239s | 0.55x (slower) |
| 128        | 0.022234s | 0.047967s | 2.16x (faster) |
| 512        | 0.025100s | 0.193185s | 7.70x (faster) |
| 2048       | 0.026970s | 0.774073s | 28.70x (faster) |
| 4096       | 0.028419s | 1.664312s | 58.56x (faster) |
| 8192       | 0.028015s | 3.206242s | 114.45x (faster) |
| 16384      | 0.032951s | 6.397304s | 194.14x (faster) |

**Conclusion:** JAX shows dramatic speedup for complex operations as batch size increases. The crossover point where JAX becomes faster is around batch size 128. For very large batches (16384), JAX is almost 200x faster than NumPy, demonstrating exceptional scaling for vectorized operations.

### 3. Neural Network Operations (Small Network)

Testing a small neural network (input_dim=24, hidden_dim=128):

| Batch Size | JAX FP32 | JAX BF16 | PyTorch (MPS) | BF16 vs FP32 | JAX FP32 vs PyTorch |
|------------|----------|----------|---------------|--------------|---------------------|
| 1          | 0.001018s | 0.004695s | 0.087721s | 0.22x (slower) | 86.21x (faster) |
| 8          | 0.016852s | 0.014060s | 0.086608s | 1.20x (faster) | 5.14x (faster)  |
| 32         | 0.013711s | 0.017508s | 0.088002s | 0.78x (slower) | 6.42x (faster)  |
| 128        | 0.022096s | 0.031861s | 0.091412s | 0.69x (slower) | 4.14x (faster)  |
| 512        | 0.032290s | 0.043718s | 0.108793s | 0.74x (slower) | 3.37x (faster)  |
| 4096       | 0.136468s | 0.179161s | 0.177391s | 0.76x (slower) | 1.30x (faster)  |
| 8192       | 0.262608s | 0.285562s | 0.256568s | 0.92x (slower) | 0.98x (similar) |

### 4. Neural Network Operations (Large Network)

Testing a larger neural network (input_dim=128, hidden_dim=512):

| Batch Size | JAX FP32 | JAX BF16 | PyTorch (MPS) | BF16 vs FP32 | JAX FP32 vs PyTorch |
|------------|----------|----------|---------------|--------------|---------------------|
| 1          | 0.006088s | 0.021004s | 0.086584s | 0.29x (slower) | 14.22x (faster) |
| 8          | 0.036042s | 0.057660s | 0.091909s | 0.63x (slower) | 2.55x (faster)  |
| 32         | 0.064223s | 0.096303s | 0.098357s | 0.67x (slower) | 1.53x (faster)  |
| 128        | 0.147556s | 0.176981s | 0.114616s | 0.83x (slower) | 0.78x (slower)  |
| 512        | 0.259506s | 0.312683s | 0.197676s | 0.83x (slower) | 0.76x (slower)  |
| 4096       | 1.290984s | 1.394152s | 0.639607s | 0.93x (slower) | 0.50x (slower)  |
| 8192       | 2.418536s | 2.686304s | 0.750632s | 0.90x (slower) | 0.31x (slower)  |

**Conclusions on Neural Networks:**
1. JAX FP32 is generally faster than JAX BF16 on CPU (contrary to expectations), though the gap narrows for larger batch sizes.
2. JAX significantly outperforms PyTorch at small batch sizes, especially for single example inference.
3. As batch size increases, PyTorch with MPS becomes increasingly more efficient than JAX running on CPU.
4. For the largest model and batch sizes, PyTorch with MPS is 3x faster than JAX on CPU.

## Precision Differences

When using BFloat16, we observed significant numerical differences compared to FP32:
- Mean absolute differences between FP32 and BF16 ranged from 0.4 to 0.8
- This level of difference could impact model convergence and quality
- Proper validation would be needed when using BF16 for the MuZero model

## Optimal Batch Size Thresholds

Based on our benchmarks, here are the recommended batch size thresholds for switching between NumPy and JAX:

| Operation | Recommended Threshold | Notes |
|-----------|------------------------|-------|
| Board Rotation | 4096 | JAX becomes faster than NumPy around batch size 4096 |
| Block Rule Checking | 128 | JAX provides significant speedup beyond batch size 128 |
| Neural Networks (CPU) | Any size for single inference, < 8192 for batches | JAX is better for small batches, PyTorch better for large batches |

## Recommendations

Based on the benchmark results, we recommend:

1. **For board rotation and simple operations:**
   - Use NumPy for batch sizes < 4096
   - Use JAX for batch sizes ≥ 4096
   - The overhead of JAX compilation isn't worth it for simple operations unless using very large batch sizes

2. **For complex game logic operations:**
   - Use JAX with JIT compilation for batch sizes > 128
   - The vectorized implementation shows dramatic speedups for large batches (up to 194x)
   - For small batches (< 32), stick with NumPy

3. **For neural network operations:**
   - JAX provides significant speedup over PyTorch for small to medium batch sizes when running on CPU
   - If hardware acceleration (GPU/MPS) is available, prefer PyTorch for large batch sizes (>512)
   - On Apple Silicon, JAX FP32 is generally faster than JAX BF16 for CPU operations

4. **For MuZero implementation:**
   - Use JAX for the neural network components for small batches and single-inference cases
   - Use JAX for batched game rule checking during MCTS (with batch sizes >128)
   - Continue using NumPy for individual board manipulations
   - Use PyTorch with MPS/GPU for large batch neural network operations (training)
   - BF16 precision may not be beneficial on CPU; test on GPU/TPU if available

## Future Work

Future benchmarking efforts should:
1. Test on GPU hardware where available
2. Evaluate the impact of BF16 precision on model training convergence
3. Measure the end-to-end performance of the full MuZero implementation
4. Consider using XLA compilation for PyTorch to compare with JAX

## Conclusion

JAX provides significant acceleration for complex operations with large batch sizes, making it well-suited for certain components of the MuZero implementation. However, it's important to selectively apply JAX to the right components of the system, as it can actually slow down simple operations at small batch sizes. The ideal approach is a hybrid system that dynamically selects the right framework (JAX, NumPy, or PyTorch) based on the operation type and batch size. 