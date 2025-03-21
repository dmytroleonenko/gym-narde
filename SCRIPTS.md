# Scripts Documentation

This file documents the purpose and usage of key scripts in the project.

## Game Generation Scripts

### final_complete_game_generation.py
- **Purpose**: Generates complete Narde games with proper bearing off of all checkers
- **Usage**: For production training data generation and integration with the parallel training pipeline
- **Features**:
  - Fixed action encoding for bearing off moves
  - Caching of valid moves for performance (~9x speedup)
  - Proper handling of no-move situations
  - Complete game generation with ~245 moves per game
  - Detailed logging and statistics
  - XLA optimization support for TPU/GPU acceleration

### final_optimized_game_generation.py
- **Purpose**: Optimized version of game generation focused on performance
- **Usage**: For high-throughput game generation when complete games aren't required
- **Features**:
  - Optimized for speed with partial caching
  - Simplified logging for better performance
  - Doesn't guarantee complete bearing off of all checkers

### analyze_game_generation_bottlenecks.py
- **Purpose**: Analyzes performance bottlenecks in game generation
- **Usage**: For debugging and optimization work
- **Note**: Used for development and performance profiling

## Training Scripts

### muzero/parallel_training.py
- **Purpose**: Complete training pipeline integrating parallel game generation and batched learning
- **Usage**: Main training entry point for MuZero on Narde
- **Features**:
  - Parallel game generation using multiple CPU cores
  - Efficient batched learning using GPU acceleration
  - Periodic model evaluation and checkpointing
  - Automatic resumption from checkpoints

### train_and_evaluate_muzero.py
- **Purpose**: Training and evaluation of MuZero models
- **Usage**: For training MuZero with various configurations
- **Features**:
  - Support for different hardware (CUDA, MPS, CPU)
  - Configurable hyperparameters
  - Evaluation capabilities

## MuZero Implementation Files

### muzero/mcts_batched_xla.py
- **Purpose**: XLA-optimized implementation of batched Monte Carlo Tree Search for MuZero
- **Usage**: For accelerated MCTS operations on TPUs and GPUs with XLA support
- **Features**:
  - Reduced CPU-GPU synchronization
  - Batched tensor operations
  - Pre-allocated tensors for better performance
  - Compatible with standard PyTorch but optimized for XLA

### muzero/xla_utils.py
- **Purpose**: Utilities for XLA-friendly PyTorch code
- **Usage**: Support library for XLA optimizations across the codebase
- **Features**:
  - XLA device management
  - Fallback mechanisms for non-XLA environments
  - Tensor conversion utilities
  - Step marking for compilation optimization

## Evaluation Scripts

### evaluate_with_agents.py
- **Purpose**: Evaluates trained models against different agent types
- **Usage**: For benchmarking model performance
- **Features**:
  - Support for different agent strategies
  - Statistical analysis of results
  - Multiple evaluation metrics

### evaluate_model.py
- **Purpose**: Evaluates model performance and tracks improvements across iterations
- **Usage**: For monitoring training progress
- **Features**:
  - Win rate calculation
  - Game length analysis
  - Performance comparison between iterations

## Benchmarking Scripts

### compare_muzero_implementations.py
- **Purpose**: Benchmarks original vs. optimized MuZero implementations
- **Usage**: For performance comparison
- **Features**:
  - Timing different implementations
  - Memory usage analysis
  - Performance metrics reporting

### profile_muzero.py
- **Purpose**: Detailed profiling of MuZero execution
- **Usage**: For identifying performance bottlenecks
- **Features**:
  - Function-level profiling
  - Time and memory profiling
  - Execution statistics

## Utility Scripts

### fix_haiku_deprecations.py
- **Purpose**: Patches Haiku library to handle deprecation warnings
- **Usage**: Run once to fix Haiku compatibility issues
- **Features**:
  - Updates imports to use newer JAX APIs
  - Fixes compatibility issues with newer JAX versions

## Testing Scripts

### run_tests.py
- **Purpose**: Runs all tests for the project
- **Usage**: For verifying correct functionality before/after changes
- **Features**:
  - Comprehensive test suite execution
  - Configurable test filtering
  - Detailed test reporting

### test_worker.py
- **Purpose**: Tests worker functionality for parallel execution
- **Usage**: For verifying parallel processing components
- **Features**:
  - Tests worker initialization
  - Tests task distribution
  - Tests result collection 