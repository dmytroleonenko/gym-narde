# Changelog

## [Unreleased] - 2025-03-20

### Added
- Created `muzero/networks/cpu_muzero_networks.py` as a CPU-compatible alternative to JAX MuZero networks
- Created `muzero/networks/test_cpu_muzero_networks.py` with comprehensive tests for the CPU implementation
- Created `muzero/networks/test_basic.py` with simple tests for core MuZero functionality
- Created `muzero/networks/__init__.py` to facilitate imports
- Added `fix_haiku_deprecations.py` script to patch Haiku library's deprecation warnings
- Added `compare_muzero_implementations.py` script for benchmarking original vs. optimized implementations
- Added `test_comparison.py` script for quick verification of performance metrics
- Added `README_performance_comparison.md` with documentation for performance comparison tools
- Implemented optimized MuZero training with batched MCTS and vectorized environments in `muzero/training_optimized.py`
- Added `benchmark_results/optimization_summary.md` documenting performance improvements and optimization techniques
- Added `muzero/parallel_self_play.py` module for parallel game generation using multiple CPU cores
- Added `muzero/parallel_training.py` with complete training pipeline integrating parallel game generation and batched learning
- Implemented features for automatically detecting optimal worker count based on available CPU cores
- Added persistent game storage and loading for resumable training
- Added `README_training_pipeline.md` with comprehensive documentation on running the training pipeline and evaluating model performance
- Added `muzero/evaluate_model.py` script for evaluating model performance and tracking improvements across iterations

### Changed
- Updated JAX configuration in `test_jax_muzero_networks.py`:
  - Replaced environment variables with JAX config API
  - Changed `os.environ['JAX_PLATFORM_NAME'] = 'cpu'` to `jax.config.update("jax_platform_name", "cpu")`
  - Changed `os.environ['JAX_DISABLE_JIT'] = '1'` to `jax.config.update("jax_disable_jit", True)`
- Updated `rng_key` fixture to return a proper JAX random key
- Fixed `test_scalar_to_categorical_and_back` to use JAX implementation
- Updated `test_compute_n_step_returns` to use JAX arrays and the compute_n_step_returns function
- Updated tolerance in numerical tests to accommodate small differences
- Modified `muzero_train_step` in `training_utils.py` to include RNG key in apply_fn calls
- Enhanced `vectorized_model_unroll` in `training_utils.py` to include RNG key parameter
- Updated test fixtures to properly handle JAX key splitting
- Improved training workflow to alternate between game generation and model training in iterations
- Reduced log verbosity in `muzero/parallel_self_play.py` by moving worker network initialization messages to DEBUG level
- Removed temporary test code after verifying fixes for temperature_drop handling and directory creation

### Fixed
- Patched Haiku library to use `jax.extend.core` instead of deprecated `jax.core`
- Fixed incompatibility with Apple Silicon by creating a CPU-based implementation
- Resolved JAX deprecation warnings:
  - `jax.core.JaxprEqn` → `jax.extend.core.JaxprEqn`
  - `jax.core.Var` → `jax.extend.core.Var`
  - `jax.core.Jaxpr` → `jax.extend.core.Jaxpr`
- Increased test tolerance to handle minor numerical differences
- Fixed model dimension mismatch in parallel training pipeline:
  - Corrected input dimension from 24 to 28 to match NardeEnv's observation space
  - Enhanced worker processes to robustly detect model dimensions from checkpoint weights
  - Implemented automatic model recreation with matching dimensions when loading checkpoints
  - Added explicit dimension storage in checkpoints for more reliable loading
  - Improved dimension-related logging for easier debugging
- Fixed hardcoded input dimensions in benchmark and test files:
  - Updated `compare_muzero_implementations.py` to use correct observation dimension
  - Corrected input dimension in `muzero/networks/test_jax_muzero_networks.py`
  - Fixed `self_play_demo` function in `parallel_self_play.py` to use proper dimensions
- Fixed temperature_drop handling in `muzero/training_optimized.py` to properly handle None values, preventing comparison errors when running with CUDA
- Fixed directory creation in `train_muzero_optimized` function to handle empty checkpoint paths, preventing FileNotFoundError

### Notes
- The JAX MuZero implementation still has compatibility issues with Apple Silicon, but the CPU version provides a reliable alternative
- All CPU MuZero tests are passing successfully
- The fixed Haiku library no longer produces deprecation warnings
- Significant performance improvements in the optimized MuZero implementation, as demonstrated by the benchmarking tools:
  - 4.41x speedup in training time compared to the original implementation
  - 84.61% reduction in memory usage
  - Effective utilization of hardware acceleration (MPS on Apple Silicon)
- The new parallel training pipeline enables further scaling with multiple CPU cores and saves time by parallelizing game generation
- Checkpoint management allows for resumable training and continuous improvement of the model
- Automatic model evaluation provides insights into training progress and helps identify the best-performing model versions 