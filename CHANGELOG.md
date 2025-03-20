# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Created `final_complete_game_generation.py` that ensures games complete properly with bearing off all checkers
- Extended `README_training_pipeline.md` with detailed documentation on optimized self-play for Narde
- Integrated fixed bearing off action encoding into the vectorized environment and parallel training pipeline
- Improved `final_complete_game_generation.py` to use MuZero network for move selection:
  - Added proper environment integration for dice rolling and move execution
  - Implemented parallel game generation using multiple worker processes
  - Added timing and performance measurements for analysis
  - Added fallback to random selection when MuZero model loading fails
  - Added command-line option for random selection mode
- Added batched game generation using GPU-optimized batched MCTS in final_complete_game_generation.py
- Integrated batched game generation into the training pipeline with --use_batched_game_generation flag
- Added BatchedGameSimulator class for efficient self-play with GPU acceleration
- Implemented dynamic batch sizing to handle varying game completion rates 
- Improved MPS compatibility by ensuring proper tensor type conversion for Apple Silicon
- Added enhanced safety mechanisms to avoid infinite games:
  - Limited all games to a maximum of 500 moves (increased from 300)
  - Added detailed board state and dice roll logging when both players skip moves consecutively
  - Implemented tracking of consecutive skips to detect potential game stall conditions
- Added enhanced diagnostic logging for games without valid moves:
  - Detailed dice roll and board state analysis for each game 
  - Position-by-position attempt analysis for each die value
  - Bearing off validation checks with explanations of bearing conditions
  - Tracking of dice distribution patterns across games without valid moves

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
- Enhanced optimized self-play implementation to utilize multiple worker processes in parallel when using GPU acceleration, allowing better utilization of multi-core CPUs
- Improved multiprocessing support for both CUDA and MPS (Apple Silicon) in parallel training:
  - Added explicit spawn context for more robust processes with GPU access
  - Enhanced error handling with better logging and exception tracing
  - Added support for running optimized self-play with MPS acceleration on Apple Silicon
- Improved game generation for training with complete games:
  - Significantly increased average game length from <10 moves to ~245 moves
  - Enhanced the realism of training data by ensuring proper bearing off of all checkers
  - Optimized performance with caching of valid moves (~9x speedup)
  - Integrated fixes into the MuZero parallel training pipeline for high-quality training data
- Modified `final_complete_game_generation.py` to use MuZero for both agents when the --random flag is not provided, enabling full MuZero vs MuZero gameplay
- Updated all game simulation functions to use a consistent maximum move limit of 500 moves
- Enhanced debugging output by tracking and logging detailed board state information when consecutive skip situations occur
- Improved the debug output control in all scripts:
  - Modified both `narde_env.py` and `final_complete_game_generation.py` to respect the `--debug` flag
  - Updated the BatchedGameSimulator class to properly handle debug logging
  - Added consistent debug parameter propagation throughout the training pipeline
  - Made all debug print statements conditional on the debug flag being enabled
  - Set default log level to INFO to keep output clean when debug isn't needed
  - Changed verbose game action selection logs in BatchedGameSimulator from INFO to DEBUG level
- Adjusted logging levels in BatchedGameSimulator from INFO to DEBUG to reduce verbosity
- Added detailed logs for debugging dice values and valid moves when no moves are available
- Clarified initial dice roll handling in environment initialization

### Fixed
- Patched Haiku library to use jax.flatten_util instead of deprecated tree_util.tree_flatten.
- Resolved compatibility issues on Apple Silicon by working around environment limitations.
- Addressed JAX deprecation warnings by updating function calls to match current API.
- Increased test tolerance for numerical differences in JAX computation.
- Fixed model dimension mismatches to maintain internal consistency.
- Removed hardcoded input dimensions to allow flexible model configurations.
- Corrected handling of temperature drops during training evaluation.
- Fixed directory creation for saved models and trajectories.
- Resolved critical issues with game generation leading to invalid data:
  - Fixed action encoding issues affecting move translation.
  - Resolved turn-skipping bug causing premature game termination.
  - Corrected player perspective inconsistencies.
  - Fixed dice synchronization between environment and game objects.
  - Added proper tracking of remaining dice and valid moves.
  - Added better error handling for edge cases.
  - Fixed bearing off mechanics to properly end games.
- Improved logging and handling of stalled games.
- Fixed integration issues between batched game generation and the training pipeline.
- Corrected misleading game completion messages.
- Fixed premature game terminations.
- Enhanced dice handling and move validation.
- Modified game generation to rely solely on environment methods for valid move generation, removing manual generation throughout the codebase.
- Verified all valid move generation is now exclusively handled by environment methods (env.game.get_valid_moves()), eliminating any manual valid move generation via nested loops.
- Removed redundant diagnostics that attempted to second-guess the environment's valid move determination, fully trusting the environment's perspective on move validity.

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
- Game generation now produces complete and realistic Narde games with proper bearing off mechanics
- Training data quality is significantly improved with full game trajectories 
- Enhanced safety mechanisms prevent infinite games and provide detailed debugging information when unusual game states occur
- Batched game generation now works properly with the training pipeline, allowing for much faster game generation with GPU acceleration 