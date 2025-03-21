# MuZero Training Pipeline

This document explains how to use the MuZero parallel training pipeline for efficient model training and evaluation.

## Overview

The training pipeline implements a complete cycle:
1. Parallel game generation using multiple CPU cores
2. Efficient batched learning using GPU acceleration
3. Periodic model evaluation and checkpointing
4. Automatic resumption from checkpoints

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Multiprocessing support

### Basic Usage

To start training with default parameters:

```bash
python -m muzero.parallel_training
```

This will:
- Create a `muzero_training` directory with subdirectories for games, models, and logs
- Auto-detect available CPU cores and use them for parallel game generation
- Use the best available hardware (CUDA GPU, Apple Silicon MPS, or CPU)
- Run the training pipeline for 10 iterations

### Command Line Arguments

The training pipeline supports many customization options:

```bash
python -m muzero.parallel_training \
  --base_dir muzero_training \
  --batch_size 128 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --games_per_iteration 2000 \
  --num_simulations 50 \
  --num_epochs 10 \
  --num_workers 8 \
  --temperature 1.0 \
  --mcts_batch_size 16 \
  --training_iterations 20 \
  --save_checkpoint_every 1 \
  --hidden_dim 128 \
  --latent_dim 64
```

Additionally, new options have been added for batched GPU accelerated game generation:

  --use_batched_game_generation : If set, the training pipeline will use batched game generation via GPU accelerated MCTS using the BatchedGameSimulator. This enables faster self-play simulation by aggregating game states and running MCTS in batches.
  --mcts_batch_size             : Specifies the batch size for MCTS inference during batched game generation (default: 16). 

For example, to run the training pipeline with batched game generation:

```bash
python -m muzero.parallel_training \
  --base_dir muzero_training \
  --games_per_iteration 2000 \
  --use_batched_game_generation \
  --mcts_batch_size 32 \
  --num_simulations 50 \
  --num_workers 8
```

Key parameters:
- `--games_per_iteration`: Number of games to generate before each training phase
- `--num_workers`: CPU cores to use for game generation (auto-detected if not specified)
- `--training_iterations`: Total number of iteration cycles to run
- `--save_checkpoint_every`: Save model every N iterations

## Evaluating Performance

The training pipeline automatically tracks:
- Value, policy, and reward losses
- Training and game generation times
- Performance across iterations

### Automatic Improvement Tracking

You can use the provided evaluation script to track model improvement:

```bash
python -m muzero.evaluate_model --base_dir muzero_training --iteration latest --num_games 100
```

This will:
1. Load the latest model checkpoint (or a specific iteration if specified)
2. Play a specified number of games against a baseline algorithm
3. Compute metrics including win rate, average score, and game length
4. Save evaluation results to the logs directory

### Adding Automatic Evaluation to the Pipeline

To automatically evaluate the model after each training iteration, modify `parallel_training.py`:

```python
# Add after train_on_generated_games in run_training_pipeline method
evaluation_metrics = self.evaluate_model(iteration)
logger.info(f"  Evaluation win rate: {evaluation_metrics['win_rate']:.2f}%")
```

## Implementation Details

### Parallel Game Generation

The system uses `ProcessPoolExecutor` to generate games in parallel:
- Each worker runs on a separate CPU core
- Model weights are serialized and passed to workers
- Games are saved to disk with unique identifiers

### Training with Generated Games

After generating games:
- Games are loaded and added to a replay buffer
- The model is trained for a specified number of epochs
- Batched operations leverage GPU/MPS acceleration
- Performance metrics are tracked and logged

### Checkpointing and Resumption

The system automatically:
- Saves checkpoints at specified intervals
- Records training metrics in checkpoint files
- Detects and loads the latest checkpoint when restarting
- Allows resuming from a specific iteration

## Customizing the Pipeline

The `TrainingPipeline` class can be extended to add:
- Custom evaluation metrics
- Alternative replay buffer implementations
- Different model architectures
- Advanced scheduling of hyperparameters

## Example: Custom Evaluation

```python
def evaluate_model(self, iteration: int, num_eval_games: int = 100) -> Dict[str, float]:
    """Evaluate the current model by playing against a baseline."""
    logger.info(f"Evaluating model from iteration {iteration}")
    
    # Create evaluation directory
    eval_dir = os.path.join(self.logs_dir, f"eval_iteration_{iteration}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load model for evaluation (use CPU to avoid GPU memory issues)
    eval_network = self.network.to("cpu")
    
    # Initialize metrics
    win_count = 0
    draw_count = 0
    loss_count = 0
    game_lengths = []
    
    # Run evaluation games
    for game_id in range(num_eval_games):
        # Run game against baseline
        # (Implementation depends on your game environment)
        game_result, game_length = self._play_evaluation_game(eval_network, game_id)
        
        # Track results
        if game_result > 0:  # Win
            win_count += 1
        elif game_result == 0:  # Draw
            draw_count += 1
        else:  # Loss
            loss_count += 1
            
        game_lengths.append(game_length)
    
    # Calculate metrics
    win_rate = (win_count / num_eval_games) * 100
    draw_rate = (draw_count / num_eval_games) * 100
    avg_game_length = sum(game_lengths) / len(game_lengths)
    
    # Log results
    logger.info(f"Evaluation results (iteration {iteration}):")
    logger.info(f"  Win rate: {win_rate:.2f}%")
    logger.info(f"  Draw rate: {draw_rate:.2f}%")
    logger.info(f"  Average game length: {avg_game_length:.2f} moves")
    
    # Save results to file
    results = {
        "iteration": iteration,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": 100 - win_rate - draw_rate,
        "avg_game_length": avg_game_length,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    return results
```

## Performance Monitoring

The system logs comprehensive metrics:

- **Per iteration:**
  - Game generation time
  - Training time
  - Loss values (value, policy, reward)
  
- **Per epoch:**
  - Detailed loss information
  - Gradient norms
  - Learning rate

- **Hardware utilization:**
  - CPU core usage
  - Memory consumption
  - GPU/MPS utilization

All logs are saved to the `logs` directory within your base directory. 

## Configuring Logging Verbosity

The training pipeline has various logging levels to control output verbosity:

### Log Levels

- **ERROR**: Critical issues that prevent execution
- **WARNING**: Potential problems that don't stop execution
- **INFO**: General progress information (default)
- **DEBUG**: Detailed technical information for troubleshooting

### Changing Log Level

You can adjust the log level using Python's logging module:

```python
import logging
logging.getLogger('MuZero-Worker').setLevel(logging.WARNING)  # Suppress worker messages
logging.getLogger('MuZero-Parallel').setLevel(logging.INFO)   # Keep parallel execution info
```

For command-line usage, you can set the environment variable:

```bash
# Suppress all but warnings and errors
LOGLEVEL=WARNING python -m muzero.parallel_training

# Verbose output for debugging
LOGLEVEL=DEBUG python -m muzero.parallel_training
```

### Default Behaviors

- Worker network initialization messages are at DEBUG level to avoid console clutter
- Game generation progress is logged at INFO level every 10 games
- Hardware utilization warnings are at WARNING level
- Training metrics and performance statistics are at INFO level 

## Optimized Self-Play for Narde

The `--use_optimized_self_play` flag enables significant optimizations for Narde game generation, incorporating several critical findings:

### Complete Game Generation

Traditional implementations often terminated games prematurely due to environment stepping or turn mechanics. Our optimized version ensures:

- Games complete properly through bearing off all 15 checkers
- Average game length of ~245 moves (compared to <10 moves in non-optimized version)
- Proper tracking of bearing off progress for both players

### Key Optimizations

1. **Action Encoding Fix**: We corrected a critical issue with bearing off moves where:
   ```python
   # Incorrect (original)
   move_index = from_pos  # For bearing off moves
   
   # Correct (fixed)
   move_index = from_pos * 24  # Always multiply by 24 even for bearing off
   ```
   This ensures proper decoding in the environment's `_decode_action` method.

2. **Valid Moves Caching**: Computation of valid moves is cached based on board state and dice:
   ```python
   @lru_cache(maxsize=1024)
   def cached_get_valid_moves(board_tuple, dice_tuple):
       # Convert tuples back to arrays and compute valid moves
   ```
   This provides a ~9x speedup in game generation.

3. **No-Move Handling**: Properly implements Narde rules for when no valid moves exist - instead of terminating the game, it skips the current player's turn and passes to the next player, as specified in the official rules: "If no moves exist, skip; if only one is possible, use the higher die."

### Vectorized Environment Integration

The fixed action encoding has been integrated into the vectorized environment used by parallel training:

1. **Updated `get_valid_actions_batch` in `VectorizedNardeEnv`**:
   ```python
   # Bearing off moves are now correctly encoded
   if move[1] == 'off':
       # Multiply by 24 to ensure proper decoding back to from_pos
       action_idx = move[0] * 24
   ```

2. **Fixed `decode_actions` method**:
   ```python
   # Bear off move - correct calculation of from_pos
   # since we multiply by 24 in encoding
   moves.append((from_pos, 'off'))
   ```

3. **Improved Move Type Detection**:
   ```python
   # Get from_pos to determine move type
   from_pos = action_idx // 24
   
   # Check if this is a bearing off move
   unwrapped_env = env.envs[env.active_envs[i]].unwrapped
   for move in unwrapped_env.game.get_valid_moves():
       if move[0] == from_pos and move[1] == 'off':
           move_type = 1
           break
   ```

These improvements ensure that the parallel training pipeline correctly handles bearing off moves throughout the entire lifecycle of game generation, data collection, and model training.

### Integration with Parallel Training

When using the optimized self-play option with parallel workers:

```bash
python -m muzero.parallel_training --use_optimized_self_play --num_workers 8
```

Each worker generates complete games ensuring:
- Proper game termination through bearing off
- Realistic game length distribution
- Higher quality training data with complete gameplay trajectories

The `--env_batch_size` parameter controls how many environments are simulated in parallel within each worker, while `--mcts_batch_size` determines the batch size for MCTS simulations.

For optimal performance with 8 workers on modern GPUs (CUDA), we recommend:
```bash
python -m muzero.parallel_training --num_simulations 20 --batch_size 32768 --hidden_dim 512 --device cuda --use_optimized_self_play --mcts_batch_size 256 --env_batch_size 1024 --num_workers 8
```

This configuration balances game generation throughput with training efficiency, utilizing both CPU cores for parallel game generation and GPU acceleration for network inference.

### Performance Characteristics

- **Game Generation Speed**: ~20-25 complete games per second with 8 workers
- **Average Game Length**: ~245 moves per game
- **Memory Usage**: Scales with `env_batch_size` and `hidden_dim`
- **GPU Utilization**: Optimized through batching of inference requests

Compared to the non-optimized version, this represents a significant improvement in both game quality and generation speed, leading to more effective training. 

## Retraining Models with Existing Game Data

In addition to the full training pipeline, you can use the `retrain_model.py` script to retrain models using previously generated game data without generating new games.

### Basic Usage

```bash
python retrain_model.py --games_dir games --model_path muzero_training/models/muzero_retrained.pt --input_dim 28 --action_dim 576 --hidden_dim 128
```

This will:
- Load saved game data from the specified directory
- Convert the game histories into training trajectories
- Train a model with the given parameters
- Save the trained model to the specified path

### Command Line Arguments

```bash
python retrain_model.py \
  --games_dir games \
  --model_path muzero_training/models/muzero_retrained.pt \
  --input_dim 28 \
  --action_dim 576 \
  --hidden_dim 128 \
  --lr 0.0005 \
  --batch_size 32 \
  --num_epochs 5 \
  --device cuda  # or "cpu" or "mps"
```

Key parameters:
- `--games_dir`: Directory containing saved game files (.pkl format)
- `--model_path`: Path to save the retrained model
- `--input_dim`: Dimension of the observation space
- `--action_dim`: Dimension of the action space
- `--hidden_dim`: Hidden dimension for the MuZero network
- `--lr`: Learning rate for training
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--device`: Device to use for training (cuda, cpu, mps)

### Robust Game Data Handling

The retrain_model.py script includes several robustness features:
- Supports different game data formats (single games or lists of games)
- Handles various tensor shapes and dimensions
- Contains comprehensive error handling for corrupted game files
- Automatically skips invalid game data
- Provides detailed logging of the training process
- Supports games generated by the batched game simulator via `--use_batched_game_generation`

### Use Cases

Retraining is particularly useful for:
- Fine-tuning existing models with new game data
- Experimenting with different model architectures on the same game data
- Quickly iterating on hyperparameters without generating new games
- Recovering from interrupted training sessions 

### Troubleshooting

#### Empty Game Files
If you're seeing messages like "Loaded 0 games from [file]", check that:
- The game files in your `--games_dir` contain valid game data
- The game files were generated with the same environment version
- The pickle format is compatible with your Python version

You can verify game file contents with:
```bash
python -c "import pickle; print(len(pickle.load(open('path_to_game_file.pkl', 'rb'))))"
```

#### Model Dimension Mismatch
When evaluating a retrained model, you may encounter dimension mismatch errors:
```
Error loading model: size mismatch for representation_network.fc.6.weight: copying a param with shape torch.Size([128, 256]) from checkpoint, the shape in current model is torch.Size([256, 256]).
```

This occurs when:
- The model was trained with different `--hidden_dim` than what the evaluation script expects
- The default model architecture in evaluation scripts uses `hidden_dim=256`

To resolve this:
1. Specify the same dimensions when evaluating:
```bash
python evaluate_with_agents.py --model muzero_training/models/muzero_retrained.pt --hidden_dim 128 --episodes 10 --agent1 muzero_agents.MuZeroAgent --agent2 narde_agents.RandomAgent
```

2. Or retrain with the dimensions expected by evaluation scripts:
```bash
python retrain_model.py --games_dir games --model_path muzero_training/models/muzero_retrained.pt --input_dim 28 --action_dim 576 --hidden_dim 256 --lr 0.0005 --batch_size 32 --num_epochs 5 --device cuda
```

Always ensure that `--hidden_dim` is consistent between training and evaluation to avoid architecture mismatch errors. 