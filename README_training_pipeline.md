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