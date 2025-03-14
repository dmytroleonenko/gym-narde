# Narde Game AI Training

This directory contains scripts for training and evaluating a Deep Q-Network (DQN) agent to play the Narde board game.

## Files Overview

- `train_deepq_pytorch.py`: Trains a DQN agent using PyTorch
- `play_against_ai.py`: Play against the trained AI
- `evaluate_model.py`: Evaluate the trained model against a random agent

## Installation

Make sure you have the required dependencies:

```bash
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install gym
```

## Training the AI

Train the AI model with default parameters:

```bash
python train_deepq_pytorch.py
```

You can customize training parameters:

```bash
python train_deepq_pytorch.py --episodes 5000 --epsilon 1.0 --epsilon-decay 0.995 --lr 0.0001
```

Parameters:
- `--episodes`: Number of episodes to train (default: 10000)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-decay`: Epsilon decay rate (default: 0.995)
- `--lr`: Learning rate (default: 0.0001)

Models are saved in the `saved_models` directory.

## Playing Against the AI

Play against the trained AI:

```bash
python play_against_ai.py
```

To use a specific model:

```bash
python play_against_ai.py --model narde_model_5000.pt
```

## Evaluating the AI

Evaluate the trained AI against a random agent:

```bash
python evaluate_model.py
```

Options:
- `--model`: Name of the model to evaluate (default: narde_model_final.pt)
- `--games`: Number of games to play (default: 100)
- `--render`: Render the first game (optional)

Example:

```bash
python evaluate_model.py --model narde_model_5000.pt --games 200 --render
```

## Training Approach

The training approach uses a Deep Q-Network with:
- Allow any moves, but penalize invalid ones
- Reward based on game progress (bearing off checkers)
- Experience replay buffer for stability
- Target network for stable learning

The agent learns the rules of the game through experience by receiving penalties for invalid moves.

## Tips for Improvement

1. Run longer training (20,000+ episodes)
2. Adjust the exploration rate decay for better exploration
3. Experiment with the network architecture (add more layers or neurons)
4. Adjust rewards/penalties to encourage valid play more strongly