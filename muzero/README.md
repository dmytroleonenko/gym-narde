# MuZero for Narde

This package implements the MuZero algorithm for playing the board game Narde.

## Features

- Complete MuZero implementation with Representation, Dynamics, and Prediction networks
- Monte Carlo Tree Search (MCTS) for planning
- Prioritized replay buffer for efficient training
- Supports TensorFlow-like API for training and evaluation
- MuZeroAgent class that's compatible with the existing benchmark framework

## Components

- `models.py`: Neural network models for MuZero (Representation, Dynamics, Prediction)
- `mcts.py`: Monte Carlo Tree Search implementation
- `replay.py`: Replay buffer for storing and sampling game trajectories
- `training.py`: Training loop and utilities
- `agent.py`: MuZeroAgent class that implements the GenericNardeAgent interface

## Usage

### Training from Scratch

To train a MuZero agent from scratch:

```bash
python train_and_evaluate_muzero.py --episodes 500 --simulations 50
```

This will train the agent for 500 episodes and then evaluate it against a RandomAgent.

### Evaluation Only

To evaluate a pretrained model:

```bash
python train_and_evaluate_muzero.py --eval-only --model-path muzero/models/muzero_model_final.pth
```

### Training Parameters

- `--episodes`: Number of episodes to train for (default: 200)
- `--buffer-size`: Size of the replay buffer (default: 10000)
- `--batch-size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--simulations`: Number of MCTS simulations per move (default: 50)
- `--temp-init`: Initial temperature for action selection (default: 1.0)
- `--temp-final`: Final temperature for action selection (default: 0.1)
- `--discount`: Discount factor for rewards (default: 0.997)
- `--save-interval`: Save the model every N episodes (default: 50)
- `--hidden-dim`: Dimensionality of the latent state (default: 256)
- `--device`: Device to run the training on ("auto", "cpu", "cuda", "mps")

### Evaluation Parameters

- `--eval-episodes`: Number of episodes for evaluation (default: 100)
- `--eval-max-steps`: Maximum steps per episode during evaluation (default: 500)
- `--model-path`: Path to MuZero model for evaluation

## Algorithm Details

MuZero learns a model of the environment using three main networks:

1. **Representation Network**: Encodes the current observation into a hidden state
2. **Dynamics Network**: Predicts the next hidden state and reward from the current hidden state and action
3. **Prediction Network**: Predicts the policy (action probabilities) and value from the hidden state

During training, MuZero uses Monte Carlo Tree Search (MCTS) to generate improved policies. The network is then trained to predict these improved policies, as well as the actual value returns and rewards.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Numpy
- Gymnasium 