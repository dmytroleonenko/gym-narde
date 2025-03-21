import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
from glob import glob
import pickle
import random
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_games_from_file(filepath):
    """
    Load games from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        list: List of game dictionaries
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Handle both single game and list of games
        if isinstance(data, dict):
            return [data]  # Return as a list of one game
        elif isinstance(data, list):
            # Make sure each item in the list is a dict
            return [game for game in data if isinstance(game, dict)]
        else:
            logging.warning(f"Unexpected data type in {filepath}: {type(data)}")
            return []
    except Exception as e:
        logging.error(f"Error loading games from {filepath}: {e}")
        return []

def is_dict_game_format(game_dict):
    """
    Check if the game data is in the expected dictionary format with a game_history key.
    Returns True if the format is valid, False otherwise.
    """
    try:
        # Check if it's a dictionary
        if not isinstance(game_dict, dict):
            return False
            
        # Check for game_history key
        if 'game_history' not in game_dict:
            return False
            
        game_history = game_dict['game_history']
        
        # Check if game_history is a dictionary
        if not isinstance(game_history, dict):
            return False
            
        # Check for required keys in game_history
        if not all(key in game_history for key in ['states', 'actions', 'rewards']):
            return False
            
        # Get the components
        states = game_history['states']
        actions = game_history['actions']
        rewards = game_history['rewards']
        
        # Check if actions and rewards are non-empty
        if len(actions) <= 0 or len(rewards) <= 0:
            return False
            
        # Check if states has a length attribute (is iterable)
        if not hasattr(states, '__len__'):
            return False
            
        # Get the lengths
        states_len = len(states)
        actions_len = len(actions)
        
        # Check if states length is valid compared to actions length
        # There should be at least one more state than actions (final state)
        if states_len <= actions_len:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Error in game format validation: {e}")
        return False

def convert_game_to_trajectories(game_dict):
    """
    Convert a game dictionary into a list of tuples (observation, value, reward, policy)
    for the replay buffer.
    
    Args:
        game_dict: Dictionary containing game history with states, actions, and rewards
        
    Returns:
        list: List of (observation, value, reward, policy) tuples
    """
    try:
        game_history = game_dict['game_history']
        states = game_history['states']
        actions = game_history['actions']
        rewards = game_history['rewards']
        winner = game_dict.get('winner', 0)  # Default to draw (0) if winner not specified
        
        trajectories = []
        
        logging.debug(f"States type: {type(states)}, first state type: {type(states[0]) if len(states) > 0 else 'Empty'}")
        
        # Process each state-action-reward triplet
        for idx in range(len(actions)):
            try:
                # Get state as numpy array, ensure it's a copy to avoid reference issues
                state_raw = states[idx]
                
                # Convert to numpy array if it's not already
                if isinstance(state_raw, np.ndarray):
                    state = state_raw.copy()
                else:
                    state = np.array(state_raw)
                
                # Ensure state is 1D
                state = state.flatten()
                
                # Check if state is the correct dimension
                if state.size != 28:  # Expected input dimension
                    logging.warning(f"State has incorrect dimension: {state.size}, expected 28")
                    continue
                
                action = actions[idx]
                reward = rewards[idx]
                
                # Create a one-hot encoded policy for this action
                policy = np.zeros(576)  # Action dimension
                policy[action] = 1.0
                
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state)
                
                # Add to trajectories
                trajectories.append((state_tensor, reward, reward, policy))
            except Exception as e:
                logging.error(f"Error processing state-action-reward at index {idx}: {e}")
                continue
        
        # Add the final state with a dummy action and the game outcome as value
        try:
            # Get final state, ensure it's a copy
            final_state_raw = states[-1]
            
            # Convert to numpy array if it's not already
            if isinstance(final_state_raw, np.ndarray):
                final_state = final_state_raw.copy()
            else:
                final_state = np.array(final_state_raw)
            
            # Ensure it's 1D
            final_state = final_state.flatten()
            
            # Check if state is the correct dimension
            if final_state.size != 28:  # Expected input dimension
                logging.warning(f"Final state has incorrect dimension: {final_state.size}, expected 28")
                return trajectories
            
            final_state_tensor = torch.FloatTensor(final_state)
            
            # Final reward based on game outcome
            final_reward = 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)
            
            # Dummy policy for final state (uniform distribution)
            final_policy = np.zeros(576)
            final_policy.fill(1.0/576)  # Uniform policy
            
            # Add final state trajectory
            trajectories.append((final_state_tensor, final_reward, final_reward, final_policy))
        except Exception as e:
            logging.error(f"Error processing final state: {e}")
        
        return trajectories
    except Exception as e:
        logging.error(f"Error converting game to trajectories: {e}")
        return []

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain MuZero model with existing games")
    parser.add_argument("--games_dir", type=str, default="muzero_training/games",
                        help="Directory containing game data files (.pkl)")
    parser.add_argument("--model_path", type=str, default="muzero_training/models/muzero_model.pt",
                        help="Path to save the trained model")
    parser.add_argument("--input_dim", type=int, default=64,
                        help="Input dimension for the network")
    parser.add_argument("--action_dim", type=int, default=4162, 
                        help="Action dimension for the network")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for the network")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Add the current directory to Python path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Import MuZero specific modules
    from muzero.models import MuZeroNetwork
    from muzero.replay import ReplayBuffer
    
    # Initialize model
    logger.info(f"Initializing MuZero network with hidden_dim={args.hidden_dim}")
    network = MuZeroNetwork(
        input_dim=args.input_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    # Load games
    game_files = glob(os.path.join(args.games_dir, "*.pkl"))
    if not game_files:
        logger.error(f"No game files found in {args.games_dir}")
        return
    
    logger.info(f"Found {len(game_files)} game files")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=1000000)
    
    # Load games into replay buffer
    total_games = 0
    valid_games = 0
    for game_file in game_files:
        logger.info(f"Loading games from {game_file}")
        try:
            game_list = load_games_from_file(game_file)
            logger.info(f"Loaded {len(game_list)} games from {game_file}")
            
            for game_dict in game_list:
                try:
                    # Validate game format
                    if not is_dict_game_format(game_dict):
                        logger.warning(f"Skipping game with invalid format")
                        continue
                    
                    # Convert game dictionary to trajectories format expected by replay buffer
                    trajectories = convert_game_to_trajectories(game_dict)
                    
                    # Save trajectories to replay buffer
                    replay_buffer.save_game(trajectories)
                    
                    valid_games += 1
                except Exception as e:
                    logger.error(f"Error processing game: {str(e)}")
            
            total_games += len(game_list)
        except Exception as e:
            logger.error(f"Error loading games from {game_file}: {str(e)}")
    
    logger.info(f"Successfully processed {valid_games} out of {total_games} total games")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Training loop
    logger.info(f"Starting training with lr={args.lr}, batch_size={args.batch_size}")
    
    for epoch in range(args.num_epochs):
        epoch_value_loss = 0
        epoch_policy_loss = 0
        epoch_reward_loss = 0
        
        # Calculate the number of batches to process per epoch
        # This prevents overfitting when buffer size is small
        max_batches_per_epoch = 100
        
        # Calculate effective buffer size
        buffer_size = len(replay_buffer.buffer) + len(replay_buffer.terminal_buffer)
        
        if buffer_size == 0:
            logging.warning("Buffer size is empty, can't train.")
            break
            
        # Set batch size to be the minimum of the specified batch size and the buffer size
        effective_batch_size = min(args.batch_size, buffer_size)
        if effective_batch_size < args.batch_size:
            logging.warning(f"Using batch size of {effective_batch_size} instead of {args.batch_size} due to small buffer")
        
        if effective_batch_size < 2:
            logging.warning("Not enough samples for training. Need at least 2 samples.")
            break
            
        num_batches = min(max_batches_per_epoch, buffer_size // effective_batch_size)
        
        logging.info(f"Epoch {epoch+1}/{args.num_epochs} - Processing {num_batches} batches")
        
        for _ in range(num_batches):
            batch = replay_buffer.sample_batch(batch_size=effective_batch_size, device=args.device)
            
            if not batch or len(batch) < effective_batch_size // 2:
                logging.warning(f"Sampled fewer positions ({len(batch) if batch else 0}) than half the batch size. Skipping batch.")
                continue
                
            # Filter out empty items from the batch
            filtered_batch = []
            for item in batch:
                try:
                    if isinstance(item, tuple) and len(item) == 4 and isinstance(item[0], torch.Tensor) and item[0].numel() > 0:
                        filtered_batch.append(item)
                except:
                    # Skip any items that cause errors
                    pass
            
            if len(filtered_batch) < 2:
                logging.warning(f"After filtering, not enough valid samples in batch. Skipping.")
                continue
                
            # Format the batch data
            observations = torch.stack([item[0] for item in filtered_batch]).to(args.device)
            target_values = torch.tensor([item[1] for item in filtered_batch], device=args.device).view(-1, 1)
            target_rewards = torch.tensor([item[2] for item in filtered_batch], device=args.device).view(-1, 1)
            
            # For target policies, convert from one-hot to indices for CrossEntropyLoss
            target_policies_one_hot = torch.stack([torch.FloatTensor(item[3]) for item in filtered_batch]).to(args.device)
            target_policies = torch.argmax(target_policies_one_hot, dim=1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            value_pred, policy_pred, reward_pred = network(observations)
            
            # Calculate losses
            value_loss = torch.nn.MSELoss()(value_pred, target_values)
            policy_loss = torch.nn.CrossEntropyLoss()(policy_pred, target_policies)
            reward_loss = torch.nn.MSELoss()(reward_pred, target_rewards)
            
            # Total loss
            loss = value_loss + policy_loss + reward_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_reward_loss += reward_loss.item()
        
        # Print epoch losses
        if num_batches > 0:
            logging.info(f"Epoch {epoch+1} - Value Loss: {epoch_value_loss/num_batches:.6f}, Policy Loss: {epoch_policy_loss/num_batches:.6f}, Reward Loss: {epoch_reward_loss/num_batches:.6f}")
    
    # Save the model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(network.state_dict(), args.model_path)
    logger.info(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main() 