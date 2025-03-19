#!/usr/bin/env python3
"""
Script to evaluate DQN vs DQN for Narde game.
This script will pit the same DQN model against itself for a specified number of episodes
and collect statistics on the outcomes.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import torch
import random
import argparse
import time
from train_simpledqn import DQN  # Import the DQN model from training script

def evaluate_dqn_vs_dqn(model_path, episodes=100, max_steps=500, verbose=False):
    """
    Evaluate a DQN model playing against itself.
    """
    # Create environment
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space[0].n * env.action_space[1].n
    
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    # Tracking metrics
    white_wins = 0
    black_wins = 0
    draws = 0
    episode_steps = []
    episode_borne_off_white = []
    episode_borne_off_black = []
    
    print(f"Evaluating model against itself: {model_path}")
    print(f"Total episodes: {episodes}")
    print("-" * 60)
    
    # Enable verbose debug for first 5 episodes if verbose mode is on
    debug_episodes = 5 if verbose else 0
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        step_count = 0
        
        # Track the current player
        current_player_is_white = True  # Game always starts with White
        
        # Debug info
        debug_this_episode = episode <= debug_episodes
        if debug_this_episode:
            print(f"\nEpisode {episode} starting")
            print(f"Initial board state: {env_unwrapped.game.board}")
            print(f"Initial dice: {env_unwrapped.dice}")
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            if debug_this_episode:
                print(f"Step {step}, Current player: {'White' if current_player_is_white else 'Black'}")
                print(f"Board: {env_unwrapped.game.board}")
                print(f"Dice: {env_unwrapped.dice}")
                print(f"Valid moves: {valid_moves}")
                print(f"White borne off: {env_unwrapped.game.borne_off_white}")
                print(f"Black borne off: {env_unwrapped.game.borne_off_black}")
            
            if not valid_moves:
                # Reset dice when no valid moves are available
                env_unwrapped.dice = []  # Clear existing dice
                
                if debug_this_episode:
                    print(f"No valid moves, skipping turn")
                
                # Re-roll the dice for the next player
                if hasattr(env_unwrapped, "_roll_dice"):
                    env_unwrapped._roll_dice()
                else:
                    # Fallback to manually rolling dice
                    d1 = random.randint(1, 6)
                    d2 = random.randint(1, 6)
                    if d1 == d2:
                        env_unwrapped.dice = [d1, d1, d1, d1]  # Double dice give four moves
                    else:
                        env_unwrapped.dice = [d1, d2]
                
                # Rotate board for next player
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                
                if debug_this_episode:
                    print(f"Board after rotation: {env_unwrapped.game.board}")
                    print(f"New dice: {env_unwrapped.dice}")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
                
                continue
            
            # If current player is Black, we need to handle the board rotation for DQN
            if not current_player_is_white:
                # Create a rotated board for decision making
                rotated_board = -np.flip(env_unwrapped.game.board)
                original_board = env_unwrapped.game.board.copy()
                env_unwrapped.game.board = rotated_board
                
                # Get a new state observation based on the rotated board
                temp_state = env_unwrapped._get_obs()
                temp_state = torch.FloatTensor(temp_state).to(device)
                
                # Get valid moves from the rotated perspective
                rotated_valid_moves = env_unwrapped.game.get_valid_moves()
                
                if debug_this_episode:
                    print(f"Black's turn - showing rotated board for decision making:")
                    print(f"Rotated board: {rotated_board}")
                    print(f"Rotated valid moves: {rotated_valid_moves}")
                
                # Create action mask for valid moves
                valid_action_mask = torch.zeros(action_dim, device=device)
                
                for move in rotated_valid_moves:
                    from_pos, to_pos = move
                    if to_pos == 'off':
                        # Bear off move
                        action_idx = from_pos * 24 + 0
                        move_type = 1
                        valid_action_mask[action_idx * 2 + move_type] = 1
                    else:
                        # Regular move
                        action_idx = from_pos * 24 + to_pos
                        move_type = 0
                        valid_action_mask[action_idx * 2 + move_type] = 1
                
                # Get model prediction
                with torch.no_grad():
                    model.eval()  # Ensure model is in evaluation mode
                    q_values = model(temp_state)
                    masked_q_values = q_values.clone()
                    valid_action_mask = valid_action_mask.reshape(q_values.shape)
                    masked_q_values[valid_action_mask == 0] = float('-inf')
                    action_index = masked_q_values.argmax().item()
                    
                    # Convert to action format
                    move_type = action_index % 2
                    move_idx = action_index // 2
                    
                    if move_type == 0:  # Regular move
                        from_pos = move_idx // 24
                        to_pos = move_idx % 24
                        rotated_selected_move = (from_pos, to_pos)
                    else:  # Bear off
                        from_pos = move_idx // 24
                        rotated_selected_move = (from_pos, 'off')
                
                # Restore the original board
                env_unwrapped.game.board = original_board
                
                # Map the move from the rotated perspective back to the original board
                if rotated_selected_move[1] == 'off':
                    # For bear-off, just flip the position
                    selected_move = (23 - rotated_selected_move[0], 'off')
                else:
                    # For regular move, flip both positions
                    selected_move = (23 - rotated_selected_move[0], 23 - rotated_selected_move[1])
                
                # Convert the selected move back to the action format for the environment
                if selected_move[1] == 'off':
                    action = (selected_move[0] * 24 + 0, 1)  # bear off action
                else:
                    action = (selected_move[0] * 24 + selected_move[1], 0)  # regular move
                
                if debug_this_episode:
                    print(f"Black (DQN) selected move: {selected_move}")
            else:
                # White's turn - DQN can use the state directly
                # Create action mask for valid moves
                valid_action_mask = torch.zeros(action_dim, device=device)
                
                for move in valid_moves:
                    from_pos, to_pos = move
                    if to_pos == 'off':
                        # Bear off move
                        action_idx = from_pos * 24 + 0
                        move_type = 1
                        valid_action_mask[action_idx * 2 + move_type] = 1
                    else:
                        # Regular move
                        action_idx = from_pos * 24 + to_pos
                        move_type = 0
                        valid_action_mask[action_idx * 2 + move_type] = 1
                
                # Get model prediction
                with torch.no_grad():
                    model.eval()  # Ensure model is in evaluation mode
                    q_values = model(state)
                    masked_q_values = q_values.clone()
                    valid_action_mask = valid_action_mask.reshape(q_values.shape)
                    masked_q_values[valid_action_mask == 0] = float('-inf')
                    action_index = masked_q_values.argmax().item()
                    
                    # Convert to action format
                    move_type = action_index % 2
                    move_idx = action_index // 2
                    
                    if move_type == 0:  # Regular move
                        from_pos = move_idx // 24
                        to_pos = move_idx % 24
                        selected_move = (from_pos, to_pos)
                    else:  # Bear off
                        from_pos = move_idx // 24
                        selected_move = (from_pos, 'off')
                
                # Convert to environment action format
                if selected_move[1] == 'off':
                    action = (from_pos * 24 + 0, 1)  # bear off action
                else:
                    action = (from_pos * 24 + to_pos, 0)  # regular move
                
                if debug_this_episode:
                    print(f"White (DQN) selected move: {selected_move}")
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Log board state after move
            if debug_this_episode:
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Board after move: {env_unwrapped.game.board}")
                print(f"White borne off after move: {env_unwrapped.game.borne_off_white}")
                print(f"Black borne off after move: {env_unwrapped.game.borne_off_black}")
                print("-" * 50)
            
            state = next_state
            step_count += 1
            
            # Track dice state before checking for rotation
            dice_before = env_unwrapped.dice.copy() if env_unwrapped.dice else []
            
            # Check if the environment automatically rotated the board inside step()
            auto_rotated = False
            # Check if "skipped_turn" or "skipped_multiple_turns" in info
            if "skipped_turn" in info or "skipped_multiple_turns" in info:
                auto_rotated = True
                current_player_is_white = not current_player_is_white
                if debug_this_episode:
                    print("Environment automatically rotated board due to skipped turn")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
            
            # Check if dice were depleted and new dice rolled (dice count increased)
            elif len(dice_before) > 0 and len(dice_before) <= 2 and len(env_unwrapped.dice) >= 2 and len(env_unwrapped.dice) > len(dice_before):
                # This suggests dice were depleted and new dice were rolled
                auto_rotated = True
                current_player_is_white = not current_player_is_white
                if debug_this_episode:
                    print("Environment automatically rotated board due to dice depletion")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
            
            if done:
                # Use the environment's built-in get_winner() method
                # This returns +1 if White won, -1 if Black won
                actual_winner = env_unwrapped.game.get_winner()
                
                if debug_this_episode:
                    print(f"Game over: {'White' if actual_winner == 1 else 'Black'} won!")
                    print(f"Final board: {env_unwrapped.game.board}")
                    print(f"White borne off: {env_unwrapped.game.borne_off_white}")
                    print(f"Black borne off: {env_unwrapped.game.borne_off_black}")
                
                # Track which color won
                if actual_winner == 1:  # White won
                    white_wins += 1
                    if debug_this_episode:
                        print("White (DQN) won!")
                else:  # Black won
                    black_wins += 1
                    if debug_this_episode:
                        print("Black (DQN) won!")
                
                if debug_this_episode:
                    print(f"Episode {episode} finished after {step_count} steps")
                
                break
                
            if truncated:
                if debug_this_episode:
                    print(f"Episode truncated after {step_count} steps")
                draws += 1
                break
                
            # Rotate board for next player if needed - only if env didn't already do it
            if not (done or truncated) and not env_unwrapped.dice and not auto_rotated:
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                
                if debug_this_episode:
                    print(f"Board rotated for next player")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'}")
        
        # Track metrics
        episode_steps.append(step_count)
        episode_borne_off_white.append(env_unwrapped.game.borne_off_white)
        episode_borne_off_black.append(env_unwrapped.game.borne_off_black)
        
        # Print progress
        if episode % 10 == 0:
            total_completed = white_wins + black_wins + draws
            print(f"Completed {episode}/{episodes} episodes")
            if total_completed > 0:
                print(f"  White Win Rate: {white_wins/episode:.2%}")
                print(f"  Black Win Rate: {black_wins/episode:.2%}")
                print(f"  Draws: {draws/episode:.2%}")
            print(f"  Average Steps: {sum(episode_steps)/episode:.2f}")
            print(f"  Average White Checkers Borne Off: {sum(episode_borne_off_white)/episode:.2f}")
            print(f"  Average Black Checkers Borne Off: {sum(episode_borne_off_black)/episode:.2f}")
    
    # Print final results
    total_games = white_wins + black_wins + draws
    print("\nDQN vs DQN Results:")
    print(f"Total Episodes: {episodes}")
    if total_games > 0:
        print(f"White Win Rate: {white_wins/episodes:.2%}")
        print(f"Black Win Rate: {black_wins/episodes:.2%}")
        print(f"Draws: {draws/episodes:.2%}")
    print(f"Average Steps Per Episode: {sum(episode_steps)/episodes:.2f}")
    print(f"Maximum Steps in an Episode: {max(episode_steps)}")
    print(f"Average White Checkers Borne Off: {sum(episode_borne_off_white)/episodes:.2f}")
    print(f"Average Black Checkers Borne Off: {sum(episode_borne_off_black)/episodes:.2f}")
    
    # Calculate average steps to bear off each checker
    total_checkers_borne_off = sum(episode_borne_off_white) + sum(episode_borne_off_black)
    if total_checkers_borne_off > 0:
        steps_per_checker = sum(episode_steps) / total_checkers_borne_off
        print(f"Average Steps Per Checker Borne Off: {steps_per_checker:.2f}")
    
    return white_wins/episodes, black_wins/episodes, sum(episode_steps)/episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DQN against itself in Narde')
    parser.add_argument('--model', type=str, default='narde_dqn_model.pth', help='Path to model file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for the first 5 episodes')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Start evaluation
    start_time = time.time()
    white_win_rate, black_win_rate, avg_steps = evaluate_dqn_vs_dqn(
        args.model, 
        args.episodes, 
        args.max_steps, 
        args.verbose
    )
    end_time = time.time()
    
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds") 