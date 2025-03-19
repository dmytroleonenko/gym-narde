#!/usr/bin/env python3
"""
Evaluation script for the trained DQN agent for Narde.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import torch
import random
import argparse
import time
from train_simpledqn import DQN  # Import the DQN model from training script

def evaluate_dqn(model_path, episodes=100, max_steps=500):
    """
    Evaluate DQN model in a balanced way playing as both first and second player.
    """
    # Evaluate DQN as Black (second player)
    dqn_as_black_wins = 0
    dqn_as_black_steps = 0
    dqn_as_black_count = 0
    
    if args.dqn_as_black:
        print("\nEvaluating DQN as Black (second player) against Random as White...")
        print("-" * 50)
        dqn_as_black_win_rate, dqn_as_black_avg_steps = evaluate_dqn_as_black(model_path, episodes, max_steps)
        dqn_as_black_wins = dqn_as_black_win_rate * episodes
        dqn_as_black_steps = dqn_as_black_avg_steps * episodes
        dqn_as_black_count = episodes
    
    # Evaluate DQN as White (first player)
    dqn_as_white_wins = 0
    dqn_as_white_steps = 0
    dqn_as_white_count = 0
    
    if args.dqn_as_white:
        print("\nEvaluating DQN as White (first player) against Random as Black...")
        print("-" * 50)
        dqn_as_white_win_rate, dqn_as_white_avg_steps = evaluate_dqn_as_white(model_path, episodes, max_steps)
        dqn_as_white_wins = dqn_as_white_win_rate * episodes
        dqn_as_white_steps = dqn_as_white_avg_steps * episodes
        dqn_as_white_count = episodes
    
    # Overall statistics
    print("\nEvaluation Results:")
    print(f"Total Episodes: {dqn_as_black_count + dqn_as_white_count}")
    
    if dqn_as_white_count > 0:
    print(f"DQN Win Rate as White: {dqn_as_white_wins/dqn_as_white_count:.2%}")
    else:
        print("DQN was not evaluated as White")
        
    if dqn_as_black_count > 0:
    print(f"DQN Win Rate as Black: {dqn_as_black_wins/dqn_as_black_count:.2%}")
    else:
        print("DQN was not evaluated as Black")
    
    # Calculate overall win rate
    total_episodes = dqn_as_black_count + dqn_as_white_count
    if total_episodes > 0:
        total_wins = dqn_as_black_wins + dqn_as_white_wins
        total_steps = dqn_as_black_steps + dqn_as_white_steps
        return total_wins/total_episodes, total_steps/total_episodes
    else:
        return 0, 0

def evaluate_random_agent(episodes=50, max_steps=500):
    """Evaluate a random agent against another random agent for comparison"""
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    white_wins = 0
    black_wins = 0
    draws = 0
    episode_steps = []
    episode_borne_off_white = []
    episode_borne_off_black = []
    
    print("Evaluating random vs random...")
    print("-" * 60)
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        
        step_count = 0
        current_player_is_white = True  # Game always starts with White
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            if not valid_moves:
                # Fix: Reset dice when no valid moves are available
                env_unwrapped.dice = []  # Clear existing dice
                
                # Re-roll the dice for the next player using the correct method
                if hasattr(env_unwrapped, "_roll_dice"):
                    env_unwrapped._roll_dice()  # Use environment's method which handles doubles
                else:
                    # Fallback to manually rolling dice if method doesn't exist
                    d1 = random.randint(1, 6)
                    d2 = random.randint(1, 6)
                    if d1 == d2:
                        env_unwrapped.dice = [d1, d1, d1, d1]  # Double dice give four moves
                    else:
                        env_unwrapped.dice = [d1, d2]
                
                # Rotate board for next player
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                continue
                
            # Choose random move
            move = random.choice(valid_moves)
            from_pos, to_pos = move
            
            # Convert to action format
            if to_pos == 'off':
                action = (from_pos * 24 + 0, 1)  # bear off move
            else:
                action = (from_pos * 24 + to_pos, 0)  # regular move
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            
            if done:
                # Track which color won based on borne_off counts
                if env_unwrapped.game.borne_off_white == 15:
                    if current_player_is_white:
                    white_wins += 1
                    else:
                        black_wins += 1
                elif env_unwrapped.game.borne_off_black == 15:
                    if current_player_is_white:
                    black_wins += 1
                    else:
                        white_wins += 1
                else:
                    # Draw or timeout
                    draws += 1
                break
                
            if truncated:
                draws += 1
                break
                
            # Rotate board for next player
            if not (done or truncated) and not env_unwrapped.dice:
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
        
        # Track metrics
        episode_steps.append(step_count)
        episode_borne_off_white.append(env_unwrapped.game.borne_off_white)
        episode_borne_off_black.append(env_unwrapped.game.borne_off_black)
        
        # Print progress
        if episode % 10 == 0:
            total_completed = white_wins + black_wins + draws
            print(f"Completed {episode}/{episodes} episodes")
            if total_completed > 0:
                print(f"  White Win Rate: {white_wins/total_completed:.2%}")
                print(f"  Black Win Rate: {black_wins/total_completed:.2%}")
                print(f"  Draws: {draws/total_completed:.2%}")
            print(f"  Average Steps: {sum(episode_steps)/episode:.2f}")
            print(f"  Average White Checkers Borne Off: {sum(episode_borne_off_white)/episode:.2f}")
            print(f"  Average Black Checkers Borne Off: {sum(episode_borne_off_black)/episode:.2f}")
    
    # Print final results
    total_games = white_wins + black_wins + draws
    print("\nRandom Agent Results:")
    print(f"Total Episodes: {episodes}")
    if total_games > 0:
        print(f"White Win Rate: {white_wins/total_games:.2%}")
        print(f"Black Win Rate: {black_wins/total_games:.2%}")
        print(f"Draws: {draws/total_games:.2%}")
    print(f"Average Steps Per Episode: {sum(episode_steps)/episodes:.2f}")
    print(f"Maximum Steps in an Episode: {max(episode_steps)}")
    print(f"Average White Checkers Borne Off: {sum(episode_borne_off_white)/episodes:.2f}")
    print(f"Average Black Checkers Borne Off: {sum(episode_borne_off_black)/episodes:.2f}")
    
    # Calculate average steps to bear off each checker
    total_checkers_borne_off = sum(episode_borne_off_white) + sum(episode_borne_off_black)
    if total_checkers_borne_off > 0:
        steps_per_checker = sum(episode_steps) / total_checkers_borne_off
        print(f"Average Steps Per Checker Borne Off: {steps_per_checker:.2f}")
    
    # Return the average win rate across both colors (comparable to DQN evaluation)
    # For random vs random, we need to count each color position separately
    # This is equivalent to (white_wins + black_wins)/(episodes*2)
    # Since each episode has one White and one Black player
    return (white_wins + black_wins)/(episodes*2), sum(episode_steps)/episodes, white_wins/episodes, black_wins/episodes

def evaluate_dqn_as_black(model_path, episodes=100, max_steps=500, render=False):
    """
    Evaluate DQN model specifically when it plays as Black (second player)
    against a random agent that always plays as White (first player).
    This helps isolate the effect of the first-player advantage.
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
    dqn_wins = 0
    random_wins = 0
    episode_steps = []
    episode_borne_off_white = []
    episode_borne_off_black = []
    dqn_checkers_borne_off = []
    total_bear_offs = 0
    
    print(f"Evaluating model: {model_path}")
    print("Testing scenario: Random agent ALWAYS plays as White (first player)")
    print("                  DQN ALWAYS plays as Black (second player)")
    print("-" * 60)
    
    # Enable verbose debug for first 5 episodes
    debug_episodes = 5
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        step_count = 0
        episode_bear_offs = 0
        
        # In this scenario, DQN ALWAYS plays as Black (second player)
        # Random agent ALWAYS plays as White (first player)
        dqn_plays_as_white = False
        
        # Initially, Random is the current player (White/first player)
        current_player_is_dqn = False
        current_player_is_white = True  # Game always starts with White
        
        # Debug info
        verbose = episode <= debug_episodes
        if verbose:
            print(f"\nEpisode {episode} starting")
            print(f"Random agent playing as White (first player)")
            print(f"DQN playing as Black (second player)")
            print(f"Initial board state: {env_unwrapped.game.board}")
            print(f"Initial dice: {env_unwrapped.dice}")
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            if verbose:
                print(f"Step {step}, Current player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
                print(f"Board: {env_unwrapped.game.board}")
                print(f"Dice: {env_unwrapped.dice}")
                print(f"Valid moves: {valid_moves}")
                print(f"White borne off: {env_unwrapped.game.borne_off_white}")
                print(f"Black borne off: {env_unwrapped.game.borne_off_black}")
            
            if not valid_moves:
                # Reset dice when no valid moves are available
                env_unwrapped.dice = []  # Clear existing dice
                
                if verbose:
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
                current_player_is_dqn = not current_player_is_dqn  # Switch player identity
                
                if verbose:
                    print(f"Board after rotation: {env_unwrapped.game.board}")
                    print(f"New dice: {env_unwrapped.dice}")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
                
                continue
            
            # Determine which policy to use based on which player's turn it is
            if current_player_is_dqn:
                # When DQN plays as Black, we need to properly handle the perspective
                # Create a rotated board for decision making
                rotated_board = -np.flip(env_unwrapped.game.board)
                original_board = env_unwrapped.game.board.copy()
                env_unwrapped.game.board = rotated_board
                
                # Get a new state observation based on the rotated board
                temp_state = env_unwrapped._get_obs()
                temp_state = torch.FloatTensor(temp_state).to(device)
                
                # Get valid moves from the rotated perspective
                rotated_valid_moves = env_unwrapped.game.get_valid_moves()
                
                if verbose:
                    print(f"DQN is Black - showing rotated board for decision making:")
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
                        episode_bear_offs += 1
                
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
                
                if verbose:
                    print(f"DQN selected move: {selected_move}")
            else:
                # Random policy for opponent (White)
                move = random.choice(valid_moves)
                from_pos, to_pos = move
                selected_move = move
                
                # Convert to action format
                if to_pos == 'off':
                    action = (from_pos * 24 + 0, 1)  # bear off move
                else:
                    action = (from_pos * 24 + to_pos, 0)  # regular move
                
                if verbose:
                    print(f"Random selected move: {selected_move}")
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Log board state after move
            if verbose:
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
                current_player_is_dqn = not current_player_is_dqn
                if verbose:
                    print("Environment automatically rotated board due to skipped turn")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
            
            # Check if dice were depleted and new dice rolled (dice count increased)
            elif len(dice_before) > 0 and len(dice_before) <= 2 and len(env_unwrapped.dice) >= 2 and len(env_unwrapped.dice) > len(dice_before):
                # This suggests dice were depleted and new dice were rolled
                auto_rotated = True
                current_player_is_white = not current_player_is_white
                current_player_is_dqn = not current_player_is_dqn
                if verbose:
                    print("Environment automatically rotated board due to dice depletion")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
            
            if done:
                # Use the environment's built-in get_winner() method
                # This returns +1 if White won, -1 if Black won
                actual_winner = env_unwrapped.game.get_winner()
                
                if verbose:
                    print(f"Game over: {'White' if actual_winner == 1 else 'Black'} won!")
                    print(f"Final board: {env_unwrapped.game.board}")
                    print(f"White borne off: {env_unwrapped.game.borne_off_white}")
                    print(f"Black borne off: {env_unwrapped.game.borne_off_black}")
                
                # Simplified win tracking logic - DQN always plays as Black
                if actual_winner == -1:  # Black (DQN) won
                    dqn_wins += 1
                    if verbose:
                        print("DQN (Black) won!")
                else:  # White (Random) won
                    random_wins += 1
                    if verbose:
                        print("Random (White) won!")
                
                if verbose:
                    print(f"Episode {episode} finished after {step_count} steps")
                    print(f"Current DQN (Black) win rate: {dqn_wins}/{episode}")
                
                break
                
            if truncated:
                if verbose:
                    print(f"Episode truncated after {step_count} steps")
                break
                
            # Rotate board for next player if needed - only if env didn't already do it
            if not (done or truncated) and not env_unwrapped.dice and not auto_rotated:
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                current_player_is_dqn = not current_player_is_dqn  # Switch player identity
                
                if verbose:
                    print(f"Board rotated for next player")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
        
        # Track metrics
        episode_steps.append(step_count)
        episode_borne_off_white.append(env_unwrapped.game.borne_off_white)
        episode_borne_off_black.append(env_unwrapped.game.borne_off_black)
        dqn_checkers_borne_off.append(env_unwrapped.game.borne_off_black)  # DQN is Black
        total_bear_offs += episode_bear_offs
        
        # Print progress
        if episode % 10 == 0:
            print(f"Completed {episode}/{episodes} episodes")
            print(f"  DQN (Black) Win Rate: {dqn_wins/episode:.2%}")
            print(f"  Random (White) Win Rate: {random_wins/episode:.2%}")
            print(f"  Average Steps: {sum(episode_steps)/episode:.2f}")
            print(f"  Average White (Random) Checkers Borne Off: {sum(episode_borne_off_white)/episode:.2f}")
            print(f"  Average Black (DQN) Checkers Borne Off: {sum(episode_borne_off_black)/episode:.2f}")
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Total Episodes: {episodes}")
    print(f"DQN (Black) Win Rate: {dqn_wins/episodes:.2%}")
    print(f"Random (White) Win Rate: {random_wins/episodes:.2%}")
    print(f"Average Steps Per Episode: {sum(episode_steps)/episodes:.2f}")
    print(f"Maximum Steps in an Episode: {max(episode_steps)}")
    print(f"Average White (Random) Checkers Borne Off: {sum(episode_borne_off_white)/episodes:.2f}")
    print(f"Average Black (DQN) Checkers Borne Off: {sum(episode_borne_off_black)/episodes:.2f}")
    print(f"Total Bear Off Moves: {total_bear_offs}")
    
    # Calculate average steps to bear off each checker
    total_checkers_borne_off = sum(episode_borne_off_white) + sum(episode_borne_off_black)
    if total_checkers_borne_off > 0:
        steps_per_checker = sum(episode_steps) / total_checkers_borne_off
        print(f"Average Steps Per Checker Borne Off: {steps_per_checker:.2f}")
    
    return dqn_wins/episodes, sum(episode_steps)/episodes

def evaluate_dqn_as_white(model_path, episodes=100, max_steps=500, render=False):
    """
    Evaluate DQN model specifically when it plays as White (first player)
    against a random agent that always plays as Black (second player).
    This helps isolate the effect of the first-player advantage.
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
    dqn_wins = 0
    random_wins = 0
    episode_steps = []
    episode_borne_off_white = []
    episode_borne_off_black = []
    dqn_checkers_borne_off = []
    total_bear_offs = 0
    
    print(f"Evaluating model: {model_path}")
    print("Testing scenario: DQN ALWAYS plays as White (first player)")
    print("                  Random agent ALWAYS plays as Black (second player)")
    print("-" * 60)
    
    # Enable verbose debug for first 5 episodes
    debug_episodes = 5
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        step_count = 0
        episode_bear_offs = 0
        
        # In this scenario, DQN ALWAYS plays as White (first player)
        # Random agent ALWAYS plays as Black (second player)
        dqn_plays_as_white = True
        
        # Initially, DQN is the current player (White/first player)
        current_player_is_dqn = True
        current_player_is_white = True  # Game always starts with White
        
        # Debug info
        verbose = episode <= debug_episodes
        if verbose:
            print(f"\nEpisode {episode} starting")
            print(f"DQN playing as White (first player)")
            print(f"Random agent playing as Black (second player)")
            print(f"Initial board state: {env_unwrapped.game.board}")
            print(f"Initial dice: {env_unwrapped.dice}")
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            if verbose:
                print(f"Step {step}, Current player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
                print(f"Board: {env_unwrapped.game.board}")
                print(f"Dice: {env_unwrapped.dice}")
                print(f"Valid moves: {valid_moves}")
                print(f"White borne off: {env_unwrapped.game.borne_off_white}")
                print(f"Black borne off: {env_unwrapped.game.borne_off_black}")
            
            if not valid_moves:
                # Reset dice when no valid moves are available
                env_unwrapped.dice = []  # Clear existing dice
                
                if verbose:
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
                current_player_is_dqn = not current_player_is_dqn  # Switch player identity
                
                if verbose:
                    print(f"Board after rotation: {env_unwrapped.game.board}")
                    print(f"New dice: {env_unwrapped.dice}")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
                
                continue
            
            # Determine which policy to use based on which player's turn it is
            if current_player_is_dqn:
                # When DQN plays as White, it can use the observation directly
                # No need to rotate perspective since it's trained from White's view
                
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
                        episode_bear_offs += 1
                
                # Convert to environment action format
                if selected_move[1] == 'off':
                    action = (from_pos * 24 + 0, 1)  # bear off action
                else:
                    action = (from_pos * 24 + to_pos, 0)  # regular move
                
                if verbose:
                    print(f"DQN selected move: {selected_move}")
            else:
                # Random policy for opponent (Black)
                move = random.choice(valid_moves)
                from_pos, to_pos = move
                selected_move = move
                
                # Convert to action format
                if to_pos == 'off':
                    action = (from_pos * 24 + 0, 1)  # bear off move
                else:
                    action = (from_pos * 24 + to_pos, 0)  # regular move
                
                if verbose:
                    print(f"Random selected move: {selected_move}")
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Log board state after move
            if verbose:
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
                current_player_is_dqn = not current_player_is_dqn
                if verbose:
                    print("Environment automatically rotated board due to skipped turn")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
            
            # Check if dice were depleted and new dice rolled (dice count increased)
            elif len(dice_before) > 0 and len(dice_before) <= 2 and len(env_unwrapped.dice) >= 2 and len(env_unwrapped.dice) > len(dice_before):
                # This suggests dice were depleted and new dice were rolled
                auto_rotated = True
                current_player_is_white = not current_player_is_white
                current_player_is_dqn = not current_player_is_dqn
                if verbose:
                    print("Environment automatically rotated board due to dice depletion")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
            
            if done:
                # Use the environment's built-in get_winner() method
                # This returns +1 if White won, -1 if Black won
                actual_winner = env_unwrapped.game.get_winner()
                
                if verbose:
                    print(f"Game over: {'White' if actual_winner == 1 else 'Black'} won!")
                    print(f"Final board: {env_unwrapped.game.board}")
                    print(f"White borne off: {env_unwrapped.game.borne_off_white}")
                    print(f"Black borne off: {env_unwrapped.game.borne_off_black}")
                
                # Simplified win tracking logic - DQN always plays as White
                if actual_winner == 1:  # White (DQN) won
                    dqn_wins += 1
                    if verbose:
                        print("DQN (White) won!")
                else:  # Black (Random) won
                    random_wins += 1
                    if verbose:
                        print("Random (Black) won!")
                
                if verbose:
                    print(f"Episode {episode} finished after {step_count} steps")
                    print(f"Current DQN (White) win rate: {dqn_wins}/{episode}")
                
                break
                
            if truncated:
                if verbose:
                    print(f"Episode truncated after {step_count} steps")
                break
                
            # Rotate board for next player if needed - only if env didn't already do it
            if not (done or truncated) and not env_unwrapped.dice and not auto_rotated:
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                current_player_is_dqn = not current_player_is_dqn  # Switch player identity
                
                if verbose:
                    print(f"Board rotated for next player")
                    print(f"Next player: {'White' if current_player_is_white else 'Black'} ({'DQN' if current_player_is_dqn else 'Random'})")
        
        # Track metrics
        episode_steps.append(step_count)
        episode_borne_off_white.append(env_unwrapped.game.borne_off_white)
        episode_borne_off_black.append(env_unwrapped.game.borne_off_black)
        dqn_checkers_borne_off.append(env_unwrapped.game.borne_off_white)  # DQN is White
        total_bear_offs += episode_bear_offs
        
        # Print progress
        if episode % 10 == 0:
            print(f"Completed {episode}/{episodes} episodes")
            print(f"  DQN (White) Win Rate: {dqn_wins/episode:.2%}")
            print(f"  Random (Black) Win Rate: {random_wins/episode:.2%}")
            print(f"  Average Steps: {sum(episode_steps)/episode:.2f}")
            print(f"  Average White (DQN) Checkers Borne Off: {sum(episode_borne_off_white)/episode:.2f}")
            print(f"  Average Black (Random) Checkers Borne Off: {sum(episode_borne_off_black)/episode:.2f}")
    
    # Print final results
    print("\nEvaluation Results:")
    print(f"Total Episodes: {episodes}")
    print(f"DQN (White) Win Rate: {dqn_wins/episodes:.2%}")
    print(f"Random (Black) Win Rate: {random_wins/episodes:.2%}")
    print(f"Average Steps Per Episode: {sum(episode_steps)/episodes:.2f}")
    print(f"Maximum Steps in an Episode: {max(episode_steps)}")
    print(f"Average White (DQN) Checkers Borne Off: {sum(episode_borne_off_white)/episodes:.2f}")
    print(f"Average Black (Random) Checkers Borne Off: {sum(episode_borne_off_black)/episodes:.2f}")
    print(f"Total Bear Off Moves: {total_bear_offs}")
    
    # Calculate average steps to bear off each checker
    total_checkers_borne_off = sum(episode_borne_off_white) + sum(episode_borne_off_black)
    if total_checkers_borne_off > 0:
        steps_per_checker = sum(episode_steps) / total_checkers_borne_off
        print(f"Average Steps Per Checker Borne Off: {steps_per_checker:.2f}")
    
    return dqn_wins/episodes, sum(episode_steps)/episodes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DQN agent performance on Narde')
    parser.add_argument('--model', type=str, default='narde_dqn_model.pth', help='Path to model file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--random', action='store_true', help='Evaluate only random agent against itself')
    parser.add_argument('--random_vs_random', action='store_true', help='Evaluate random vs random')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--dqn_as_black', action='store_true', help='Evaluate DQN playing as Black')
    parser.add_argument('--dqn_as_white', action='store_true', help='Evaluate DQN playing as White')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if args.random:
        evaluate_random_agent(args.episodes, args.max_steps)
    elif args.random_vs_random:
        random_vs_random(args.episodes, args.max_steps)
    else:
        # If neither specific option is set, default to evaluate both 
        if not args.dqn_as_black and not args.dqn_as_white:
            args.dqn_as_black = True
            args.dqn_as_white = True
            
        dqn_win_rate, dqn_avg_steps = evaluate_dqn(args.model, args.episodes, args.max_steps)
        
        # Print overall results
        if args.dqn_as_black and args.dqn_as_white:
            print(f"Overall DQN Win Rate: {dqn_win_rate:.2%}")
            print(f"Overall Average Steps: {dqn_avg_steps:.2f}") 