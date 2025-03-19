#!/usr/bin/env python3
"""
Script to play and log individual games between DQN and Random agent for analysis.
"""

import gym_narde  # Import this first to register the environment
import gymnasium as gym
import numpy as np
import torch
import random
import argparse
import time
from train_simpledqn import DQN  # Import the DQN model from training script

def play_single_game(model_path, dqn_plays_first=True, max_steps=500, log_file="game_log.txt"):
    """
    Play a single game between DQN and Random agent, with detailed logging.
    
    Args:
        model_path: Path to the trained DQN model
        dqn_plays_first: If True, DQN plays as White (first player), otherwise as Black (second player)
        max_steps: Maximum steps before truncating the game
        log_file: File path to save the game log
    """
    # Create environment
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space[0].n * env.action_space[1].n
    
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize model
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    # Open log file for writing
    with open(log_file, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"NEW GAME: {'DQN' if dqn_plays_first else 'RANDOM'} PLAYS FIRST (WHITE)\n")
        f.write("=" * 80 + "\n\n")
        
        # Reset environment to start the game
        state, info = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        # Game setup
        step_count = 0
        current_player_is_white = True  # Game always starts with White
        
        # Determine which player is DQN based on who starts
        current_player_is_dqn = dqn_plays_first
        
        # Log initial state
        f.write("INITIAL BOARD STATE:\n")
        f.write(f"{env_unwrapped.game.board}\n")
        f.write(f"Initial dice: {env_unwrapped.dice}\n\n")
        
        # Main game loop
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            # Log player turn and state
            player_name = "DQN" if current_player_is_dqn else "RND"
            color_name = "WHITE" if current_player_is_white else "BLACK"
            f.write(f"STEP {step}: {player_name} ({color_name}) TO MOVE\n")
            f.write(f"Dice: {env_unwrapped.dice}\n")
            f.write(f"Valid moves: {valid_moves}\n")
            
            if not valid_moves:
                # No valid moves, skip turn
                f.write("No valid moves available. Skipping turn.\n")
                
                # Reset dice when no valid moves are available
                env_unwrapped.dice = []  # Clear existing dice
                
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
                
                f.write(f"New dice: {env_unwrapped.dice}\n")
                f.write(f"Next player: {('DQN' if current_player_is_dqn else 'RND')} ({('WHITE' if current_player_is_white else 'BLACK')})\n\n")
                continue
            
            # Determine which policy to use based on which player's turn it is
            if current_player_is_dqn:
                # DQN agent's turn
                # Check if the board needs to be rotated for decision making
                if dqn_plays_first != current_player_is_white:  # DQN is playing as Black currently
                    # We need to present the board from Black's perspective as if it were White
                    # Temporarily rotate the board for decision making
                    rotated_board = -np.flip(env_unwrapped.game.board)
                    original_board = env_unwrapped.game.board.copy()
                    env_unwrapped.game.board = rotated_board
                    
                    # Get a new state observation based on the rotated board
                    temp_state = env_unwrapped._get_obs()
                    temp_state = torch.FloatTensor(temp_state).to(device)
                    
                    # Get valid moves from the rotated perspective
                    rotated_valid_moves = env_unwrapped.game.get_valid_moves()
                    
                    f.write("DQN is Black - showing rotated board for decision making:\n")
                    f.write(f"Rotated board: {rotated_board}\n")
                    f.write(f"Rotated valid moves: {rotated_valid_moves}\n")
                else:
                    # DQN is playing as White, use current state
                    temp_state = state
                    rotated_valid_moves = valid_moves
                
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
                    # Ensure mask has the same shape as q_values
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
                    
                    # If DQN is playing as Black, translate the action back to the original board perspective
                    if dqn_plays_first != current_player_is_white:
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
                    else:
                        # DQN is playing as White, use the move directly
                        selected_move = rotated_selected_move
                        # Convert to environment action format
                        if selected_move[1] == 'off':
                            action = (from_pos * 24 + 0, 1)  # bear off action
                        else:
                            action = (from_pos * 24 + to_pos, 0)  # regular move
                    
                    f.write(f"DQN selected move: {selected_move}\n")
            else:
                # Random agent's turn
                move = random.choice(valid_moves)
                from_pos, to_pos = move
                selected_move = move
                
                # Convert to action format
                if to_pos == 'off':
                    action = (from_pos * 24 + 0, 1)  # bear off move
                else:
                    action = (from_pos * 24 + to_pos, 0)  # regular move
                
                f.write(f"Random selected move: {selected_move}\n")
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Track if dice got refilled inside the step function (indicates player switch)
            was_board_rotated = False
            if "skipped_turn" in info or "skipped_multiple_turns" in info:
                was_board_rotated = True
                # The environment automatically rotated the board due to no valid moves
                current_player_is_white = not current_player_is_white
                current_player_is_dqn = not current_player_is_dqn
                f.write("Turn skipped: No valid moves available\n")
                f.write(f"Board rotated for next player internally\n")
                f.write(f"Next player: {('DQN' if current_player_is_dqn else 'RND')} ({('WHITE' if current_player_is_white else 'BLACK')})\n")
                f.write(f"New dice: {env_unwrapped.dice}\n")
            
            # Log board state after move
            f.write(f"Board after move: {env_unwrapped.game.board}\n")
            f.write(f"White borne off: {env_unwrapped.game.borne_off_white}\n")
            f.write(f"Black borne off: {env_unwrapped.game.borne_off_black}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"Dice remaining after move: {env_unwrapped.dice}\n")
            f.write("-" * 50 + "\n\n")
            
            state = next_state
            step_count += 1
            
            if done:
                # Use the environment's built-in get_winner() method
                # This returns +1 if White won, -1 if Black won
                actual_winner = env_unwrapped.game.get_winner()
                winner_name = "WHITE" if actual_winner == 1 else "BLACK"
                winner_agent = "DQN" if ((dqn_plays_first and actual_winner == 1) or 
                                         (not dqn_plays_first and actual_winner == -1)) else "RANDOM"
                
                f.write("GAME OVER\n")
                f.write(f"Winner: {winner_name} ({winner_agent})\n")
                f.write(f"Final board: {env_unwrapped.game.board}\n")
                f.write(f"White borne off: {env_unwrapped.game.borne_off_white}\n")
                f.write(f"Black borne off: {env_unwrapped.game.borne_off_black}\n")
                f.write(f"Game finished after {step_count} steps\n\n")
                break
                
            if truncated:
                f.write("GAME TRUNCATED (max steps reached)\n")
                f.write(f"Final board: {env_unwrapped.game.board}\n")
                f.write(f"White borne off: {env_unwrapped.game.borne_off_white}\n")
                f.write(f"Black borne off: {env_unwrapped.game.borne_off_black}\n")
                f.write(f"Game truncated after {step_count} steps\n\n")
                break
                
            # Track dice state at beginning of previous step to detect automatic rotation
            previous_dice_count = len(env_unwrapped.dice)
            
            # Check for dice that appeared when we previously had few dice
            # This indicates the environment internally rotated the board and refilled dice
            if previous_dice_count == 0 or (previous_dice_count <= 2 and len(env_unwrapped.dice) >= 3):
                # Only switch if we haven't already detected a turn skip
                if not was_board_rotated:
                    current_player_is_white = not current_player_is_white
                    current_player_is_dqn = not current_player_is_dqn
                    f.write(f"DETECTED AUTOMATIC DICE REFILL - Switching players\n")
                    f.write(f"Next player: {('DQN' if current_player_is_dqn else 'RND')} ({('WHITE' if current_player_is_white else 'BLACK')})\n")
                    f.write(f"New dice: {env_unwrapped.dice}\n\n")
                    was_board_rotated = True
                
            # Rotate board for next player if needed - the environment may have already done this
            if not (done or truncated) and not env_unwrapped.dice and not was_board_rotated:
                f.write(f"DICE DEPLETED: {env_unwrapped.dice} - Rotating board and switching players\n")
                
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                current_player_is_dqn = not current_player_is_dqn  # Switch player identity
                
                # Re-roll dice for next player
                if hasattr(env_unwrapped, "_roll_dice"):
                    env_unwrapped._roll_dice()
                
                f.write("Board rotated for next player\n")
                f.write(f"Next player: {('DQN' if current_player_is_dqn else 'RND')} ({('WHITE' if current_player_is_white else 'BLACK')})\n")
                f.write(f"New dice: {env_unwrapped.dice}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Play and log individual games between DQN and Random agent")
    parser.add_argument("--model", type=str, default="narde_dqn_model.pth", help="Path to the model file")
    parser.add_argument("--log_file", type=str, default="game_log.txt", help="File to save the game logs")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per game")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set random seeds if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # Clear the log file
    with open(args.log_file, "w") as f:
        f.write(f"Game log created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Play and log a game where DQN plays first (as White)
    print("Playing game 1: DQN starts as White...")
    play_single_game(args.model, dqn_plays_first=True, max_steps=args.max_steps, log_file=args.log_file)
    
    # Play and log a game where Random plays first (as White)
    print("Playing game 2: Random agent starts as White...")
    play_single_game(args.model, dqn_plays_first=False, max_steps=args.max_steps, log_file=args.log_file)
    
    print(f"Games completed. Logs saved to {args.log_file}")

if __name__ == "__main__":
    main() 