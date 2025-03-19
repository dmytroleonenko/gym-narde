#!/usr/bin/env python3
import gymnasium as gym
import gym_narde  # Import to register the environment
import time
import random
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

def test_narde_fairness(episodes=100, max_steps=300, print_board=False, debug=False):
    """Test if Narde game is fair between white and black players by comparing win rates"""
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Set environment debug mode if requested
    if hasattr(env_unwrapped, 'debug'):
        env_unwrapped.debug = debug
    
    white_wins = 0
    black_wins = 0
    draws = 0
    episode_steps = []
    
    print(f"Testing fairness with {episodes} random vs random games...")
    
    for episode in tqdm(range(1, episodes + 1)):
        state, info = env.reset()
        
        step_count = 0
        current_player_is_white = True  # Game always starts with White
        
        # Print initial board state if requested
        if print_board and episode == 1:
            print("\nInitial board state:")
            env_unwrapped.game.print_board()
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            if not valid_moves:
                # No valid moves - reset dice and rotate board
                env_unwrapped.dice = []  # Clear existing dice
                
                # Re-roll the dice
                if hasattr(env_unwrapped, "_roll_dice"):
                    env_unwrapped._roll_dice()
                else:
                    # Fallback
                    d1 = random.randint(1, 6)
                    d2 = random.randint(1, 6)
                    if d1 == d2:
                        env_unwrapped.dice = [d1, d1, d1, d1]
                    else:
                        env_unwrapped.dice = [d1, d2]
                
                # Rotate board for next player
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
                continue
            
            # Choose random move
            move = random.choice(valid_moves)
            from_pos, to_pos = move
            
            # Convert to action format expected by environment
            if to_pos == 'off':
                action = (from_pos * 24 + 0, 1)  # bear off move
            else:
                action = (from_pos * 24 + to_pos, 0)  # regular move
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            
            # Print board state periodically if requested
            if print_board and episode == 1 and step_count % 20 == 0:
                print(f"\nStep {step_count}:")
                print(f"Current player: {'White' if current_player_is_white else 'Black'}")
                print(f"Move: {move}")
                env_unwrapped.game.print_board()
            
            if done:
                # Check which color won based on reward
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
                    
                if print_board and episode == 1:
                    print("\nFinal board state:")
                    env_unwrapped.game.print_board()
                    print(f"Winner: {'White' if white_wins > black_wins else 'Black'}")
                
                break
                
            if truncated or step >= max_steps - 1:
                draws += 1
                break
            
            # If game not over, rotate board for next player
            if not env_unwrapped.dice:  # If no more dice, rotate board
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_white = not current_player_is_white
        
        # Track metrics
        episode_steps.append(step_count)
        
        # Print progress periodically
        if episode % 20 == 0:
            total_completed = white_wins + black_wins + draws
            print(f"\nCompleted {episode}/{episodes} episodes")
            if total_completed > 0:
                print(f"  White Win Rate: {white_wins/total_completed:.2%}")
                print(f"  Black Win Rate: {black_wins/total_completed:.2%}")
                print(f"  Draws: {draws/total_completed:.2%}")
            print(f"  Average Steps: {sum(episode_steps)/episode:.2f}")
    
    # Print final results
    total_games = white_wins + black_wins + draws
    print("\nNarde Fairness Test Results:")
    print(f"Total Episodes: {episodes}")
    if total_games > 0:
        print(f"White Win Rate: {white_wins/total_games:.2%}")
        print(f"Black Win Rate: {black_wins/total_games:.2%}")
        print(f"Draws: {draws/total_games:.2%}")
    print(f"Average Steps Per Episode: {sum(episode_steps)/episodes:.2f}")
    print(f"Maximum Steps in an Episode: {max(episode_steps)}")
    
    # Chi-square test for fairness
    if white_wins + black_wins > 0:
        # Expected frequency if game is fair: 50/50 split between White and Black
        expected = [(white_wins + black_wins) / 2, (white_wins + black_wins) / 2]
        observed = [white_wins, black_wins]
        
        chi2, p = stats.chisquare(observed, expected)
        print(f"\nStatistical Analysis:")
        print(f"Chi-square test: chi2={chi2:.4f}, p={p:.4f}")
        if p < 0.05:
            print("The game appears to be UNFAIR (statistically significant difference)")
        else:
            print("The game appears to be FAIR (no statistically significant difference)")
    
    env.close()
    return white_wins, black_wins, draws

def test_starting_player_fairness(episodes=100, max_steps=300, print_board=False, debug=False):
    """Test if starting player has advantage by comparing win rates across different configurations"""
    env = gym.make('Narde-v0')
    env_unwrapped = env.unwrapped
    
    # Set environment debug mode if requested
    if hasattr(env_unwrapped, 'debug'):
        env_unwrapped.debug = debug
    
    # Track outcomes
    first_player_wins = 0  # First player (White) wins
    second_player_wins = 0  # Second player (Black) wins
    draws = 0
    total_steps = []
    
    print(f"Testing first player advantage with {episodes} random vs random games...")
    
    for episode in tqdm(range(1, episodes + 1)):
        state, info = env.reset()
        
        step_count = 0
        current_player_is_first = True  # Game always starts with the first player (White)
        
        # Print initial board state if requested
        if print_board and episode == 1:
            print("\nInitial board state:")
            env_unwrapped.game.print_board()
        
        for step in range(max_steps):
            # Get valid moves
            valid_moves = env_unwrapped.game.get_valid_moves()
            
            if not valid_moves:
                # No valid moves - reset dice and rotate board
                env_unwrapped.dice = []  # Clear existing dice
                
                # Re-roll the dice
                if hasattr(env_unwrapped, "_roll_dice"):
                    env_unwrapped._roll_dice()
                else:
                    # Fallback
                    d1 = random.randint(1, 6)
                    d2 = random.randint(1, 6)
                    if d1 == d2:
                        env_unwrapped.dice = [d1, d1, d1, d1]
                    else:
                        env_unwrapped.dice = [d1, d2]
                
                # Rotate board for next player
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_first = not current_player_is_first
                continue
            
            # Choose random move
            move = random.choice(valid_moves)
            from_pos, to_pos = move
            
            # Convert to action format expected by environment
            if to_pos == 'off':
                action = (from_pos * 24 + 0, 1)  # bear off move
            else:
                action = (from_pos * 24 + to_pos, 0)  # regular move
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            
            if done:
                # In Narde-v0, the current player is White from the perspective of the env
                # So we need to track if the current player is the first or second player
                if env_unwrapped.game.borne_off_white == 15:
                    # White won
                    if current_player_is_first:
                        first_player_wins += 1
                    else:
                        second_player_wins += 1
                elif env_unwrapped.game.borne_off_black == 15:
                    # Black won
                    if current_player_is_first:
                        second_player_wins += 1
                    else:
                        first_player_wins += 1
                else:
                    # Draw
                    draws += 1
                
                # Print final state for the first episode
                if print_board and episode == 1:
                    print("\nFinal board state:")
                    env_unwrapped.game.print_board()
                    if env_unwrapped.game.borne_off_white == 15:
                        print("White won")
                    elif env_unwrapped.game.borne_off_black == 15:
                        print("Black won")
                    else:
                        print("Draw")
                
                break
                
            if truncated or step >= max_steps - 1:
                draws += 1
                break
            
            # If game not over, rotate board for next player
            if not env_unwrapped.dice:  # If no more dice, rotate board
                env_unwrapped.game.rotate_board_for_next_player()
                current_player_is_first = not current_player_is_first
        
        # Track metrics
        total_steps.append(step_count)
        
        # Print progress periodically
        if episode % 20 == 0:
            completed = first_player_wins + second_player_wins + draws
            print(f"\nCompleted {episode}/{episodes} episodes")
            if completed > 0:
                print(f"  First Player (White) Win Rate: {first_player_wins/completed:.2%}")
                print(f"  Second Player (Black) Win Rate: {second_player_wins/completed:.2%}")
                print(f"  Draws: {draws/completed:.2%}")
            print(f"  Average Steps: {sum(total_steps)/episode:.2f}")
    
    # Calculate statistics
    total_games = first_player_wins + second_player_wins + draws
    win_games = first_player_wins + second_player_wins  # Excluding draws
    
    # Print final results
    print("\nFirst Player Advantage Test Results:")
    print(f"Total Episodes: {episodes}")
    
    if total_games > 0:
        print(f"First Player (White) Win Rate: {first_player_wins/total_games:.2%}")
        print(f"Second Player (Black) Win Rate: {second_player_wins/total_games:.2%}")
        print(f"Draws: {draws/total_games:.2%}")
        print(f"Average Steps Per Episode: {sum(total_steps)/episodes:.2f}")
        print(f"Maximum Steps in an Episode: {max(total_steps)}")
        
        # Statistical test (excluding draws)
        if win_games > 0:
            expected = [win_games / 2, win_games / 2]
            observed = [first_player_wins, second_player_wins]
            
            chi2, p = stats.chisquare(observed, expected)
            print(f"\nStatistical Analysis for first player advantage:")
            print(f"Chi-square test: chi2={chi2:.4f}, p={p:.4f}")
            if p < 0.05:
                print("The first player appears to have an ADVANTAGE (statistically significant difference)")
            else:
                print("The first player does NOT appear to have an advantage (no statistically significant difference)")
    
    env.close()
    return first_player_wins, second_player_wins, draws

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test if Narde game is fair between White and Black players")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--print-board", action="store_true", help="Print board state for the first episode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--test-starting-player", action="store_true", help="Test starting player advantage")
    
    args = parser.parse_args()
    
    # Set random seeds if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    if args.test_starting_player:
        test_starting_player_fairness(args.episodes, args.max_steps, args.print_board, args.debug)
    else:
        test_narde_fairness(args.episodes, args.max_steps, args.print_board, args.debug) 