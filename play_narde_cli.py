#!/usr/bin/env python3
"""
Interactive CLI for playing Narde game manually.
"""

import sys
import gymnasium as gym
import numpy as np

# Import Narde environment 
import gym_narde

def print_board(board, dice, borne_off_white, borne_off_black):
    """Print the board state in a user-friendly format."""
    # Display board with position numbers (0-23)
    print("\nBoard State:")
    print("+-----+-----+-----+-----+-----+-----+")
    
    # First row (positions 0-5)
    print("|", end="")
    for i in range(6):
        print(f" {i:2d}  |", end="")
    print()
    
    print("|", end="")
    for i in range(6):
        pieces = board[i]
        if pieces > 0:
            print(f" W{pieces:2d} |", end="")
        elif pieces < 0:
            print(f" B{abs(pieces):2d} |", end="")
        else:
            print("     |", end="")
    print()
    
    print("+-----+-----+-----+-----+-----+-----+")
    
    # Second row (positions 6-11)
    print("|", end="")
    for i in range(6, 12):
        print(f" {i:2d}  |", end="")
    print()
    
    print("|", end="")
    for i in range(6, 12):
        pieces = board[i]
        if pieces > 0:
            print(f" W{pieces:2d} |", end="")
        elif pieces < 0:
            print(f" B{abs(pieces):2d} |", end="")
        else:
            print("     |", end="")
    print()
    
    print("+-----+-----+-----+-----+-----+-----+")
    
    # Third row (positions 12-17)
    print("|", end="")
    for i in range(12, 18):
        print(f" {i:2d}  |", end="")
    print()
    
    print("|", end="")
    for i in range(12, 18):
        pieces = board[i]
        if pieces > 0:
            print(f" W{pieces:2d} |", end="")
        elif pieces < 0:
            print(f" B{abs(pieces):2d} |", end="")
        else:
            print("     |", end="")
    print()
    
    print("+-----+-----+-----+-----+-----+-----+")
    
    # Fourth row (positions 18-23)
    print("|", end="")
    for i in range(18, 24):
        print(f" {i:2d}  |", end="")
    print()
    
    print("|", end="")
    for i in range(18, 24):
        pieces = board[i]
        if pieces > 0:
            print(f" W{pieces:2d} |", end="")
        elif pieces < 0:
            print(f" B{abs(pieces):2d} |", end="")
        else:
            print("     |", end="")
    print()
    
    print("+-----+-----+-----+-----+-----+-----+")
    
    # Show dice and borne off pieces
    print(f"\nDice: {dice}")
    print(f"Borne off - White: {borne_off_white}, Black: {borne_off_black}")
    print(f"Current player: White")  # Always white from env perspective

def main():
    """Run the interactive CLI for Narde."""
    print("=== Narde Game CLI ===")
    print("You will play as white, and after your move,")
    print("the board will be rotated so you'll always play as white.")
    
    # Create and initialize the Narde environment
    env = gym.make('Narde-v0')
    obs, info = env.reset()
    
    # Extract initial state
    board = env.unwrapped.game.board
    dice = env.unwrapped.dice
    borne_off_white = env.unwrapped.game.borne_off_white
    borne_off_black = env.unwrapped.game.borne_off_black
    
    # Game loop
    done = False
    reward = 0
    turn_count = 0
    
    while not done:
        turn_count += 1
        print(f"\n=== Turn {turn_count} ===")
        
        # Display current state
        print_board(board, dice, borne_off_white, borne_off_black)
        
        # Get valid moves
        valid_moves = env.unwrapped.game.get_valid_moves(dice)
        
        if not valid_moves:
            print("\nNo valid moves available. Skipping turn.")
            # Roll dice for next turn
            env.unwrapped.game.rotate_board_for_next_player()
            env.unwrapped._roll_dice()
            
            # Update state
            board = env.unwrapped.game.board
            dice = env.unwrapped.dice
            borne_off_white = env.unwrapped.game.borne_off_white
            borne_off_black = env.unwrapped.game.borne_off_black
            continue
        
        # Display valid moves with indices
        print("\nValid moves:")
        for i, move in enumerate(valid_moves):
            from_pos, to_pos = move
            to_str = f"position {to_pos}" if to_pos != 'off' else "bear off"
            print(f"[{i}] From position {from_pos} to {to_str}")
        
        # Get user input
        try:
            choice = input("\nEnter move index (or 'q' to quit): ")
            if choice.lower() == 'q':
                print("Exiting the game. Thanks for playing!")
                break
                
            move_idx = int(choice)
            if move_idx < 0 or move_idx >= len(valid_moves):
                print("Invalid index. Please try again.")
                continue
                
            selected_move = valid_moves[move_idx]
            from_pos, to_pos = selected_move
            
            # Convert move to gym action format
            if to_pos == 'off':
                move_index = from_pos * 24
                move_type = 1  # Bearing off
            else:
                move_index = from_pos * 24 + to_pos
                move_type = 0  # Regular move
                
            action = (move_index, move_type)
            
            # Execute the move
            print(f"\nExecuting move: From position {from_pos} to {to_pos}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state
            board = env.unwrapped.game.board
            dice = env.unwrapped.dice
            borne_off_white = env.unwrapped.game.borne_off_white
            borne_off_black = env.unwrapped.game.borne_off_black
            
            if done:
                print_board(board, dice, borne_off_white, borne_off_black)
                print(f"\nGame over! Reward: {reward}")
                if reward > 0:
                    print("White wins!")
                elif reward < 0:
                    print("Black wins!")
                else:
                    print("It's a draw!")
            
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main() 