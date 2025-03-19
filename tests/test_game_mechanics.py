#!/usr/bin/env python3
"""Test cases for Narde game mechanics."""

import gym_narde
import gymnasium as gym
import numpy as np
import pytest

from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde


def test_dice_mechanics():
    """Test that dice rolls correctly affect valid moves."""
    env = gym.make('Narde-v0')
    unwrapped = env.unwrapped
    
    # Reset the environment
    obs, info = env.reset()
    
    # Print board state
    print(f"Initial board: {unwrapped.game.board}")
    
    # Set specific dice values for testing
    test_dice = [3, 2]
    unwrapped.dice = test_dice
    print(f"Setting dice to: {unwrapped.dice}")
    
    # Check valid moves
    valid_moves = unwrapped.game.get_valid_moves()
    print(f"Valid moves with dice {unwrapped.dice}: {valid_moves}")
    
    # In Narde, the logic is from_pos - dice_value = to_pos
    # Let's verify this logic is working
    verified_moves = []
    for from_pos, to_pos in valid_moves:
        if to_pos != 'off':  # Skip bearing off moves
            # For each move, check if from_pos - to_pos equals one of our dice values
            distance = from_pos - to_pos
            print(f"Move: {from_pos} -> {to_pos}, distance: {distance}")
            
            # Check if this distance can be created with our dice
            # The game often uses combinations of dice, so we need to check
            # both individual dice and sums
            if distance in test_dice:
                print(f"  This matches dice value: {distance}")
                verified_moves.append((from_pos, to_pos, distance))
            elif distance == sum(test_dice):
                print(f"  This matches sum of dice: {distance}")
                verified_moves.append((from_pos, to_pos, distance))
    
    # We should have verified at least one move
    print(f"Verified moves: {verified_moves}")
    assert len(verified_moves) > 0, "No moves match our expected dice mechanics"


def test_bearing_off():
    """Test the bearing off mechanics."""
    env = gym.make('Narde-v0')
    unwrapped = env.unwrapped
    
    # Reset the environment to ensure proper initialization
    obs, info = env.reset()
    
    # Set up a position where bearing off is possible - all pieces in home board
    unwrapped.game.board = np.zeros(24, dtype=np.int32)
    # Put pieces in the home board (positions 0-5)
    unwrapped.game.board[0] = 1  # Position 1
    unwrapped.game.board[1] = 1  # Position 2
    
    # Make sure all state is consistent with this board setup
    unwrapped.game.borne_off_white = 13  # 13 pieces already borne off, 2 remain
    unwrapped.game.borne_off_black = 0
    
    # Set specific dice
    unwrapped.dice = [1, 2]
    
    # Print current board state for debugging
    print(f"Board state: {unwrapped.game.board}")
    print(f"White pieces borne off: {unwrapped.game.borne_off_white}")
    print(f"Dice: {unwrapped.dice}")
    
    # The game needs to know all pieces are in the home board
    # Make sure the logic for "can bear off" is properly set
    print(f"Can bear off check: {np.sum(unwrapped.game.board[6:] > 0) == 0}")
    
    # Check valid moves
    valid_moves = unwrapped.game.get_valid_moves()
    print(f"Valid moves: {valid_moves}")
    
    # Check if bearing off moves are available
    bearing_off_moves = [(from_pos, to_pos) for from_pos, to_pos in valid_moves if to_pos == 'off']
    print(f"Bearing off moves: {bearing_off_moves}")
    
    # We should have at least one bearing off move
    assert len(bearing_off_moves) > 0, "No bearing off moves available"
    
    # Execute a bearing off move if available
    if bearing_off_moves:
        from_pos, to_pos = bearing_off_moves[0]
        action_idx = from_pos * 24
        action = (action_idx, 1)  # 1 indicates bearing off
        
        # Step with the bearing off move
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Check that the piece was borne off
        print(f"After bearing off, white pieces borne off: {unwrapped.game.borne_off_white}")
        assert unwrapped.game.borne_off_white > 13, "Piece was not borne off"
        
        # Print the new board state
        print(f"Board after bearing off: {unwrapped.game.board}")
    else:
        print("No bearing off moves available to execute")


def test_win_condition():
    """Test that the game ends correctly when a player wins."""
    env = gym.make('Narde-v0')
    unwrapped = env.unwrapped
    
    # Reset the environment first
    obs, info = env.reset()
    
    # Set up a position where White is about to win
    unwrapped.game.board = np.zeros(24, dtype=np.int32)
    unwrapped.game.board[0] = 1  # Last white piece at position 1 (0-indexed)
    unwrapped.game.borne_off_white = 14  # Already borne off 14 pieces
    unwrapped.game.borne_off_black = 0
    
    # Print the current board state for debugging
    print(f"Board state: {unwrapped.game.board}")
    print(f"White pieces borne off: {unwrapped.game.borne_off_white}")
    print(f"Can bear off check: {np.sum(unwrapped.game.board[6:] > 0) == 0}")
    
    # Set dice to allow bearing off
    unwrapped.dice = [1]
    print(f"Dice: {unwrapped.dice}")
    
    # Check valid moves
    valid_moves = unwrapped.game.get_valid_moves()
    print(f"Valid moves: {valid_moves}")
    
    # Bearing off moves
    bearing_off_moves = [(from_pos, to_pos) for from_pos, to_pos in valid_moves if to_pos == 'off']
    print(f"Bearing off moves: {bearing_off_moves}")
    
    # There should be a bearing off move
    assert len(bearing_off_moves) > 0, "Cannot bear off the last piece"
    
    # Execute the winning move
    from_pos, to_pos = bearing_off_moves[0]
    action_idx = from_pos * 24
    action = (action_idx, 1)  # 1 indicates bearing off
    
    # Step with the winning move
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Check results of the action
    print(f"After action - Reward: {reward}, Done: {done}, Borne off white: {unwrapped.game.borne_off_white}")
    
    # Check that the game ended
    assert done, "Game should end when all pieces are borne off"
    
    # Check that the reward is positive (win)
    assert reward > 0, "Winning should yield a positive reward"


def test_block_rule():
    """Test the block rule (can't form a 6-checker contiguous block)."""
    env = gym.make('Narde-v0')
    unwrapped = env.unwrapped
    
    # Reset first
    obs, info = env.reset()
    
    # Set up a position where a block could be formed
    unwrapped.game.board = np.zeros(24, dtype=np.int32)
    
    # Set up 5 consecutive checkers (positions 5-9)
    for i in range(5, 10):
        unwrapped.game.board[i] = 1
    
    # Position to potentially complete the block
    unwrapped.game.board[10] = 1  # A sixth consecutive checker would form a block
    unwrapped.game.board[4] = 0   # This position is empty
    unwrapped.game.board[15] = 1  # Piece that could complete the block by moving to position 4
    
    # Set opponent piece beyond the potential block
    unwrapped.game.board[3] = -1  # Opponent piece is "ahead" of where the block would form
    
    # Print the board for debugging
    print(f"Board setup for block rule test:")
    print(f"Board: {unwrapped.game.board}")
    
    # Set specific dice to allow the move that would form a block
    # We need dice value 11 to move from position 15 to position 4
    unwrapped.dice = [11]
    print(f"Dice: {unwrapped.dice}")
    
    # Check valid moves
    valid_moves = unwrapped.game.get_valid_moves()
    print(f"Valid moves: {valid_moves}")
    
    # Special check for understanding the block rule
    for i in range(24):
        if unwrapped.game.board[i] > 0:  # For each white piece
            for d in unwrapped.dice:
                to_pos = i - d
                if to_pos >= 0:  # If within board
                    move = (i, to_pos)
                    # Check if move would form a block
                    print(f"Move {move}: Would form block? {unwrapped.game._would_violate_block_rule(move)}")
    
    # The move from 15 to 4 should not be valid due to block rule
    block_forming_move = (15, 4)
    assert block_forming_move not in valid_moves, "Block-forming move should not be valid"


if __name__ == "__main__":
    print("Testing dice mechanics...")
    test_dice_mechanics()
    
    print("\nTesting bearing off...")
    test_bearing_off()
    
    print("\nTesting win condition...")
    test_win_condition()
    
    print("\nTesting block rule...")
    test_block_rule()
    
    print("\nAll game mechanics tests passed!") 