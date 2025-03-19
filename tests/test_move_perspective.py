import numpy as np
import pytest
from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv


class TestMovePerspective:
    """Test suite for move perspective handling in Narde"""
    
    def setup_environment(self):
        """Set up a clean environment for testing"""
        env = NardeEnv()
        env.reset()
        return env
    
    def test_board_updates_after_move(self):
        """Test that the board updates correctly after a move"""
        game = Narde()
        
        # White's starting position has 15 pieces at position 23
        assert game.board[23] == 15
        
        # Set up dice
        game.dice = [2]  # Move from 23 to 21
        
        # Execute a move
        game.execute_move((23, 21))
        
        # Check board updates
        assert game.board[23] == 14  # One piece moved from 23
        assert game.board[21] == 1   # One piece moved to 21
    
    def test_multiple_moves(self):
        """Test that multiple moves update the board correctly"""
        game = Narde()
        
        # First move: 23 -> 21
        game.dice = [2]
        valid_moves = game.get_valid_moves()
        assert (23, 21) in valid_moves, "Move 23->21 should be valid"
        
        game.execute_move((23, 21))
        assert game.board[23] == 14, "Source position should have one less piece"
        assert game.board[21] == 1, "Destination position should have one piece"
        
        # Get valid moves for second move
        game.dice = [4]
        valid_moves = game.get_valid_moves()
        
        # Find a valid move from the list
        if valid_moves:
            from_pos, to_pos = valid_moves[0]
            game.execute_move((from_pos, to_pos))
            
            # Verify board update
            if from_pos == 21:
                assert game.board[21] == 0, "Source position should be empty"
                assert game.board[to_pos] == 1, "Destination position should have one piece"
            else:
                assert game.board[from_pos] == (15 - 2 if from_pos == 23 else 0), "Source position should have one less piece"
                assert game.board[to_pos] == 1, "Destination position should have one piece"
            
        # Print final board state for debugging
        print(f"Final board state: {game.board}")
        
        # Generally verify integrity of the board
        # Count the absolute number of pieces (positive for white, negative for black)
        white_pieces = np.sum(game.board[game.board > 0])
        black_pieces = np.sum(-game.board[game.board < 0])
        
        assert white_pieces == 15, f"Should have 15 white pieces, found {white_pieces}"
        assert black_pieces == 15, f"Should have 15 black pieces, found {black_pieces}"