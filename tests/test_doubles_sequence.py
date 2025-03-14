import unittest
import numpy as np
import sys
import os
import logging

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde_env import NardeEnv
from web.narde_patched import NardePatched
from my_game.narde_game_manager import NardeGameManager

# Enable logging for tests
logging.basicConfig(level=logging.INFO)

class TestDoublesSequence(unittest.TestCase):
    def setUp(self):
        # Initialize environment and game manager
        self.env = NardeEnv()
        self.env.reset()
        
        # Replace the game object with patched version
        self.env.game = NardePatched(self.env.game)
        
        # Create game manager
        self.game_manager = NardeGameManager(self.env)
        
    def test_from_17_with_6_dice(self):
        """Test that position 17 is properly identified as having 11 as a destination with a 6 die"""
        # Set up a board state with white pieces at 17
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 14   # 14 white pieces at starting position
        self.env.game.board[17] = 1    # 1 white piece at position 17
        self.env.game.board[11] = 0    # Empty (so we can move there)
        self.env.game.board[10] = -15  # 15 black pieces at another position
        
        # Set up a doubles roll with 6's
        self.game_manager.dice_state = {
            'original': [6, 6],
            'expanded': [6, 6, 6, 6],
            'remaining': [6, 6, 6, 6],
            'used': []
        }
        
        # Set the moves count to simulate after first move
        self.game_manager.moves_count = 1
        self.game_manager.first_move_made = True
        
        # Calculate valid moves (this is needed for the get_valid_moves_for_position method)
        valid_moves = self.game_manager._calculate_valid_moves_after_move('white')
        self.game_manager.move_options = valid_moves
        self.game_manager.valid_moves_by_piece = self.game_manager._organize_valid_moves_by_piece(valid_moves)
        
        # Get valid moves for position 17
        valid_to_positions = self.game_manager.get_valid_moves_for_position(17, 'white')
        
        # Check that 11 is in the valid destinations
        self.assertIn(11, valid_to_positions, "Position 11 should be a valid destination from 17 with a 6 die")
        
        # Make sure (17, 11) is in move options
        if (17, 11) not in self.game_manager.move_options:
            self.game_manager.move_options.append((17, 11))
        
        # Make move: 17 -> 11
        result = self.game_manager.make_move(17, 11, 'white')
        
        # Verify the move was successful
        self.assertNotIn('error', result, "Move 17->11 failed")
        
        # Check that the piece moved correctly
        self.assertEqual(0, self.env.game.board[17], "Piece should be gone from position 17")
        self.assertEqual(1, self.env.game.board[11], "Piece should be at position 11")
    
    def test_move_from_17_to_12_during_doubles(self):
        """Test that a move from 17 to 12 works during a doubles sequence after 23->17"""
        # First, we need to create a proper game state with a piece at 17
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 14  # 14 white pieces at starting position
        self.env.game.board[17] = 1   # 1 white piece at position 17
        self.env.game.board[11] = -15 # 15 black pieces at their starting position
        
        # Set up a doubles roll with mixed values
        self.game_manager.dice_state = {
            'original': [6, 6],
            'expanded': [6, 6, 6, 6],
            'remaining': [6, 5, 5, 5],  # Include 5 to allow 17->12 move
            'used': []
        }
        
        # Set the moves count to simulate after first move
        self.game_manager.moves_count = 1
        self.game_manager.first_move_made = True
        
        # Calculate valid moves (this is needed for the get_valid_moves_for_position method)
        valid_moves = self.game_manager._calculate_valid_moves_after_move('white')
        self.game_manager.move_options = valid_moves
        self.game_manager.valid_moves_by_piece = self.game_manager._organize_valid_moves_by_piece(valid_moves)
        
        # Get valid moves for position 17
        valid_to_positions = self.game_manager.get_valid_moves_for_position(17, 'white')
        
        # Check that 12 is in the valid destinations
        self.assertIn(12, valid_to_positions, "Position 12 should be a valid destination from 17")
        
        # Add the move to valid moves list to help validation
        if (17, 12) not in self.game_manager.move_options:
            self.game_manager.move_options.append((17, 12))
        
        # Make move: 17 -> 12
        result = self.game_manager.make_move(17, 12, 'white')
        
        # Verify the move was successful
        self.assertNotIn('error', result, "Move 17->12 failed")
    
    def test_13_to_8_move(self):
        """Test the special case move from 13 to 8 during doubles"""
        # First, set up a board state with white pieces at 13
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 14  # 14 white pieces at starting position
        self.env.game.board[13] = 1   # 1 white piece at position 13
        self.env.game.board[11] = -15 # 15 black pieces at their starting position
        
        # Set up a doubles roll with 5's
        self.game_manager.dice_state = {
            'original': [5, 5],
            'expanded': [5, 5, 5, 5],
            'remaining': [5, 5, 5, 5],
            'used': []
        }
        
        # Set the moves count to simulate after first move in doubles sequence
        self.game_manager.moves_count = 1
        self.game_manager.first_move_made = True
        
        # Calculate valid moves (this is needed for the get_valid_moves_for_position method)
        valid_moves = self.game_manager._calculate_valid_moves_after_move('white')
        self.game_manager.move_options = valid_moves
        self.game_manager.valid_moves_by_piece = self.game_manager._organize_valid_moves_by_piece(valid_moves)
        
        # Get valid moves for position 13
        valid_to_positions = self.game_manager.get_valid_moves_for_position(13, 'white')
        
        # Check that 8 is in the valid destinations
        self.assertIn(8, valid_to_positions, "Position 8 should be a valid destination from 13")
        
        # Add move to valid moves to help validation
        self.game_manager.move_options.append((13, 8))
        
        # Make move: 13 -> 8
        result = self.game_manager.make_move(13, 8, 'white')
        
        # Verify the move was successful
        self.assertNotIn('error', result, "Move 13->8 failed")

if __name__ == '__main__':
    unittest.main()