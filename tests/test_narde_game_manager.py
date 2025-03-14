import unittest
import sys
import os
import numpy as np
from collections import defaultdict

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde import Narde
from gym_narde.envs.narde_env import NardeEnv
from web.narde_patched import NardePatched
from my_game.narde_game_manager import NardeGameManager, HEAD_POSITIONS

class TestNardeGameManager(unittest.TestCase):
    """Test case for the NardeGameManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.env = NardeEnv()
        self.env.reset()
        # Replace with patched version for better validation
        self.env.game = NardePatched(self.env.game)
        self.manager = NardeGameManager(self.env)
    
    def test_dice_management(self):
        """Test dice management and tracking."""
        # Test standard dice roll
        dice, _ = self.manager.roll_dice('white')
        self.assertEqual(len(dice), 2)
        self.assertEqual(len(self.manager.dice_state['original']), 2)
        self.assertEqual(len(self.manager.dice_state['remaining']), 2)
        self.assertEqual(len(self.manager.dice_state['used']), 0)
        
        # Force dice to be non-doubles for predictable testing
        self.manager.dice_state['original'] = [6, 5]
        self.manager.dice_state['expanded'] = [6, 5]
        self.manager.dice_state['remaining'] = [6, 5]
        self.manager.valid_moves = [(23, 17), (23, 18)]
        
        # Make a move that uses a die value of 6
        self.manager.make_move(23, 17, 'white')
        
        # Verify that dice state is updated correctly
        self.assertEqual(len(self.manager.dice_state['remaining']), 1)
        self.assertEqual(self.manager.dice_state['remaining'][0], 5)
        self.assertEqual(len(self.manager.dice_state['used']), 1)
        self.assertEqual(self.manager.dice_state['used'][0], 6)
    
    def test_basic_move_validation(self):
        """Test basic move validation."""
        # Set up dice and valid moves
        self.manager.dice_state['original'] = [6, 5]
        self.manager.dice_state['expanded'] = [6, 5]
        self.manager.dice_state['remaining'] = [6, 5]
        self.manager.valid_moves = [(23, 17), (23, 18)]
        
        # Test valid move
        result = self.manager.make_move(23, 17, 'white')
        self.assertNotIn('error', result)
        
        # Test invalid move (destination not in valid moves)
        invalid_result = self.manager.make_move(17, 11, 'white')
        self.assertIn('error', invalid_result)
        
        # Reset dice for new test
        self.manager.dice_state['original'] = [6, 5]
        self.manager.dice_state['expanded'] = [6, 5]
        self.manager.dice_state['remaining'] = [6, 5]
        self.manager.valid_moves = [(23, 17), (23, 18)]
        
        # Test move with non-existent piece
        invalid_result = self.manager.make_move(10, 5, 'white')
        self.assertIn('error', invalid_result)
        self.assertIn('piece', invalid_result['error'])
    
    def test_head_rule(self):
        """Test the head rule (only 1 piece can leave the head per turn)."""
        # Set up a standard dice roll
        self.manager.dice_state['original'] = [6, 5]
        self.manager.dice_state['expanded'] = [6, 5]
        self.manager.dice_state['remaining'] = [6, 5]
        self.manager.valid_moves = [(23, 17), (23, 18)]
        
        # Make a move from head position
        self.manager.make_move(HEAD_POSITIONS['white'], HEAD_POSITIONS['white'] - 6, 'white')
        
        # Setup for second move
        self.manager.valid_moves = [(23, 18)]
        self.manager.first_move_made = False
        
        # Try to make another move from head - should fail
        result = self.manager.make_move(HEAD_POSITIONS['white'], HEAD_POSITIONS['white'] - 5, 'white')
        self.assertIn('error', result)
        self.assertIn('head position', result['error'].lower())
    
    def test_special_doubles_head_rule(self):
        """Test the special case for doubles 3, 4, or 6 on first turn."""
        # Set up special doubles roll (6, 6)
        self.manager.dice_state['original'] = [6, 6]
        self.manager.dice_state['expanded'] = [6, 6, 6, 6]
        self.manager.dice_state['remaining'] = [6, 6, 6, 6]
        self.manager.valid_moves = [(23, 17)]
        
        # Set first turn flag
        self.manager.env.game.first_turn_white = True
        self.manager.is_first_turn_special_doubles = True
        self.manager.max_head_moves['white'] = 2
        
        # First move from head
        result1 = self.manager.make_move(HEAD_POSITIONS['white'], HEAD_POSITIONS['white'] - 6, 'white')
        self.assertNotIn('error', result1)
        
        # Setup for second move
        self.manager.valid_moves = [(23, 17)]
        self.manager.first_move_made = False
        
        # Second move from head should be allowed for special doubles
        result2 = self.manager.make_move(HEAD_POSITIONS['white'], HEAD_POSITIONS['white'] - 6, 'white')
        self.assertNotIn('error', result2)
        
        # Setup for third move
        self.manager.valid_moves = [(23, 17)]
        self.manager.first_move_made = False
        
        # Third move from head should NOT be allowed
        result3 = self.manager.make_move(HEAD_POSITIONS['white'], HEAD_POSITIONS['white'] - 6, 'white')
        self.assertIn('error', result3)
        self.assertIn('head position', result3['error'].lower())
    
    def test_dice_usage_after_move(self):
        """Test that dice are correctly tracked after moves."""
        # Set up dice
        self.manager.dice_state['original'] = [6, 3]
        self.manager.dice_state['expanded'] = [6, 3]
        self.manager.dice_state['remaining'] = [6, 3]
        self.manager.valid_moves = [(23, 17), (23, 20)]
        
        # Make a move using die value 6
        self.manager.make_move(23, 17, 'white')
        
        # Check that die value 6 was removed from remaining dice
        self.assertEqual(self.manager.dice_state['remaining'], [3])
        self.assertEqual(self.manager.dice_state['used'], [6])
        
        # Setup for second move
        self.manager.first_move_made = False
        self.manager.valid_moves = [(17, 14)]
        
        # Make another move using die value 3
        self.manager.make_move(17, 14, 'white')
        
        # Check that all dice are used
        self.assertEqual(self.manager.dice_state['remaining'], [])
        self.assertEqual(sorted(self.manager.dice_state['used']), [3, 6])
    
    def test_calculate_valid_moves_after_move(self):
        """Test calculation of valid moves after a move."""
        # Set up a board where there's a piece at position 17
        board = self.env.game.board.copy()
        board[:] = 0
        board[23] = 14  # 14 pieces at starting position
        board[17] = 1   # 1 piece at position 17
        self.env.game.board = board
        
        # Set up dice
        self.manager.dice_state['original'] = [6, 3]
        self.manager.dice_state['expanded'] = [6, 3]
        self.manager.dice_state['remaining'] = [3]  # Only the 3 value die remains
        
        # Add a valid move directly for testing - the game would normally generate this
        self.manager.valid_moves = [(17, 14)]
        
        # Calculate valid moves
        valid_moves = self.manager._calculate_valid_moves_after_move('white')
        
        # Check if valid moves include move from position 17 (where we have a piece)
        has_moves_from_position_17 = any(m[0] == 17 for m in valid_moves)
        self.assertTrue(has_moves_from_position_17)
    
    def test_enhance_valid_moves_for_doubles(self):
        """Test that valid moves are enhanced correctly for doubles."""
        # Setup a specific board state where 13->8 move should be valid
        board = self.env.game.board.copy()
        
        # Clear the board and place a white piece at position 13
        board[:] = 0
        board[13] = 1  # One white piece at position 13
        board[23] = 14  # Rest of white pieces at starting position
        self.env.game.board = board.copy()
        
        # Set up dice state with a 5
        self.manager.dice_state['original'] = [5, 5]
        self.manager.dice_state['expanded'] = [5, 5, 5, 5]
        self.manager.dice_state['remaining'] = [5]  # Only one die left
        self.manager.moves_count = 2  # This is the third move in sequence
        
        # Initially, valid_moves doesn't include the 13->8 move
        valid_moves = []
        
        # Call enhance_valid_moves_for_doubles
        enhanced_moves = self.manager._enhance_valid_moves_for_doubles(valid_moves, 'white')
        
        # Check if 13->8 was added
        self.assertTrue(any(m[0] == 13 and m[1] == 8 for m in enhanced_moves))
    
    def test_undo_moves(self):
        """Test the undo_moves functionality."""
        # Save original board state
        original_board = self.env.game.board.copy()
        
        # Set up a move
        self.manager.dice_state['original'] = [6, 5]
        self.manager.dice_state['expanded'] = [6, 5]
        self.manager.dice_state['remaining'] = [6, 5]
        self.manager.valid_moves = [(23, 17)]
        
        # Save state before making a move
        self.manager.saved_state = {
            'board': original_board.copy(),
            'borne_off_white': self.env.game.borne_off_white,
            'borne_off_black': self.env.game.borne_off_black
        }
        
        # Make a move
        self.manager.make_move(23, 17, 'white')
        
        # Verify board has changed
        self.assertFalse(np.array_equal(original_board, self.env.game.board))
        
        # Undo moves
        result = self.manager.undo_moves()
        
        # Verify board is restored
        self.assertTrue(np.array_equal(original_board, self.env.game.board))
    
    def test_is_game_over(self):
        """Test game over detection."""
        # Initially, game is not over
        is_over, winner = self.manager.is_game_over()
        self.assertFalse(is_over)
        self.assertIsNone(winner)
        
        # Set board so white has borne off all pieces
        self.env.game.borne_off_white = 15
        self.env.game.board[23] = 0
        
        # Check game is over and white wins
        is_over, winner = self.manager.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, 'white')
        
        # Reset and check for black winning
        self.env.game.borne_off_white = 0
        self.env.game.borne_off_black = 15
        
        # Check game is over and black wins
        is_over, winner = self.manager.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, 'black')


if __name__ == '__main__':
    unittest.main()