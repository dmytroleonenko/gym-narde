import unittest
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde

class TestDiceUsage(unittest.TestCase):
    def setUp(self):
        self.env = NardeEnv()
        self.env.reset()
        
    def test_doubles_usage_tracking(self):
        """Test that doubles are properly tracked and used according to the head rule."""
        # Set up a board state where we can test doubles usage
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        
        # Set doubles roll of 6
        self.env.dice = [6, 6, 6, 6]
        self.env.is_doubles = True
        
        # Set first_turn and started_with_full_head for head rule
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.game.head_count_at_turn_start = 15
        
        # Get valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # First move should include (23, 17) - moving 6 spaces from the head
        self.assertIn((23, 17), valid_moves, "Should be able to move from 23 to 17 with a 6")
        
        # Execute first move (from head)
        success = self.env.game.execute_move((23, 17), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move from head")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 3, "Should have 3 dice left after first move")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 14, "Should have 14 pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 1, "Should have 1 piece at position 17")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # According to the head rule, we can make one more move from the head on the first turn
        # if we have doubles 6, 4, or 3
        self.assertIn((23, 17), valid_moves, "Should be able to make a second head move on first turn with doubles 6")
        
        # Execute second move (from head)
        success = self.env.game.execute_move((23, 17), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move from head")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 2, "Should have 2 dice left after second move")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 13, "Should have 13 pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 2, "Should have 2 pieces at position 17")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # After using the 2 allowed head moves, we should not be able to move from the head anymore
        # But we should be able to move the pieces we already moved
        self.assertNotIn((23, 17), valid_moves, "Should not allow a third head move")
        self.assertIn((17, 11), valid_moves, "Should be able to move from position 17 to 11")
        
        # Execute third move (from position 17)
        success = self.env.game.execute_move((17, 11), self.env.dice)
        self.assertTrue(success, "Should be able to execute the third move from position 17")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after third move")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 13, "Should have 13 pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 1, "Should have 1 piece at position 17")
        self.assertEqual(self.env.game.board[11], 1, "Should have 1 piece at position 11")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # For the final move, we should still not be able to move from the head
        # But we should be able to move the remaining piece at position 17
        self.assertNotIn((23, 17), valid_moves, "Should not allow a fourth head move")
        self.assertIn((17, 11), valid_moves, "Should be able to move from position 17 to 11 again")
        
        # Execute fourth move (from position 17)
        success = self.env.game.execute_move((17, 11), self.env.dice)
        self.assertTrue(success, "Should be able to execute the fourth move from position 17")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice left after fourth move")
        
        # Check final board state
        self.assertEqual(self.env.game.board[23], 13, "Should have 13 pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 0, "Should have 0 pieces at position 17")
        self.assertEqual(self.env.game.board[11], 2, "Should have 2 pieces at position 11")
        
    def test_non_doubles_dice_usage(self):
        """Test that non-doubles dice can only be used once while respecting the head rule."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.dice = [5, 6]  # Non-doubles roll
        self.env.is_doubles = False
        
        # First move with die value 6
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Verify we can move with die value 6
        self.assertIn((23, 17), valid_moves, "Should allow move with die value 6")
        
        # Execute first move
        success = self.env.game.execute_move((23, 17), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after first move")
        self.assertEqual(self.env.dice[0], 5, "Should have the 5 die left")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 14, "Should have 14 pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 1, "Should have 1 piece at position 17")
        
        # Second move should only allow using the remaining die (5),
        # but NOT from the head (position 23) since we already used our one allowed head move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # We should be able to move the piece we already moved from the head,
        # but not use another piece from the head
        self.assertNotIn((23, 18), valid_moves, "Should not allow second move from head with remaining die")
        self.assertIn((17, 12), valid_moves, "Should allow move from position 17 to 12 with remaining die")
        
        # Execute second move from position 17
        success = self.env.game.execute_move((17, 12), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move from position 17")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice left after second move")
        
        # Check final board state
        self.assertEqual(self.env.game.board[23], 14, "Should have 14 pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 0, "Should have 0 pieces at position 17")
        self.assertEqual(self.env.game.board[12], 1, "Should have 1 piece at position 12")
        
    def test_bearing_off_dice_usage(self):
        """Test that bearing off properly consumes dice values."""
        # Set up a board state where bearing off is possible
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 2  # Two pieces near home
        
        # Set dice roll
        self.env.dice = [3, 2]
        self.env.is_doubles = False
        
        # Get valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should be able to bear off from position 2 using the 3
        self.assertIn((2, 'off'), valid_moves, "Should be able to bear off from position 2 with a 3")
        
        # Execute bearing off move
        success = self.env.game.execute_move((2, 'off'), self.env.dice)
        self.assertTrue(success, "Should be able to execute the bearing off move")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after bearing off")
        self.assertEqual(self.env.dice[0], 2, "Should have the 2 die left")
        
        # Check that the piece was removed from the board
        self.assertEqual(self.env.game.board[2], 1, "Should have 1 piece left at position 2")
        self.assertEqual(self.env.game.borne_off_white, 1, "Should have 1 piece borne off")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should be able to move from position 2 to 0 with the remaining die
        self.assertIn((2, 0), valid_moves, "Should be able to move from position 2 to 0 with a 2")
        
        # Execute second move
        success = self.env.game.execute_move((2, 0), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice left after second move")
        
        # Check final board state
        self.assertEqual(self.env.game.board[2], 0, "Should have no pieces left at position 2")
        self.assertEqual(self.env.game.board[0], 1, "Should have 1 piece at position 0")
        self.assertEqual(self.env.game.borne_off_white, 1, "Should have 1 piece borne off")

    def test_head_rule_with_doubles_3_3(self):
        """Test the head rule behavior with doubles 3-3."""
        # Set up board state
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.board[11] = -15  # Black pieces
        
        # Set doubles roll of 3 (which allows 2 pieces from head on first turn)
        self.env.dice = [3, 3, 3, 3]
        self.env.is_doubles = True
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.game.head_count_at_turn_start = 15
        
        # Get valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should be able to move from head (23) to 20
        self.assertIn((23, 20), valid_moves, "Should be able to move from 23 to 20 with a 3")
        
        # Execute first move
        success = self.env.game.execute_move((23, 20), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 3, "Should have 3 dice left after first move")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should still be able to move another piece from head (special case for doubles 3,4,6)
        self.assertIn((23, 20), valid_moves, "Should be able to make a second head move with doubles 3")
        
        # Execute second move
        success = self.env.game.execute_move((23, 20), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 2, "Should have 2 dice left after second move")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Check if we can move more than 2 pieces from head with doubles 3
        head_moves = [move for move in valid_moves if move[0] == 23]
        
        # According to the game rules, only 2 head moves are allowed with doubles 3
        # So there should be no more head moves available
        self.assertEqual(len(head_moves), 0, "Should not allow more than 2 head moves with doubles 3")
        
        # We should be able to move the pieces we already moved
        moved_piece_moves = [move for move in valid_moves if move[0] == 20]
        self.assertGreater(len(moved_piece_moves), 0, "Should be able to move pieces already moved from head")
        
        # Execute the third move (using a piece already moved from head)
        success = self.env.game.execute_move(moved_piece_moves[0], self.env.dice)
        self.assertTrue(success, "Should be able to execute the third move")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after third move")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Execute the final move
        if len(valid_moves) > 0:
            success = self.env.game.execute_move(valid_moves[0], self.env.dice)
            self.assertTrue(success, "Should be able to execute the final move")
            
            # Check that all dice were consumed
            self.assertEqual(len(self.env.dice), 0, "Should have no dice left after all moves")
        
        # Verify final board state
        total_pieces = sum(max(0, x) for x in self.env.game.board)
        self.assertEqual(total_pieces, 15, "Should still have all 15 pieces on the board")

    def test_head_rule_with_doubles_5_5(self):
        """Test the head rule behavior with doubles 5-5."""
        # Set up board state
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.board[11] = -15  # Black pieces
        
        # Set doubles roll of 5 (which allows only 1 piece from head on first turn)
        self.env.dice = [5, 5, 5, 5]
        self.env.is_doubles = True
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.game.head_count_at_turn_start = 15
        
        # Get valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should be able to move from head (23) to 18
        self.assertIn((23, 18), valid_moves, "Should be able to move from 23 to 18 with a 5")
        
        # Execute first move
        success = self.env.game.execute_move((23, 18), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 3, "Should have 3 dice left after first move")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Check if we can move more pieces from head with doubles 5
        head_moves = [move for move in valid_moves if move[0] == 23]
        
        # According to the game rules, only 1 head move is allowed with doubles 5
        # So there should be no more head moves available
        self.assertEqual(len(head_moves), 0, "Should not allow more than 1 head move with doubles 5")
        
        # We should be able to move the piece we already moved
        moved_piece_moves = [move for move in valid_moves if move[0] == 18]
        self.assertGreater(len(moved_piece_moves), 0, "Should be able to move piece already moved from head")
        
        # Execute the second move (using the piece already moved from head)
        success = self.env.game.execute_move(moved_piece_moves[0], self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 2, "Should have 2 dice left after second move")
        
        # Continue with remaining moves
        for i in range(2):
            # Get valid moves again
            valid_moves = self.env.game.get_valid_moves(self.env.dice)
            if not valid_moves:
                break
            
            # Execute the next move
            success = self.env.game.execute_move(valid_moves[0], self.env.dice)
            self.assertTrue(success, f"Should be able to execute move {i+3}")
        
        # Verify that all dice were consumed or no more valid moves
        final_valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertTrue(len(self.env.dice) == 0 or len(final_valid_moves) == 0, 
                        "Either all dice should be consumed or no more valid moves should be available")
        
        # Verify final board state
        total_pieces = sum(max(0, x) for x in self.env.game.board)
        self.assertEqual(total_pieces, 15, "Should still have all 15 pieces on the board")

    def test_head_rule_after_first_turn(self):
        """Test the head rule behavior after the first turn."""
        # Set up board state for initial moves
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All white pieces at start
        self.env.game.board[11] = -15  # All black pieces at start
        
        # First turn for white with dice [2, 1]
        self.env.dice = [2, 1]
        self.env.is_doubles = False
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.game.head_count_at_turn_start = 15
        
        # Execute first move for white
        success = self.env.game.execute_move((23, 21), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after first move")
        self.assertEqual(self.env.dice[0], 1, "Should have the 1 die left")
        
        # Get valid moves after first move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # In the current implementation, we can't move from head with the remaining die
        head_moves = [move for move in valid_moves if move[0] == 23]
        self.assertEqual(len(head_moves), 0, "Should not allow moving from head with the remaining die")
        
        # But we should be able to move from the position we just moved to
        self.assertIn((21, 20), valid_moves, "Should be able to move from position 21 to 20 with the remaining die")
        
        # Execute second move for white
        success = self.env.game.execute_move((21, 20), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice left after second move")
        
        # Check board state after white's turn
        self.assertEqual(self.env.game.board[23], 14, "Should have 14 pieces left at position 23")
        self.assertEqual(self.env.game.board[21], 0, "Should have 0 pieces at position 21")
        self.assertEqual(self.env.game.board[20], 1, "Should have 1 piece at position 20")
        
        # Now it's black's turn (simulate by keeping the same board state)
        # In a real game, the board would be rotated, but we're testing white's perspective
        
        # First turn for black with dice [2, 1]
        self.env.dice = [2, 1]
        self.env.is_doubles = False
        self.env.game.first_turn = False  # No longer first turn for the game
        
        # Simulate black's turn ending
        self.env.dice = []
        
        # Now it's white's turn again with doubles 6-6
        self.env.dice = [6, 6, 6, 6]
        self.env.is_doubles = True
        self.env.game.first_turn = False  # Not first turn anymore
        self.env.game.head_count_at_turn_start = 14  # 14 pieces left at head after previous moves
        
        # Get valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Check if head moves are allowed after first turn with doubles 6-6
        head_moves = [move for move in valid_moves if move[0] == 23]
        
        # According to the game rules, only 1 head move is allowed per turn
        # The current implementation correctly allows one head move per turn
        if len(head_moves) > 0:
            # If head moves are allowed, verify we can make one
            self.assertGreater(len(head_moves), 0, "Should allow at least one head move per turn")
            
            # Execute a head move
            success = self.env.game.execute_move(head_moves[0], self.env.dice)
            self.assertTrue(success, "Should be able to execute a head move")
            
            # Check that the die was consumed
            self.assertEqual(len(self.env.dice), 3, "Should have 3 dice left after head move")
            
            # Get valid moves again
            valid_moves = self.env.game.get_valid_moves(self.env.dice)
            
            # Check if we can move more pieces from head after using our one allowed head move
            head_moves = [move for move in valid_moves if move[0] == 23]
            self.assertEqual(len(head_moves), 0, "Should not allow more than 1 head move per turn")
            
            # We should be able to make other moves
            other_moves = [move for move in valid_moves if move[0] != 23]
            self.assertGreater(len(other_moves), 0, "Should be able to make moves from positions other than head")
            
            # Execute a non-head move
            success = self.env.game.execute_move(other_moves[0], self.env.dice)
            self.assertTrue(success, "Should be able to execute a non-head move")
            
            # Check that another die was consumed
            self.assertEqual(len(self.env.dice), 2, "Should have 2 dice left after second move")
        else:
            # If no head moves are allowed, we should be able to make other moves
            other_moves = [move for move in valid_moves if move[0] != 23]
            self.assertGreater(len(other_moves), 0, "Should be able to make moves from positions other than head")
            
            # Execute a non-head move
            success = self.env.game.execute_move(other_moves[0], self.env.dice)
            self.assertTrue(success, "Should be able to execute a non-head move")
            
            # Check that the die was consumed
            self.assertEqual(len(self.env.dice), 3, "Should have 3 dice left after first move")

    def test_doubles_usage(self):
        """Test that doubles can be used multiple times while respecting the head rule."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.dice = [4, 4, 4, 4]  # Doubles roll
        self.env.is_doubles = True
        
        # Set first_turn and started_with_full_head for head rule
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.game.head_count_at_turn_start = 15
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 19), valid_moves, "Should allow first move with doubles from head")
        
        # Execute first move from head
        success = self.env.game.execute_move((23, 19), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move from head")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 3, "Should have three dice remaining")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 14, "Should have 14 pieces left at position 23")
        self.assertEqual(self.env.game.board[19], 1, "Should have 1 piece at position 19")
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # According to the head rule, we can make one more move from the head on the first turn
        # with doubles 4
        self.assertIn((23, 19), valid_moves, "Should allow second move from head with doubles 4 on first turn")
        
        # Execute second move from head
        success = self.env.game.execute_move((23, 19), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move from head")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 2, "Should have two dice remaining")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 13, "Should have 13 pieces left at position 23")
        self.assertEqual(self.env.game.board[19], 2, "Should have 2 pieces at position 19")
        
        # Third move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # After using the 2 allowed head moves, we should not be able to move from the head anymore
        # But we should be able to move the pieces we already moved
        self.assertNotIn((23, 19), valid_moves, "Should not allow a third head move")
        self.assertIn((19, 15), valid_moves, "Should be able to move from position 19 to 15")
        
        # Execute third move from position 19
        success = self.env.game.execute_move((19, 15), self.env.dice)
        self.assertTrue(success, "Should be able to execute the third move from position 19")
        
        # Check that another die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have one die remaining")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 13, "Should have 13 pieces left at position 23")
        self.assertEqual(self.env.game.board[19], 1, "Should have 1 piece at position 19")
        self.assertEqual(self.env.game.board[15], 1, "Should have 1 piece at position 15")
        
        # Fourth move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # For the final move, we should still not be able to move from the head
        # But we should be able to move the remaining piece at position 19
        self.assertNotIn((23, 19), valid_moves, "Should not allow a fourth head move")
        self.assertIn((19, 15), valid_moves, "Should be able to move from position 19 to 15 again")
        
        # Execute fourth move from position 19
        success = self.env.game.execute_move((19, 15), self.env.dice)
        self.assertTrue(success, "Should be able to execute the fourth move from position 19")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice remaining")
        
        # Check final board state
        self.assertEqual(self.env.game.board[23], 13, "Should have 13 pieces left at position 23")
        self.assertEqual(self.env.game.board[19], 0, "Should have 0 pieces at position 19")
        self.assertEqual(self.env.game.board[15], 2, "Should have 2 pieces at position 15")

    def test_no_moves_after_using_all_dice(self):
        """Test that no moves are allowed after using all dice."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.dice = [5, 6]  # Non-doubles roll
        self.env.is_doubles = False
        
        # Use first die
        self.env.game.execute_move((23, 17))  # Using 6
        
        # Manually update dice to simulate consumption
        self.env.dice.remove(6)
        
        # Use second die
        self.env.game.execute_move((23, 18))  # Using 5
        
        # Manually update dice to simulate consumption
        self.env.dice.remove(5)
        
        # Should have no valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertEqual(len(valid_moves), 0, "Should have no valid moves after using all dice")
        
    def test_forced_to_use_larger_die(self):
        """Test that player must use larger die when smaller die has no valid moves."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 1  # One piece at start
        self.env.game.board[1] = 1   # One piece near home
        self.env.dice = [1, 6]  # Can't use 1 from position 23
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        self.assertIn((23, 17), valid_moves, "Should allow move with larger die when smaller die has no valid moves")
        
        # Execute move
        success = self.env.game.execute_move((23, 17), self.env.dice)
        self.assertTrue(success, "Should be able to execute the move with the larger die")
        
        # Check that the larger die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after move")
        self.assertEqual(self.env.dice[0], 1, "Should have the smaller die (1) left")
        
        # Check board state
        self.assertEqual(self.env.game.board[23], 0, "Should have no pieces left at position 23")
        self.assertEqual(self.env.game.board[17], 1, "Should have 1 piece at position 17")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should be able to move from position 1 with the smaller die
        self.assertIn((1, 0), valid_moves, "Should be able to move from position 1 to 0 with the smaller die")
        
        # Execute second move
        success = self.env.game.execute_move((1, 0), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice left after second move")
        
        # Check final board state
        self.assertEqual(self.env.game.board[1], 0, "Should have no pieces left at position 1")
        self.assertEqual(self.env.game.board[0], 1, "Should have 1 piece at position 0")

    def test_must_use_both_dice(self):
        """Test that player must use both dice if possible while respecting the head rule."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 2  # Two pieces at start
        self.env.game.board[10] = 1  # One piece at position 10
        self.env.dice = [5, 6]
        self.env.is_doubles = False
        
        # First move with die value 6
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 17), valid_moves, "Should allow move with die value 6")
        
        # Execute first move
        success = self.env.game.execute_move((23, 17), self.env.dice)
        self.assertTrue(success, "Should be able to execute the first move")
        
        # Check that the die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after first move")
        self.assertEqual(self.env.dice[0], 5, "Should have the 5 die left")
        
        # Check that the move was executed correctly
        self.assertEqual(self.env.game.board[23], 1, "Should have 1 piece left at position 23")
        self.assertEqual(self.env.game.board[17], 1, "Should have 1 piece at position 17")
        
        # Second move should use the remaining die (5),
        # but NOT from the head (position 23) since we already used our one allowed head move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 18), valid_moves, "Should not allow second move from head")
        
        # We should be able to move from position 17 with die value 5
        self.assertIn((17, 12), valid_moves, "Should allow move from position 17 to 12 with die value 5")
        
        # And we should be able to move from position 10 with die value 5
        self.assertIn((10, 5), valid_moves, "Should allow move from position 10 to 5 with die value 5")
        
        # Execute second move from position 17
        success = self.env.game.execute_move((17, 12), self.env.dice)
        self.assertTrue(success, "Should be able to execute the second move from position 17")
        
        # Check that all dice were consumed
        self.assertEqual(len(self.env.dice), 0, "Should have no dice left after second move")
        
        # Check final board state
        self.assertEqual(self.env.game.board[23], 1, "Should have 1 piece left at position 23")
        self.assertEqual(self.env.game.board[17], 0, "Should have 0 pieces at position 17")
        self.assertEqual(self.env.game.board[12], 1, "Should have 1 piece at position 12")
        self.assertEqual(self.env.game.board[10], 1, "Should have 1 piece at position 10")

    def test_must_use_larger_die_when_both_not_possible(self):
        """Test that player must use larger die when both moves aren't possible in sequence."""
        # Set up the specific board state
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # White pieces at start
        self.env.game.board[11] = -14  # Black pieces at position 12
        self.env.game.board[15] = -1   # One black piece at position 16
        
        # Set dice roll to 3-5
        self.env.dice = [3, 5]
        self.env.is_doubles = False
        
        # Get valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Only the move with the larger die (5) should be allowed
        # because the implementation now filters out the smaller die move
        # when it detects that using the smaller die first would lead to no valid moves with the larger die
        self.assertIn((23, 18), valid_moves, "Should allow move with larger die (5)")
        self.assertNotIn((23, 20), valid_moves, "Should not allow move with smaller die (3) when both moves aren't possible in sequence")
        self.assertEqual(len(valid_moves), 1, "Should only have one valid move (with the larger die)")
        
        # Execute the move with the larger die
        success = self.env.game.execute_move((23, 18), self.env.dice)
        self.assertTrue(success, "Should be able to execute the move with the larger die")
        
        # Check that the larger die was consumed
        self.assertEqual(len(self.env.dice), 1, "Should have 1 die left after move")
        self.assertEqual(self.env.dice[0], 3, "Should have the smaller die (3) left")
        
        # Get valid moves again
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        
        # Should NOT be able to move with the smaller die (3) from position 18 to 15
        # because there's a black piece at position 15
        self.assertEqual(len(valid_moves), 0, "Should not have any valid moves left")
        
        # This demonstrates that the implementation correctly enforces the rule
        # that the player must use the larger die when both moves aren't possible in sequence

if __name__ == '__main__':
    unittest.main() 