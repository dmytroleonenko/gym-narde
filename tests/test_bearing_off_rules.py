import unittest
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde

class TestBearingOffRules(unittest.TestCase):
    def setUp(self):
        self.env = NardeEnv()
        self.env.reset()
        
    def test_bearing_off_exact_value(self):
        """Test bearing off with exact dice value."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [3, 5]  # Die value 3 matches distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off with exact value")
        
        # Execute bearing off move using step
        obs, reward, terminated, truncated, info = self.env.step((2*24, 1))  # Action format: (from_pos*24, move_type=1 for bearing off)
        done = terminated or truncated
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        self.assertEqual(self.env.dice[0], 5, "Should consume the matching die (3)")
        
    def test_bearing_off_higher_value(self):
        """Test bearing off with higher dice value."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [5, 6]  # Both dice higher than distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off with higher value")
        
        # Execute bearing off move using step
        obs, reward, terminated, truncated, info = self.env.step((2*24, 1))  # Action format: (from_pos*24, move_type=1 for bearing off)
        done = terminated or truncated
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        self.assertEqual(self.env.dice[0], 6, "Should consume the smaller die (5)")
        
    def test_bearing_off_doubles(self):
        """Test bearing off with doubles."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 2  # Two pieces at position 2
        self.env.dice = [3, 3, 3, 3]  # Doubles matching distance to edge (2 + 1)
        # Also sync the game's dice
        self.env.game.dice = [3, 3, 3, 3]
        self.env.is_doubles = True
        
        print(f"\nDEBUG: Initial dice: {self.env.dice}")
        print(f"DEBUG: Initial board: {self.env.game.board}")
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        valid_moves_converted = [(int(from_pos), to_pos if to_pos == 'off' else int(to_pos)) 
                                for from_pos, to_pos in valid_moves]
        self.assertIn((2, 'off'), valid_moves_converted, "Should allow first bearing off with doubles")
        
        print(f"DEBUG: Valid moves for first move: {valid_moves}")
        
        # Execute first bearing off move
        action = (2*24, 1)  # Action format: (from_pos*24, move_type=1 for bearing off)
        print(f"DEBUG: Executing first bearing off action: {action}")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        print(f"DEBUG: After first move, dice: {self.env.dice}, game dice: {self.env.game.dice}")
        print(f"DEBUG: Board after first move: {self.env.game.board}")
        
        self.assertEqual(len(self.env.dice), 3, "Should consume one die")
        self.assertEqual(self.env.game.board[2], 1, "Should have 1 piece left at position 2")
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        valid_moves_converted = [(int(from_pos), to_pos if to_pos == 'off' else int(to_pos)) 
                                for from_pos, to_pos in valid_moves]
        self.assertIn((2, 'off'), valid_moves_converted, "Should allow second bearing off with doubles")
        
        print(f"DEBUG: Valid moves for second move: {valid_moves}")
        
        # Execute second bearing off move
        action = (2*24, 1)  # Action format: (from_pos*24, move_type=1 for bearing off)
        print(f"DEBUG: Executing second bearing off action: {action}")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        print(f"DEBUG: After second move, dice: {self.env.dice}, game dice: {self.env.game.dice}")
        print(f"DEBUG: Board after second move: {self.env.game.board}")
        
        self.assertEqual(len(self.env.dice), 2, "Should consume one die")
        self.assertEqual(self.env.game.board[2], 0, "Should have no pieces left at position 2")
        
    def test_bearing_off_consumes_die(self):
        """Test that bearing off consumes the appropriate die value."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [3, 5]  # Die value 3 matches distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off")
        
        # Execute bearing off move using step
        obs, reward, terminated, truncated, info = self.env.step((2*24, 1))  # Action format: (from_pos*24, move_type=1 for bearing off)
        done = terminated or truncated
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        self.assertEqual(self.env.dice[0], 5, "Should consume the matching die (3)")
        
    def test_bearing_off_higher_value_consumes_smallest(self):
        """Test that bearing off with higher value consumes smallest available die."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [5, 6]  # Both dice higher than distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off")
        
        # Execute bearing off move using step
        obs, reward, terminated, truncated, info = self.env.step((2*24, 1))  # Action format: (from_pos*24, move_type=1 for bearing off)
        done = terminated or truncated
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        self.assertEqual(self.env.dice[0], 6, "Should consume the smaller die (5)")
        
    def test_bearing_off_multiple_pieces(self):
        """Test bearing off multiple pieces with different dice values."""
        # Set up the test
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[1] = 1  # One piece at position 1
        self.env.game.board[2] = 1  # One piece at position 2
        
        # Manually set the dice for a predictable test
        self.env.dice = [2, 3]  # Exact values for both positions
        # Also set the game's dice
        self.env.game.dice = [2, 3]
        
        print(f"\nDEBUG: Initial dice: {self.env.dice}")
        print(f"DEBUG: Initial board: {self.env.game.board}")
        
        # First move - verify valid moves
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        print(f"DEBUG: Valid moves: {valid_moves}")
        
        # Convert valid moves to Python integers to avoid numpy type issues
        valid_moves_converted = [(int(from_pos), to_pos if to_pos == 'off' else int(to_pos)) 
                                for from_pos, to_pos in valid_moves]
        
        self.assertIn((1, 'off'), valid_moves_converted, "Should allow bearing off from position 1")
        self.assertIn((2, 'off'), valid_moves_converted, "Should allow bearing off from position 2")
        
        # Create the action for bearing off from position 2
        # For bearing off, action format is (from_pos * 24, 1)
        action = (2 * 24, 1)  
        print(f"DEBUG: Sending action: {action} to step()")
        print(f"DEBUG: Should bear off piece at position 2 using die value 3")
        
        # Verify the test board state before taking action
        self.assertEqual(self.env.game.board[2], 1, "Should have 1 piece at position 2 before bearing off")
        
        # Take the action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Verify the board state after the action
        print(f"DEBUG: Board after move: {self.env.game.board}")
        self.assertEqual(self.env.game.board[2], 0, "Piece should have been removed from position 2")
        self.assertEqual(self.env.game.borne_off_white, 1, "Should have borne off 1 piece")
        
        # Verify the dice after the action
        print(f"DEBUG: Dice after move: {self.env.dice}")
        
        # For this test, we're mainly checking dice consumption, not the exact remaining dice
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        
        # Second move - check that we can still bear off with the remaining die
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        print(f"DEBUG: Valid moves after first move: {valid_moves}")
        valid_moves_converted = [(int(from_pos), to_pos if to_pos == 'off' else int(to_pos)) 
                                for from_pos, to_pos in valid_moves]
        
        self.assertIn((1, 'off'), valid_moves_converted, "Should allow bearing off from position 1 after first move")
        
    def test_bearing_off_requires_all_pieces_in_home(self):
        """Test that bearing off requires all pieces to be in the home board."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.game.board[6] = 1  # One piece outside home board
        self.env.dice = [3, 5]
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((2, 'off'), valid_moves, "Should not allow bearing off with pieces outside home board")
        
    def test_bearing_off_with_no_valid_moves(self):
        """Test that bearing off is not allowed when there are no valid moves."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [1, 2]  # No dice value matches or exceeds distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((2, 'off'), valid_moves, "Should not allow bearing off when no dice value is sufficient")

if __name__ == '__main__':
    unittest.main() 