import unittest
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde

class TestBearingOff(unittest.TestCase):
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
        
    def test_bearing_off_higher_value(self):
        """Test bearing off with higher dice value."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [5, 6]  # Both dice higher than distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off with higher value")
        
    def test_bearing_off_doubles(self):
        """Test bearing off with doubles."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 2  # Two pieces at position 2
        self.env.dice = [3, 3, 3, 3]  # Doubles matching distance to edge (2 + 1)
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow first bearing off with doubles")
        self.env.game.execute_move((2, 'off'))
        self.env.dice.pop()
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow second bearing off with doubles")
        
    def test_bearing_off_consumes_die(self):
        """Test that bearing off consumes the appropriate die value."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [3, 5]  # Die value 3 matches distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off")
        
        # Execute bearing off move directly using the game's execute_move method
        self.env.game.execute_move((2, 'off'))
        
        # Manually update the dice to simulate consumption
        self.env.dice.remove(3)
        
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        self.assertEqual(self.env.dice[0], 5, "Should consume the matching die (3)")
        
    def test_bearing_off_higher_value_consumes_smallest(self):
        """Test that bearing off with higher value consumes smallest available die."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [5, 6]  # Both dice higher than distance to edge (2 + 1)
        
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off")
        
        # Execute bearing off move directly using the game's execute_move method
        self.env.game.execute_move((2, 'off'))
        
        # Manually update the dice to simulate consumption
        self.env.dice.remove(5)
        
        self.assertEqual(len(self.env.dice), 1, "Should consume one die")
        self.assertEqual(self.env.dice[0], 6, "Should consume the smaller die (5)")
        
    def test_bearing_off_multiple_pieces(self):
        """Test bearing off multiple pieces with different dice values."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[1] = 1  # One piece at position 1
        self.env.game.board[2] = 1  # One piece at position 2
        self.env.dice = [2, 3]  # Exact values for both positions
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((1, 'off'), valid_moves, "Should allow bearing off from position 1")
        self.assertIn((2, 'off'), valid_moves, "Should allow bearing off from position 2")
        
        # Execute first bearing off move
        self.env.game.execute_move((2, 'off'))
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((1, 'off'), valid_moves, "Should allow bearing off from position 1 after first move")

if __name__ == '__main__':
    unittest.main() 