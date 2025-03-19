import unittest
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_narde.envs.narde_env import NardeEnv
from gym_narde.envs.narde import Narde

class TestHeadRule(unittest.TestCase):
    def setUp(self):
        self.env = NardeEnv()
        self.env.reset()
        
    def test_first_turn_doubles_3(self):
        """Test that doubles 3 on first turn allows two pieces from head."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.dice = [3, 3, 3, 3]
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 20), valid_moves, "Should allow first piece from head with doubles 3")
        self.env.game.execute_move((23, 20))
        self.env.dice.pop()
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 20), valid_moves, "Should allow second piece from head with doubles 3")
        self.env.game.execute_move((23, 20))
        self.env.dice.pop()
        
        # Manually set head_moves_this_turn to 2 since we've made two moves from the head
        self.env.game.head_moves_this_turn = 2
        
        # Manually update board state to reflect the moves made
        self.env.game.board[23] = 13  # 15 - 2 = 13 pieces left at head
        self.env.game.board[20] = 2   # 2 pieces moved to position 20
        
        # Third move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 20), valid_moves, "Should not allow third piece from head")
        self.assertIn((20, 17), valid_moves, "Should allow moving pieces already moved from head")
        
    def test_first_turn_doubles_4(self):
        """Test that doubles 4 on first turn allows two pieces from head."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.dice = [4, 4, 4, 4]
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 19), valid_moves, "Should allow first piece from head with doubles 4")
        self.env.game.execute_move((23, 19))
        self.env.dice.pop()
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 19), valid_moves, "Should allow second piece from head with doubles 4")
        self.env.game.execute_move((23, 19))
        self.env.dice.pop()
        
        # Manually set head_moves_this_turn to 2 since we've made two moves from the head
        self.env.game.head_moves_this_turn = 2
        
        # Manually update board state to reflect the moves made
        self.env.game.board[23] = 13  # 15 - 2 = 13 pieces left at head
        self.env.game.board[19] = 2   # 2 pieces moved to position 19
        
        # Third move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 19), valid_moves, "Should not allow third piece from head")
        self.assertIn((19, 15), valid_moves, "Should allow moving pieces already moved from head")
        
    def test_first_turn_doubles_6(self):
        """Test that doubles 6 on first turn allows two pieces from head."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.dice = [6, 6, 6, 6]
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 17), valid_moves, "Should allow first piece from head with doubles 6")
        self.env.game.execute_move((23, 17))
        self.env.dice.pop()
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 17), valid_moves, "Should allow second piece from head with doubles 6")
        self.env.game.execute_move((23, 17))
        self.env.dice.pop()
        
        # Manually set head_moves_this_turn to 2 since we've made two moves from the head
        self.env.game.head_moves_this_turn = 2
        
        # Manually update board state to reflect the moves made
        self.env.game.board[23] = 13  # 15 - 2 = 13 pieces left at head
        self.env.game.board[17] = 2   # 2 pieces moved to position 17
        
        # Third move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 17), valid_moves, "Should not allow third piece from head")
        self.assertIn((17, 11), valid_moves, "Should allow moving pieces already moved from head")
        
    def test_first_turn_doubles_2(self):
        """Test that doubles 2 on first turn only allows one piece from head."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = True
        self.env.dice = [2, 2, 2, 2]
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 21), valid_moves, "Should allow first piece from head with doubles 2")
        self.env.game.execute_move((23, 21))
        self.env.dice.pop()
        
        # Manually set head_moves_this_turn to 1 since we've made one move from the head
        self.env.game.head_moves_this_turn = 1
        
        # Manually update board state to reflect the moves made
        self.env.game.board[23] = 14  # 15 - 1 = 14 pieces left at head
        self.env.game.board[21] = 1   # 1 piece moved to position 21
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 21), valid_moves, "Should not allow second piece from head with doubles 2")
        self.assertIn((21, 19), valid_moves, "Should allow moving the piece already moved from head")
        
    def test_non_first_turn_doubles(self):
        """Test that doubles after first turn only allow one piece from head."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 15  # All pieces at start
        self.env.game.first_turn = False
        self.env.game.started_with_full_head = True
        self.env.dice = [6, 6, 6, 6]
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 17), valid_moves, "Should allow first piece from head")
        self.env.game.execute_move((23, 17))
        self.env.dice.pop()
        
        # Manually set head_moves_this_turn to 1 since we've made one move from the head
        self.env.game.head_moves_this_turn = 1
        
        # Manually update board state to reflect the moves made
        self.env.game.board[23] = 14  # 15 - 1 = 14 pieces left at head
        self.env.game.board[17] = 1   # 1 piece moved to position 17
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 17), valid_moves, "Should not allow second piece from head after first turn")
        self.assertIn((17, 11), valid_moves, "Should allow moving the piece already moved from head")
        
    def test_non_full_head_doubles(self):
        """Test that doubles with non-full head only allow one piece from head."""
        self.env.game.board = np.zeros(24, dtype=np.int32)
        self.env.game.board[23] = 14  # Not all pieces at start
        self.env.game.board[22] = 1   # One piece moved
        self.env.game.first_turn = True
        self.env.game.started_with_full_head = False
        self.env.dice = [6, 6, 6, 6]
        self.env.is_doubles = True
        
        # First move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertIn((23, 17), valid_moves, "Should allow first piece from head")
        self.env.game.execute_move((23, 17))
        self.env.dice.pop()
        
        # Manually set head_moves_this_turn to 1 since we've made one move from the head
        self.env.game.head_moves_this_turn = 1
        
        # Manually update board state to reflect the moves made
        self.env.game.board[23] = 13  # 14 - 1 = 13 pieces left at head
        self.env.game.board[17] = 1   # 1 piece moved to position 17
        
        # Second move
        valid_moves = self.env.game.get_valid_moves(self.env.dice)
        self.assertNotIn((23, 17), valid_moves, "Should not allow second piece from head with non-full head")
        self.assertIn((17, 11), valid_moves, "Should allow moving the piece already moved from head")
        self.assertIn((22, 16), valid_moves, "Should allow moving the piece not on head")

if __name__ == '__main__':
    unittest.main() 