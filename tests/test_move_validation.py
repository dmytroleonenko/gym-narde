import numpy as np
import unittest

from gym_narde.envs.narde import Narde

class TestMoveValidation(unittest.TestCase):
    def setUp(self):
        self.game = Narde()
        # For testing, set up a known board state
        # White has all checkers on point 24 (index 23) and Black on point 12 (index 11) by default
        # You can modify for other tests if needed

    def test_valid_moves_initial_white(self):
        # White's valid moves with a given roll
        roll = [3, 5]
        valid_moves = self.game.get_valid_moves(roll, current_player=1)
        # Expect some moves, at least one valid move
        self.assertTrue(len(valid_moves) > 0, "No valid moves for white initial state.")

    def test_validate_move_method(self):
        roll = [3, 5]
        # Get moves
        valid_moves = self.game.get_valid_moves(roll, current_player=1)
        if valid_moves:
            move = valid_moves[0]
            is_valid = self.game.validate_move(move, roll, current_player=1)
            self.assertTrue(is_valid)
        else:
            self.skipTest('No valid moves to test validate_move method')

if __name__ == '__main__':
    unittest.main()
