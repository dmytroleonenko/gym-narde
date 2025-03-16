import numpy as np
import pytest
from gym_narde.envs.narde import Narde, rotate_board
from gym_narde.envs.narde_env import NardeEnv

class TestNardeEnv:
    def test_initial_board(self):
        """Verify initial board setup"""
        env = NardeEnv()
        assert env.game.board[23] == 15  # White's starting position
        assert env.game.borne_off_white == 0
        assert env.game.borne_off_black == 0

    def test_board_rotation(self):
        """Test board rotation mechanics"""
        # Original board with a checker at position 1 (index 0)
        test_board = np.zeros(24, dtype=np.int32)
        test_board[0] = 1  # White checker at position 1
        rotated = rotate_board(test_board)
        # Rotated board's index 12 (original 0's position) should be -1
        assert rotated[12] == -1
        # Position 24 (original index 23) rotated to first part's index 11
        assert rotate_board(Narde().board)[11] == -15  # White's start rotated to -15

    def test_black_perspective_obs(self):
        """Black's observation should show rotated board"""
        env = NardeEnv()
        env.current_player = -1
        obs = env._get_obs()[:24]
        # Black's starting point (original index 11) should be +15 in rotated view
        assert obs[23] == 15  # rotated's index 23 = original index 11 (Black's start)
        # White's starting point (original 23) should be -15 at rotated index 11
        assert obs[11] == -15

    def test_move_execution_perspective(self):
        """Check rotated board after move execution"""
        env = NardeEnv()
        env.current_player = 1
        # Simulate a valid move (e.g., move from White's starting position)
        env.game.execute_rotated_move((23, 19), 1)  # Move from 23 to 19
        env.current_player = -1  # Switch to Black's perspective
        obs = env._get_obs()[:24]
        # Check if the move is reflected correctly in Black's perspective
        assert obs[23] == 14  # White should have 14 checkers left at its starting point
        assert obs[19] == 1  # White should have 1 checker at position 19 in Black's perspective
        assert obs[11] == -15  # Black's starting point should remain unchanged

    def test_first_turn_rule(self):
        """Test first turn doubling rule"""
        env = NardeEnv()
        env.current_player = 1
        env.game.first_turn_white = True
        valid_moves = env.game.get_valid_moves([3, 3], 1)
        assert len(valid_moves) >= 2  # Should allow at least 2 moves for doubles on first turn

    def test_bear_off_condition(self):
        """Check bearing off logic after rotation"""
        env = NardeEnv()
        env.current_player = 1
        env.game.board[:6] = 15  # All White checkers in home
        valid_moves = env.game.get_valid_moves([1,1], 1)
        assert len(valid_moves) > 0  # Should have valid bearing off moves

    def test_observation_structure(self):
        """Ensure observation includes rotated board for Black"""
        env = NardeEnv()
        env.current_player = -1
        obs = env._get_obs()
        board_part = obs[:24]
        # Check Black's starting point (original index 11) is 15 in rotated view
        assert board_part[23] == 15

@pytest.mark.parametrize("position, expected_rotated_position, expected_rotated_value", [
    (23, 11, -15),  # White start rotated to -15 in first part
    (11, 23, 15),  # Black's start shows 15 in rotated view
    (0, 12, -1)  # Single checker rotation
])
def test_rotate_board_positions(position, expected_rotated_position, expected_rotated_value):
    board = np.zeros(24)
    board[position] = 15 if position == 23 else 1
    rotated = rotate_board(board)
    assert rotated[expected_rotated_position] == expected_rotated_value

def test_action_to_idx_conversion():
    move = ((23, 'off'), (0, 0))  # White bearing off move
    idx = NardeEnv.action_to_idx(move)
    assert idx == 576 * 23  # Verify encoding logic

def test_invalid_move_rotation():
    env = NardeEnv()
    # Simulate a move that violates the block rule
    env.game.board[10:16] = 6  # Create a block of 6 checkers
    env.current_player = 1
    valid_moves = env.game.get_valid_moves([1,1], 1)
    assert len(valid_moves) == 0  # No valid moves due to block rule violation

def test_observation_components():
    env = NardeEnv()
    obs, _ = env.reset()
    assert obs.shape == (28,)
    assert np.array_equal(obs[24:26], env.dice)  # dice part of observation
