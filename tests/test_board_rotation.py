import numpy as np
import pytest
from gym_narde.envs.narde import Narde, rotate_board
from gym_narde.envs.narde_env import NardeEnv


class TestBoardRotation:
    """Test suite for board rotation and perspective handling in Narde"""
    
    def test_rotate_board_simple(self):
        """Test basic board rotation logic"""
        # Create a board with some checkers
        board = np.zeros(24, dtype=np.int32)
        board[23] = 5  # White checkers at 24
        board[20] = 3  # White checkers at 21
        board[11] = -8  # Black checkers at 12
        board[8] = -4  # Black checkers at 9
        
        rotated = rotate_board(board)
        
        # After rotation:
        # - White pieces should be negative
        # - Black pieces should be positive
        # - Index 23 should map to 11, 20 to 8, 11 to 23, 8 to 20
        assert rotated[11] == -5  # White from 23 -> 11
        assert rotated[8] == -3   # White from 20 -> 8
        assert rotated[23] == 8   # Black from 11 -> 23
        assert rotated[20] == 4   # Black from 8 -> 20
    
    def test_rotate_board_entire_board(self):
        """Test rotation of a full board setup"""
        # Create a board with checkers at every position
        board = np.zeros(24, dtype=np.int32)
        for i in range(12):
            board[i] = i + 1       # White checkers in first half
            board[i + 12] = -(i + 1)  # Black checkers in second half
            
        rotated = rotate_board(board)
        
        # Looking at the rotate_board implementation:
        # rotated = np.concatenate((-board[12:], -board[:12])).astype(np.int32)
        
        # First half of rotated (indices 0-11) should be -1 * board[12:24]
        for i in range(12):
            assert rotated[i] == -board[i + 12]
            
        # Second half of rotated (indices 12-23) should be -1 * board[0:12]
        for i in range(12):
            assert rotated[i + 12] == -board[i]
    
    def test_rotate_empty_positions(self):
        """Test rotation with empty positions"""
        # Create sparse board with only a few checkers
        board = np.zeros(24, dtype=np.int32)
        board[5] = 3   # White at pos 6
        board[18] = -2  # Black at pos 19
        
        rotated = rotate_board(board)
        
        # Verify specific positions
        assert rotated[17] == -3  # White from 5 -> 17
        assert rotated[6] == 2    # Black from 18 -> 6
        
        # Verify zeros stay zeros
        for i in range(24):
            if i != 17 and i != 6:
                assert rotated[i] == 0
    
    def test_double_rotation_identity(self):
        """Test that rotating a board twice returns the original board"""
        # Original board
        board = np.zeros(24, dtype=np.int32)
        board[23] = 15   # White start
        board[11] = -15  # Black start
        
        # Rotate twice
        rotated_once = rotate_board(board)
        rotated_twice = rotate_board(rotated_once)
        
        # Board should be back to the original state
        assert np.array_equal(board, rotated_twice)


class TestPlayerPerspective:
    """Test suite for player perspective handling"""
    
    def test_invalid_landing_on_opponent(self):
        """Test that moves landing on opponent's checkers are invalid"""
        game = Narde()
        
        # Place a Black checker at position 18 (index 17)
        game.board[17] = -1
        
        # Try to move White checker from 24 to 18
        valid_moves = game.get_valid_moves([6])
        assert (23, 17) not in valid_moves  # Cannot land on opponent's checker
    
    def test_observation_perspective(self):
        """Test that observations are correctly adjusted for player perspective"""
        env = NardeEnv()
        
        # White's perspective
        env.current_player = 1
        env.dice = [3, 4]
        env.game.borne_off_white = 2
        env.game.borne_off_black = 1
        
        obs_white = env._get_obs()
        # Board part should be identical to the actual board
        assert np.array_equal(obs_white[:24], env.game.board)
        # Borne off should be [white, black]
        assert np.array_equal(obs_white[25:27], [2, 1])


@pytest.mark.parametrize("dice_roll,positions,expected_valid_count", [
    ([1, 2], np.zeros(24), 0),  # Empty board, no valid moves
    ([3, 3], np.array([0]*23 + [15]), 2),  # First turn with doubles 3, allows 2 head moves
    ([6, 6], np.array([0]*23 + [15]), 2),  # First turn with doubles 6, allows 2 head moves
    ([5, 5], np.array([0]*23 + [15]), 2),  # First turn with doubles 5, allows 2 head moves
])
def test_valid_moves_with_dice(dice_roll, positions, expected_valid_count):
    """Test valid move generation with different dice rolls"""
    game = Narde()
    game.board = positions.copy()
    
    # Get valid moves
    valid_moves = game.get_valid_moves(dice_roll)
    
    # Check if count matches expectations
    assert len(valid_moves) == expected_valid_count

def test_board_state_consistency_after_multiple_moves():
    """Test that the board state remains consistent (15 checkers per player) after multiple moves."""
    game = Narde()
    
    # Initial state check
    white_count = np.sum(game.board[game.board > 0])
    black_count = np.abs(np.sum(game.board[game.board < 0]))
    assert white_count == 15, f"Initial white count should be 15, got {white_count}"
    assert black_count == 15, f"Initial black count should be 15, got {black_count}"
    
    # White's first move: 24 -> 20
    game.execute_move((23, 19))
    
    # Check board state after white's move
    white_count = np.sum(game.board[game.board > 0])
    black_count = np.abs(np.sum(game.board[game.board < 0]))
    total_white = white_count + game.borne_off_white
    total_black = black_count + game.borne_off_black
    assert total_white == 15, f"After white's first move, white count should be 15, got {total_white}"
    assert total_black == 15, f"After white's first move, black count should be 15, got {total_black}"
    
    # White's second move: 24 -> 18
    game.execute_move((23, 17))
    
    # Check board state after white's second move
    white_count = np.sum(game.board[game.board > 0])
    black_count = np.abs(np.sum(game.board[game.board < 0]))
    total_white = white_count + game.borne_off_white
    total_black = black_count + game.borne_off_black
    assert total_white == 15, f"After white's second move, white count should be 15, got {total_white}"
    assert total_black == 15, f"After white's second move, black count should be 15, got {total_black}"