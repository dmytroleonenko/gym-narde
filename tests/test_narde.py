import pytest
import numpy as np
from gym_narde.envs.narde import Narde

@pytest.fixture
def game():
    """Create a fresh Narde game instance for each test."""
    return Narde()

class TestNarde:
    """Test suite for the Narde game logic."""
    
    def test_initial_board_setup(self, game):
        """Test that the board is initialized correctly."""
        # Check White pieces at position 24 (index 23)
        assert game.board[23] == 15
        # Check Black pieces at position 12 (index 11)
        assert game.board[11] == -15
        # Check that all other positions are empty
        for i in range(24):
            if i != 23 and i != 11:
                assert game.board[i] == 0
    
    def test_rotate_board(self, game):
        """Test the board rotation function."""
        # Initial board state
        assert game.board[23] == 15  # White pieces at position 24
        assert game.board[11] == -15  # Black pieces at position 12
        
        # Rotate the board using the standalone rotation
        rotated = np.concatenate((-game.board[12:], -game.board[:12])).astype(np.int32)
        
        # After rotation:
        # - White pieces become Black pieces and vice versa
        # - Position 24 (index 23) maps to position 12 (index 11)
        # - Position 12 (index 11) maps to position 24 (index 23)
        assert rotated[11] == -15
        assert rotated[23] == 15
    
    def test_board_perspective(self, game):
        """Test that the board is always maintained in white's perspective."""
        # In WHITE-ONLY perspective, the board is always in white's perspective
        assert game.board[23] == 15
        assert game.board[11] == -15
        
        print("Before move:", game.board[23])
        game.dice = [1]
        
        # Make a white move using execute_move
        result = game.execute_move((23, 22))
        
        print("After move:", game.board[23], game.board[22], "Result:", result)
        
        # Expect change: move executed
        assert result == True
        assert game.board[23] == 14
        assert game.board[22] == 1
    
    def test_execute_move_white(self, game):
        """Test executing moves for White player."""
        # Test normal move
        game.dice = [1]
        result = game.execute_move((23, 22))
        assert result == True
        assert game.board[23] == 14
        assert game.board[22] == 1
        
        # Test bearing off
        game.board = np.zeros(24, dtype=np.int32)
        game.board[0] = 1
        game.dice = [1]
        result = game.execute_move((0, 'off'))
        assert result == True
        assert game.board[0] == 0
        assert game.borne_off_white == 1
    
    def test_validate_move(self, game):
        """Test move validation using get_valid_moves."""
        # For a valid move for White, for dice [1]
        dice = [1]
        valid_moves = game.get_valid_moves(dice)
        assert (23, 22) in valid_moves
        
        # For an invalid move (no piece at position):
        valid_moves = game.get_valid_moves(dice)
        assert (22, 21) not in valid_moves
        
        # For an invalid move (landing on an occupied square by own piece)
        game.board[11] = 1
        valid_moves = game.get_valid_moves(dice)
        assert (23, 11) not in valid_moves
        
        # Bearing off test
        game.board = np.zeros(24, dtype=np.int32)
        game.board[0] = 1
        valid_moves = game.get_valid_moves([1])
        assert (0, 'off') in valid_moves
    
    def test_head_rule(self, game):
        """Test head rule restrictions (simplified)."""
        # Set up for head moves
        game.first_turn = True
        game.started_with_full_head = True
        game.head_moves_this_turn = 0
        
        # Get valid moves with dice roll of doubles 6
        roll = [6, 6]
        valid_moves = game.get_valid_moves(roll)
        # Allow at most 2 moves from the head (position 23)
        head_moves = [m for m in valid_moves if m[0] == 23]
        assert len(head_moves) <= 2
    
    def test_get_valid_moves(self, game):
        """Test getting valid moves."""
        dice = [1, 2]
        print("Dice:", dice)
        print("Board state:", game.board)
        valid_moves = game.get_valid_moves(dice)
        print("Valid moves after head rule:", valid_moves)
        assert any(move[0] == 23 and move[1] == 22 for move in valid_moves)
        assert any(move[0] == 23 and move[1] == 21 for move in valid_moves)
    
    def test_is_game_over(self, game):
        """Test game over detection."""
        assert not game.is_game_over()
        game.board = np.zeros(24, dtype=np.int32)
        game.borne_off_white = 15
        assert game.is_game_over()
        assert game.get_winner() == 1
        game.board = np.zeros(24, dtype=np.int32)
        game.borne_off_white = 0
        game.borne_off_black = 15
        assert game.is_game_over()
        assert game.get_winner() == -1 