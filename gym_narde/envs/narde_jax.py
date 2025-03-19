"""
JAX-compatible implementation of the Narde game logic.
This version replaces numpy with jax.numpy for potential acceleration and uses bfloat16 for memory efficiency.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import List, Tuple, Dict, Union, Any, Optional

# Enable bfloat16 computations in JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_dtype_bits", 16)

def rotate_board(board):
    """
    Flip the board to show the other player's perspective.
    This makes all checkers negative (flips White/Black roles).
    """
    # JAX version of rotating the board (making a new array)
    return -1 * jnp.flip(board)


class NardeJAX:
    """
    A JAX-compatible Narde game in "White-only perspective."
    After White's turn, we rotate so that Black also sees itself as 'White.'
    """

    def __init__(self):
        # 24 points on the board
        self.board = jnp.zeros(24, dtype=jnp.int32)
        # White starts with 15 on point 24 (index 23),
        # Black starts with 15 on point 12 (index 11), stored as negative.
        self.board = self.board.at[23].set(15)   # White
        self.board = self.board.at[11].set(-15)  # Black

        # Borne-off counters
        self.borne_off_white = 0
        self.borne_off_black = 0

        # Current player: +1 means "White" in the current perspective
        self.current_player = 1

        # Dice can be 2 or 4 values if doubles
        self.dice = []

        # Track how many head moves have been made *this turn*
        self.head_moves_this_turn = 0
        
        # For compatibility with tests
        self.first_turn = True
        self.started_with_full_head = True
        
        # Store the initial head count at the start of the turn
        self.head_count_at_turn_start = 15
        
        # Initialize the turn
        self.start_turn()

    def start_turn(self):
        """
        Reset per-turn counters and update turn-start state.
        This method should be called at the beginning of each turn.
        """
        # Reset head moves for the new turn
        self.head_moves_this_turn = 0
        
        # Store the head count at the start of the turn
        self.head_count_at_turn_start = int(self.board[23])

    def rotate_board_for_next_player(self):
        """
        Rotate the board so that the *other* player is now in White's seat.
        Also swap borne-off counts. Then flip current_player = -current_player.
        Reset any per-turn counters (like head_moves_this_turn).
        """
        # In JAX, we need to create a new array rather than modifying in-place
        self.board = rotate_board(self.board)
        self.borne_off_white, self.borne_off_black = self.borne_off_black, self.borne_off_white
        self.current_player = -self.current_player
        
        # Start a new turn for the next player
        self.start_turn()

    def is_game_over(self):
        """Return True if either side has borne off all 15 checkers."""
        return (self.borne_off_white == 15) or (self.borne_off_black == 15)

    def get_winner(self):
        """Return +1 if White won, -1 if Black won, else None if not finished."""
        if self.borne_off_white == 15:
            return 1
        elif self.borne_off_black == 15:
            return -1
        return None

    def _get_white_positions(self, board):
        """Return positions with white checkers (board > 0)."""
        # Simple implementation without JIT for better compatibility
        indices = []
        for i, val in enumerate(board):
            if val > 0:
                indices.append(i)
        return indices

    def _can_bear_off(self, board):
        """Check if White can bear off (all checkers in [0..5])."""
        # Simple implementation without JIT
        for i in range(6, 24):
            if board[i] > 0:
                return False
        return True

    def get_valid_moves(self, dice=None, head_moves_count=None):
        """
        Return all valid moves from the perspective of self.current_player==+1 (the 'White' seat).
        Moves are (from_pos, to_pos) or (from_pos, 'off') for bearing off.
        
        This is a pure function that doesn't modify any game state.
        
        Parameters:
        - dice: The dice values to use. If None, uses self.dice.
        - head_moves_count: The number of head moves already made this turn.
                           If None, uses self.head_moves_this_turn.
        
        Returns:
        - A list of valid moves as tuples (from_pos, to_pos) or (from_pos, 'off').
        """
        if dice is None:
            dice = self.dice
            
        if head_moves_count is None:
            head_moves_count = self.head_moves_this_turn

        if not dice:
            return []

        moves = []
        # Find all points with White checkers (board > 0)
        white_positions = self._get_white_positions(self.board)
        if len(white_positions) == 0:
            return []

        # Check if White can bear off (all checkers in [0..5])
        can_bear_off = self._can_bear_off(self.board)

        # First, determine if we can move from the head (position 23)
        # By default, 1 is allowed
        max_head_moves = 1

        # If we had 15 at the start *and* we rolled doubles in [3,4,6] on first turn, allow 2
        had_full_head_at_turn_start = (self.head_count_at_turn_start == 15)
        if had_full_head_at_turn_start and self.first_turn and self.started_with_full_head:
            # Are the dice double(3,4,6)?
            unique_vals = set(dice)
            if len(unique_vals) == 1 and len(dice) >= 2:
                val = list(unique_vals)[0]
                if val in [3, 4, 6]:
                    max_head_moves = 2

        # Check if we've already used up our head moves
        can_move_from_head = head_moves_count < max_head_moves

        # Loop through all dice and see if we can use them
        valid_moves = []
        for die in dice:
            # Loop through positions with White checkers
            for from_pos in white_positions:
                # Special head movement restriction
                if from_pos == 23 and not can_move_from_head:
                    continue

                # Calculate new destination
                to_pos = from_pos - die

                # Bearing off logic
                if can_bear_off and to_pos < 0:
                    # Only checkers on points [1..6] (indices 0..5) can use exact rolls
                    # to bear off, but point 1 (index 0) can bear off with any die
                    if from_pos == 0 or die == from_pos + 1:
                        valid_moves.append((from_pos, 'off'))
                    # If the die value is larger than needed, we can use it
                    # to bear off if we have no checkers on higher points
                    elif die > from_pos + 1:
                        # Check higher points for white checkers
                        has_higher_checkers = False
                        for i in range(from_pos + 1, 6):
                            if self.board[i] > 0:
                                has_higher_checkers = True
                                break
                        
                        if not has_higher_checkers:
                            valid_moves.append((from_pos, 'off'))
                    continue

                # Skip invalid destinations (off the board)
                if to_pos < 0:
                    continue

                # Check if destination is valid (empty or has White)
                if self.board[to_pos] >= 0:  # empty or has White
                    # Temporary board to check the move
                    temp_board = self.board.copy()
                    temp_board = temp_board.at[from_pos].add(-1)  # Remove from source
                    temp_board = temp_board.at[to_pos].add(1)     # Add to destination

                    # Check for block rule: 
                    # - Can't move if it creates 6 consecutive points with >=2 checkers
                    block_length = 0
                    block_found = False
                    for i in range(24):
                        if temp_board[i] >= 2:
                            block_length += 1
                            if block_length >= 6:
                                block_found = True
                                break
                        else:
                            block_length = 0

                    if not block_found:
                        valid_moves.append((from_pos, to_pos))

        # Sort them nicely
        def move_sort_key(m):
            f, t = m
            if t == 'off':
                return (f, 9999)
            else:
                return (f, t)
        valid_moves.sort(key=move_sort_key)

        return valid_moves

    def execute_move(self, move, dice=None):
        """
        Execute a single move (from_pos, to_pos or (from_pos, 'off')).
        Return True if successful, False otherwise.
        """
        if dice is None:
            dice = self.dice

        if not dice:
            return False

        # Validate the move using get_valid_moves (now safe to call as it's a pure function)
        valid_moves = self.get_valid_moves(dice)
        if move not in valid_moves:
            return False
            
        from_pos, to_pos = move
        
        # For bearing off
        if to_pos == 'off':
            # Calculate distance to edge
            distance = from_pos + 1
            
            # Find the appropriate die to use
            usable_dice = [d for d in dice if d >= distance]
            if usable_dice:
                die_used = min(usable_dice)
            else:
                die_used = min(dice)
                
            # Bear off
            self.board = self.board.at[from_pos].add(-1)
            self.borne_off_white += 1
        else:
            # For regular moves
            # Execute the move
            self.board = self.board.at[from_pos].add(-1)
            self.board = self.board.at[to_pos].add(1)
            
            # If moving off the head (point 24→index 23), track it
            if from_pos == 23:
                self.head_moves_this_turn += 1

            # Use the die
            die_used = abs(from_pos - to_pos)

        # Remove that die from dice
        if die_used in dice:
            dice.remove(die_used)

        return True 