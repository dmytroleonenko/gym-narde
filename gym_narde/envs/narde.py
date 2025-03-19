'''
Nardy Rules - Long Nardy – Ultra-Short (for LLM):
    1. Setup: White's 15 checkers on point 24; Black's 15 on point 12.
    2. Movement: Both move checkers CCW into home (White 1–6, Black 13–18), then bear off.
    3. Starting: Each rolls 1 die; higher is White and goes first.
    4. Turns: Roll 2 dice, move checkers exactly by each value. No landing on opponent. If no moves exist, skip; if only one is possible, use the higher die.
    5. Head Rule: Only 1 checker may leave the head (White 24, Black 12) per turn. Exception on the first turn: if you roll double 6, 4, or 3, you can move 2 checkers from the head; after that, no more head moves.
    6. Bearing Off: Once all your checkers reach home, bear them off with exact or higher rolls.
    7. Ending/Scoring: Game ends when someone bears off all. If the loser has none off, winner scores 2 (mars); otherwise 1 (oin). Some events allow a last roll to tie.
    8. Block (Bridge): You cannot form a contiguous block of 6 checkers unless at least 1 opponent checker is still ahead of it. Fully trapping all 15 opponent checkers is banned—even a momentary 6‑block that would leave no opponent checkers in front is disallowed.
'''

import numpy as np


def rotate_board(board):
    """
    Rotate Black's perspective to White's and vice versa:
      - The first 12 points become the last 12 (negated),
      - The last 12 points become the first 12 (negated).
      - White's checkers become Black's (positive->negative) and vice versa.
    """
    rotated = np.concatenate((-board[12:], -board[:12])).astype(np.int32)
    return rotated


class Narde:
    """
    A simplified Long Nardy game in "White-only perspective."
    After White's turn, we rotate so that Black also sees itself as 'White.'
    """

    def __init__(self):
        # 24 points on the board
        self.board = np.zeros(24, dtype=np.int32)
        # White starts with 15 on point 24 (index 23),
        # Black starts with 15 on point 12 (index 11), stored as negative.
        self.board[23] = 15   # White
        self.board[11] = -15  # Black

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
        self.head_count_at_turn_start = self.board[23]

    def rotate_board_for_next_player(self):
        """
        Rotate the board so that the *other* player is now in White's seat.
        Also swap borne-off counts. Then flip current_player = -current_player.
        Reset any per-turn counters (like head_moves_this_turn).
        """
        self.board = rotate_board(self.board)
        self.borne_off_white, self.borne_off_black = self.borne_off_black, self.borne_off_white

        self.current_player *= -1
        
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
        white_positions = np.where(self.board > 0)[0]
        if len(white_positions) == 0:
            return []

        # Check if White can bear off (all checkers in [0..5])
        can_bear_off = (np.sum(self.board[6:] > 0) == 0)

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

        for d in dice:
            for from_pos in white_positions:
                # Skip head moves if we've already used our allowed moves
                if from_pos == 23 and not can_move_from_head:
                    continue
                    
                # Attempt a "regular move" = from_pos - d
                to_pos = from_pos - d
                if to_pos >= 0:
                    # Make sure no opponent checkers are blocking
                    if self.board[to_pos] >= 0:
                        # Potentially valid
                        moves.append((from_pos, to_pos))

                # If can bear off, see if from_pos < 6
                # Then a move of (from_pos, 'off') might be valid
                if can_bear_off and from_pos < 6:
                    # Two conditions: exact roll OR higher roll if no pieces behind
                    # 1) exact
                    if d == (from_pos + 1):
                        moves.append((from_pos, 'off'))
                    # 2) higher + no pieces behind
                    elif d > (from_pos + 1):
                        # Check if there's no checker behind from_pos
                        if np.sum(self.board[:from_pos] > 0) == 0:
                            moves.append((from_pos, 'off'))

        # Filter out any moves that cause an illegal 6-prime block that traps the opponent
        valid_moves = []
        for mv in moves:
            if not self._would_violate_block_rule(mv):
                valid_moves.append(mv)

        # Special case for non-doubles with exactly two possible moves
        if len(valid_moves) == 2 and len(dice) == 2 and dice[0] != dice[1]:
            smaller_die = min(dice)
            larger_die = max(dice)
            
            # Find the moves corresponding to each die
            smaller_die_moves = []
            larger_die_moves = []
            
            for move in valid_moves:
                from_pos, to_pos = move
                if to_pos == 'off':
                    # For bearing off, we need to check which die is being used
                    distance = from_pos + 1
                    if smaller_die >= distance and larger_die >= distance:
                        # If both dice can be used, the smaller one would be used
                        smaller_die_moves.append(move)
                    elif larger_die >= distance:
                        larger_die_moves.append(move)
                else:
                    # For regular moves
                    distance = abs(from_pos - to_pos)
                    if distance == smaller_die:
                        smaller_die_moves.append(move)
                    elif distance == larger_die:
                        larger_die_moves.append(move)
            
            # If we have exactly one move for each die
            if len(smaller_die_moves) == 1 and len(larger_die_moves) == 1:
                smaller_move = smaller_die_moves[0]
                
                # Simulate the smaller die move on a copy of the board
                board_copy = np.copy(self.board)
                borne_off_white_copy = self.borne_off_white
                
                # Apply the smaller move to the copy
                from_pos, to_pos = smaller_move
                board_copy[from_pos] -= 1
                if to_pos == 'off':
                    borne_off_white_copy += 1
                else:
                    board_copy[to_pos] += 1
                
                # Create a temporary game state to check for valid moves after the smaller move
                temp_game = Narde()
                temp_game.board = board_copy
                temp_game.borne_off_white = borne_off_white_copy
                temp_game.borne_off_black = self.borne_off_black
                temp_game.current_player = self.current_player
                temp_game.first_turn = self.first_turn
                temp_game.started_with_full_head = self.started_with_full_head
                temp_game.head_count_at_turn_start = self.head_count_at_turn_start
                temp_game.head_moves_this_turn = head_moves_count + (1 if from_pos == 23 else 0)
                
                # Check if there are valid moves with the larger die after the smaller die move
                remaining_dice = [larger_die]
                next_valid_moves = temp_game.get_valid_moves(remaining_dice)
                
                # If no valid moves remain after using the smaller die,
                # only return the move with the larger die
                if len(next_valid_moves) == 0:
                    return larger_die_moves

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
            self.board[from_pos] -= 1
            self.borne_off_white += 1
        else:
            # For regular moves
            # Execute the move
            self.board[from_pos] -= 1
            self.board[to_pos] += 1
            
            # If moving off the head (point 24→index 23), track it
            if from_pos == 23:
                self.head_moves_this_turn += 1

            # Use the die
            die_used = abs(from_pos - to_pos)

        # Remove that die from dice
        if die_used in dice:
            dice.remove(die_used)

        return True

    def _would_violate_block_rule(self, move):
        """
        Temporarily apply the move to the board, check if it creates
        an illegal 6-prime that fully traps all 15 opposing checkers,
        then revert. If it does, return True (violates the rule).
        """
        from_pos, to_pos = move

        # Save board
        saved = self.board.copy()
        saved_borne_white = self.borne_off_white
        saved_borne_black = self.borne_off_black

        # Simulate
        if to_pos == 'off':
            saved[from_pos] -= 1
            saved_borne_white += 1
        else:
            saved[from_pos] -= 1
            saved[to_pos] += 1

        # Check if we now have a contiguous 6-block that *completely* traps the opponent's 15 checkers.
        # Official rule: "Fully trapping all 15 opponent checkers is disallowed, even momentarily."
        if self._violates_block_rule(saved, saved_borne_white, saved_borne_black):
            return True
        return False

    def _violates_block_rule(self, board, bo_white, bo_black):
        """
        True if there's a 6+ prime that leaves no opponent checkers in front of it.
        Official Nardy rule #8: "You cannot form a contiguous block of 6 points if
        that block completely traps all 15 of the opponent's checkers."
        """
        # If the opponent is all borne off, no violation
        # or if you haven't pinned *all* of them behind the block
        # We do a simpler approach: we find any contiguous 6 points with >=2 checkers
        # If the total black checkers are behind that block with no black checkers ahead, it's illegal.

        # Count total black checkers on board
        black_on_board = -np.sum(board[board < 0])  # negative counts
        black_total = black_on_board + bo_black
        if black_total < 15:
            # Means White hasn't placed them properly or there's some mismatch, but let's not fail.
            black_total = 15

        # If black_on_board == 0, there's no block violation possible (all borne or behind the head).
        if black_on_board == 0:
            return False

        # Look for a 6+ prime of White checkers (board[i]>=2).
        i = 0
        while i < 24:
            if board[i] >= 2:
                length = 1
                j = i + 1
                while j < 24 and board[j] >= 2:
                    length += 1
                    j += 1
                # Now we have a block [i.. j-1] of length 'length'
                if length >= 6:
                    # If *all* black checkers remain behind that block with no black checkers ahead,
                    # that means black is fully trapped. By rule #8, that's illegal even momentarily.
                    # "Behind that block" = any black checkers on points < i
                    black_behind = -np.sum(board[:i][board[:i] < 0])
                    if black_behind == black_on_board:
                        # i.e. all black checkers are behind or on i, so they're trapped
                        return True
                i = j
            else:
                i += 1

        return False
        
    def print_board(self):
        """
        Print a human-readable representation of the board.
        Shows White checkers as positive numbers, Black as negative.
        """
        print("\nBoard State:")
        print("=" * 50)
        print("Index: Position | Count")
        print("-" * 50)
        for i in range(24):
            if self.board[i] != 0:
                color = "White" if self.board[i] > 0 else "Black"
                count = abs(self.board[i])
                print(f"{i:2d}: Point {24-i:2d} | {count:2d} {color} checker{'s' if count > 1 else ''}")
        print("-" * 50)
        print(f"White checkers borne off: {self.borne_off_white}")
        print(f"Black checkers borne off: {self.borne_off_black}")
        print("=" * 50)