'''
Nardy Rules - Long Nardy – Ultra-Short (for LLM):
    1. Setup: White’s 15 checkers on point 24; Black’s 15 on point 12.
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
    return np.concatenate((-board[12:], -board[:12])).astype(np.int32)


class Narde:
    def __init__(self):
        # Initialize for White perspective in Narde.
        self.board = np.zeros(24, dtype=np.int32)
        self.board[23] = 15  # White checkers on point 24 (index 23)
        self.board[11] = -15  # Black checkers on point 12 (index 11)
        self.borne_off_white = 0
        self.borne_off_black = 0
        self.first_turn_white = True
        self.first_turn_black = True

    def get_perspective_board(self, current_player):
        return self.board.copy()

    def execute_rotated_move(self, move, current_player):
        # Always execute the move as-is (board is kept in White's perspective).
        self._execute_move(move)

        # After executing, mark first turn as done.
        if current_player == 1:
            self.first_turn_white = False
        else:
            self.first_turn_black = False

    def get_valid_moves(self, roll, current_player=1):
        roll = sorted(roll, reverse=True)
        board = self.board  # Always use self.board (i.e., always in White's perspective)
        moves = []
        direction = -1  # Always move counter-clockwise

        for die in roll:
            for pos in range(24):
                if board[pos] <= 0:
                    continue  # Only current player's checkers (positive)

                new_pos = pos + direction * die
                if 0 <= new_pos < 24:
                    if board[new_pos] >= 0:  # Can only land on empty or own point
                        moves.append((pos, new_pos))
                elif new_pos < 0:
                    # Distinguish White vs. Black for bearing off:
                    if current_player == 1:
                        # White: must have all 15 in 0..5, check board[:6] for +15 total
                        if np.sum(np.maximum(board[:6], 0)) == 15:
                            if die >= pos + 1:  # distance from pos to off
                                moves.append((pos, 'off'))
                    else:
                        # Black: must have all 15 in indices 18..23, check board[18:24] for -15 total
                        if np.sum(np.minimum(board[18:24], 0)) == -15:
                            # distance from pos to off is (24 - pos)
                            if die >= (24 - pos):
                                moves.append((pos, 'off'))
        # --- NEW: Filter moves based on block rule, but allow escape if already in violation ---
        filtered_moves = []
        for move in moves:
            board_copy = board.copy()
            if move[1] == 'off':
                board_copy[move[0]] -= 1
            else:
                board_copy[move[0]] -= 1
                board_copy[move[1]] += 1

            already_violated = self._violates_block_rule(board)
            new_violated = self._violates_block_rule(board_copy)
            if not (new_violated and not already_violated):
                filtered_moves.append(move)
        moves = filtered_moves
        # --- End of new block ---
        first_turn = self.first_turn_white if current_player == 1 else self.first_turn_black
        return self._validate_head_moves(moves, roll, first_turn)

    def _validate_head_moves(self, moves, roll, first_turn):
        head_pos = 23  # Always current player's head

        # Head rule logic using current player's perspective
        # According to Rule 5: Only 1 checker may leave the head per turn.
        # With an exception on the first turn for doubles 3, 4, or 6.
        if first_turn and sorted(roll) in [[3,3], [4,4], [6,6]]:
            max_head_moves = 2  # Special case for first turn with specific doubles
        else:
            max_head_moves = 1  # Standard case: 1 checker from head per turn
            
        # Apply the head rule filtering
        return self._filter_head_moves(moves, head_pos, max_head_moves)

    def _execute_move(self, move):
        from_pos, to_pos = move
        if to_pos == 'off':
            # Bearing off: update borne_off counter and remove a checker.
            # For White: allow bearing off if the piece is from a home position (0–5)
            if from_pos >= 0 and from_pos <= 5:
                if self.board[from_pos] > 0:  # White's checker
                    self.board[from_pos] -= 1
                    self.borne_off_white += 1
            else:
                # Otherwise, treat it as Black's bearing off move (or add additional logic if needed)
                if self.board[from_pos] < 0:
                    self.board[from_pos] -= 1
                    self.borne_off_black += 1
        else:
            # Normal move: move one checker from `from_pos` to `to_pos`
            if self.board[from_pos] > 0:
                self.board[from_pos] -= 1
                self.board[to_pos] += 1
            else:
                self.board[from_pos] += 1
                self.board[to_pos] -= 1

    def _filter_head_moves(self, moves, head_pos, max_head_moves):
        allowed_moves = []
        head_moves_count = 0
        for move in moves:
            if move[0] == head_pos:
                if head_moves_count < max_head_moves:
                    allowed_moves.append(move)
                    head_moves_count += 1
            else:
                allowed_moves.append(move)
        return allowed_moves

    def _violates_block_rule(self, board):
        """
        Checks whether the board (in the current player's perspective)
        contains a contiguous block of 6 or more checkers (positive numbers)
        that traps all opponent checkers (i.e. no opponent checker—negative—
        is found ahead of the block), where 'ahead' means at any index lower than
        the first index of the block.
        
        In Narde, 'ahead' means positions that will be encountered in the direction
        of movement (counter-clockwise), which are positions with lower indices.
        """
        # Look for continuous blocks of the current player's checkers (positive values)
        i = 0
        while i < 24:
            # Find start of a block of current player's checkers
            if board[i] > 0:
                block_start = i
                block_end = i
                block_length = 1  # Count consecutive points, not total checkers
                
                # Find end of continuous block
                j = i + 1
                while j < 24 and board[j] > 0:
                    block_length += 1
                    block_end = j
                    j += 1
                    
                # Check if block is 6 or more consecutive points
                if block_length >= 6:
                    # 'Ahead' of the block means lower indices (0 to block_start-1)
                    # Check if any opponent checkers exist ahead of the block
                    has_opponent_ahead = False
                    for k in range(0, block_start):
                        if board[k] < 0:
                            has_opponent_ahead = True
                            break
                    
                    # If no opponent checkers ahead, this violates Rule 8
                    if not has_opponent_ahead:
                        return True
                        
                i = j  # Move past the end of this block
            else:
                i += 1  # No block at this position, move to next
                
        return False

    def validate_move(self, move, roll, current_player=1):
        """
        Validates if the given move is in the list of legal moves for a given roll and player.
        Returns True if valid, False otherwise.
        """
        valid_moves = self.get_valid_moves(roll, current_player)
        return move in valid_moves
