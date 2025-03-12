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
        if current_player == 1:  # Always white perspective
            return self.board.copy()
        return rotate_board(self.board)

    def execute_rotated_move(self, move, current_player):
        if current_player != 1:
            from_pos, to_pos = [(m + 12) % 24 for m in move]
            self._execute_move((from_pos, to_pos))
            # Removed board rotation to keep self.board in white perspective.
        else:
            self._execute_move(move)
        # After executing, mark first turn as done for the current player.
        if current_player == 1:
            self.first_turn_white = False
        else:
            self.first_turn_black = False

    def get_valid_moves(self, roll, current_player=1):
        roll = sorted(roll, reverse=True)
        board = self.board if current_player == 1 else rotate_board(self.board)
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
                    # Check if bearing off is possible
                    if np.sum(np.maximum(board[6:], 0)) == 0:  # All checkers are in the home quadrant
                        if die >= pos + 1:
                            moves.append((pos, 'off'))
        # --- NEW: Filter out moves that create an illegal block (Rule 8) ---
        filtered_moves = []
        for move in moves:
            board_copy = board.copy()
            if move[1] == 'off':
                board_copy[move[0]] -= 1
            else:
                board_copy[move[0]] -= 1
                board_copy[move[1]] += 1
            if not self._violates_block_rule(board_copy):
                filtered_moves.append(move)
        moves = filtered_moves
        # --- End of new block ---
        first_turn = self.first_turn_white if current_player == 1 else self.first_turn_black
        return self._validate_head_moves(moves, roll, first_turn)

    def _validate_head_moves(self, moves, roll, first_turn):
        head_pos = 23  # Always current player's head

        # Head rule logic using current player's perspective
        if first_turn and sorted(roll) in [[3,3], [4,4], [6,6]]:
            max_head_moves = 2
        else:
            max_head_moves = 1 if first_turn else 0

        return self._filter_head_moves(moves, head_pos, max_head_moves)

    def _execute_move(self, move):
        from_pos, to_pos = move
        if to_pos == 'off':
            # Bearing off: update borne_off counter and remove a checker.
            if self.board[from_pos] > 0:  # White
                self.board[from_pos] -= 1
                self.borne_off_white += 1
            else:  # Black (in rotated board, numbers are reversed)
                self.board[from_pos] += 1
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
        the first index of the block).
        """
        i = 0
        while i < 24:
            if board[i] > 0:
                block_start = i
                block_total = board[i]
                # Continue forward (i.e. increasing index) to find contiguous points with checkers.
                j = i + 1
                while j < 24 and board[j] > 0:
                    block_total += board[j]
                    j += 1
                if block_total >= 6:
                    # 'Ahead' of the block means indices 0 .. block_start-1.
                    # If no opponent checker (negative value) is found there, the move violates Rule 8.
                    if not np.any(board[:block_start] < 0):
                        return True
                i = j
            else:
                i += 1
        return False
