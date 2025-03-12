import numpy as np

def rotate_board(board):
    return np.concatenate((-board[12:], -board[:12])).astype(np.int32)

class Backgammon:
    def __init__(self):
        # Initialize for White perspective
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
            self.board = rotate_board(self.board)
        else:
            self._execute_move(move)
        
        # After executing, mark first turn as done for the current player.
        if current_player == 1:
            self.first_turn_white = False
        else:
            self.first_turn_black = False

    def get_valid_moves(self, roll):
        moves = []
        board = self.board  # Already in current player's perspective
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
                    if np.sum(self.board[6:]) == 0:  # All checkers in home quadrant
                        # For extra-large die, check if no lower-point checkers
                        if pos + die > 24 or np.sum(self.board[:pos]) == 0:
                            moves.append((pos, 'off'))
        return self._validate_head_moves(moves, roll, 
            self.first_turn_white if board[23] > 0 else self.first_turn_black)

    def _validate_head_moves(self, moves, roll, first_turn):
        head_pos = 23  # Always current player's head
        head_checkers = self.board[head_pos]

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
        # Placeholder for head move filtering logic
        return moves
