import numpy as np

def rotate_board(board):
    rotated = np.zeros(24, dtype=np.int)
    for i in range(24):
        rotated[(i + 12) % 24] = -board[i]
    return rotated

class Backgammon:
    def __init__(self):
        # Initialize for White perspective
        self.board = np.zeros(24, dtype=np.int)
        self.board[23] = 15  # Starting position
        self.board[11] = -15  # Opponent starting position

    def get_perspective_board(self, current_player):
        if current_player == 1:  # Always white perspective
            return self.board.copy()
        return rotate_board(self.board)

    def execute_rotated_move(self, move, current_player):
        if current_player != 1:
            # Rotate move coordinates before execution
            from_pos, to_pos = [(m + 12) % 24 for m in move]
            self._execute_move((from_pos, to_pos))
            self.board = rotate_board(self.board)
        else:
            self._execute_move(move)

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
                    if board[new_pos] >= -1:  # Cannot land on opponent
                        moves.append((pos, new_pos))
        return moves

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
        # Placeholder for actual move execution logic
        pass

    def _filter_head_moves(self, moves, head_pos, max_head_moves):
        # Placeholder for head move filtering logic
        return moves
