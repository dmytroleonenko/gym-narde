import numpy as np
import logging

def rotate_board(board):
    return np.concatenate((-board[12:], -board[:12])).astype(np.int32)

class NardePatched:
    def __init__(self, original_narde):
        self.original = original_narde
        self.board = original_narde.board
        self.borne_off_white = original_narde.borne_off_white
        self.borne_off_black = original_narde.borne_off_black
        self.first_turn_white = original_narde.first_turn_white
        self.first_turn_black = original_narde.first_turn_black
        
    # Add helper methods for move validation
    def _is_valid_move(self, from_pos, to_pos, current_player, board=None):
        """Check if a move is valid according to game rules"""
        if board is None:
            board = self.board.copy()
            
        # Use white's perspective for all logic
        is_white = (current_player == 'white' or current_player == 1)
        piece_value = 1 if is_white else -1
        direction = -1  # Always move counter-clockwise in Narde
            
        # Check source position has player's checker
        if (is_white and board[from_pos] <= 0) or (not is_white and board[from_pos] >= 0):
            return False
            
        # Calculate move distance
        move_distance = from_pos - to_pos
        if move_distance <= 0:  # Can only move counterclockwise (decreasing position)
            return False
            
        # Check destination is empty or has player's checkers
        if (is_white and board[to_pos] < 0) or (not is_white and board[to_pos] > 0):
            return False
            
        # Clone board and apply move to check for block rule violation
        board_after_move = board.copy()
        board_after_move[from_pos] -= piece_value
        board_after_move[to_pos] += piece_value
        
        # Check block rule
        if is_white:
            if self._violates_block_rule(board_after_move):
                return False
        else:
            # Need to check from black's perspective by rotating the board
            if self._violates_block_rule(rotate_board(board_after_move)):
                return False
                
        return True
    
    def _can_bear_off(self, current_player, board=None):
        """Check if a player can bear off"""
        if board is None:
            board = self.board.copy()
            
        is_white = (current_player == 'white' or current_player == 1)
        
        if is_white:
            # White can bear off if all checkers are in the home quadrant (0-5)
            return np.sum(np.maximum(board[6:], 0)) == 0
        else:
            # For black, we need to use rotated board
            rotated_board = rotate_board(board)
            return np.sum(np.maximum(rotated_board[6:], 0)) == 0
    
    def _valid_bear_off(self, from_pos, dice_value, current_player, board=None):
        """Check if a bearing off move is valid"""
        if board is None:
            board = self.board.copy()
            
        is_white = (current_player == 'white' or current_player == 1)
        
        # Can only bear off if all checkers are in the home quadrant
        if not self._can_bear_off(current_player, board):
            return False
            
        # Check dice value is enough to bear off from this position
        # In Narde, you need an exact or larger dice value to bear off
        if is_white:
            return dice_value >= from_pos + 1
        else:
            # For black, need to convert to rotated position
            rotated_pos = (from_pos + 12) % 24
            return dice_value >= rotated_pos + 1
    
    def _violates_block_rule(self, board):
        """
        Checks whether the board contains a contiguous block of 6 or more points
        that traps all opponent checkers.
        """
        # Pass to original implementation to avoid duplication
        return self.original._violates_block_rule(board)
        
    def get_perspective_board(self, current_player):
        logging.info(f"NardePatched.get_perspective_board: {current_player=}")
        if current_player == 1:  # Always white perspective
            logging.info(f"  Returning white perspective: {self.board.tolist()}")
            return self.board.copy()
        logging.info(f"  Returning black perspective (rotated): {rotate_board(self.board).tolist()}")
        return rotate_board(self.board)
    
    def execute_rotated_move(self, move, current_player):
        logging.info(f"NardePatched.execute_rotated_move: {move=}, {current_player=}")
        
        # Always execute as the original Narde implementation would
        from_pos, to_pos = move
        
        if current_player == 'white' or current_player == 1:
            # For white player, directly apply the move
            logging.info(f"  Executing white move directly: {move}")
            self._execute_move(from_pos, to_pos, is_white=True)
        elif current_player == 'black' or current_player == -1:
            # For black player, apply the move but update black pieces
            logging.info(f"  Executing black move: {move}")
            self._execute_move(from_pos, to_pos, is_white=False)
    
    def _execute_move(self, from_pos, to_pos, is_white=True):
        """Execute a move for the specified player color"""
        piece_value = 1 if is_white else -1
        player_name = "White" if is_white else "Black"
        
        logging.info(f"  _execute_move: {from_pos=}, {to_pos=}, {is_white=}")
        logging.info(f"  Board before: {self.board.tolist()}")
        
        # Validate positions are within board range
        if from_pos != 'off' and (from_pos < 0 or from_pos >= 24):
            logging.error(f"  Invalid from_pos: {from_pos}")
            return
            
        if to_pos != 'off' and (to_pos < 0 or to_pos >= 24):
            logging.error(f"  Invalid to_pos: {to_pos}")
            return
        
        # Check piece color and sign match
        if from_pos != 'off':
            if is_white and self.board[from_pos] <= 0:
                logging.error(f"  No white piece at position {from_pos}")
                return
            elif not is_white and self.board[from_pos] >= 0:
                logging.error(f"  No black piece at position {from_pos}")
                return
        
        # Bearing off
        if to_pos == 'off':
            if from_pos != 'off':
                self.board[from_pos] -= piece_value  # Decrease by +1 or -1 based on color
                if is_white:
                    self.borne_off_white += 1
                    logging.info(f"  White bearing off from {from_pos}, new borne_off_white={self.borne_off_white}")
                else:
                    self.borne_off_black += 1
                    logging.info(f"  Black bearing off from {from_pos}, new borne_off_black={self.borne_off_black}")
        else:
            # Regular move
            if from_pos != 'off' and to_pos >= 0 and to_pos < 24:
                self.board[from_pos] -= piece_value  # Decrease by +1 or -1 based on color
                self.board[to_pos] += piece_value    # Increase by +1 or -1 based on color
                logging.info(f"  {player_name} moving from {from_pos} to {to_pos}")
        
        logging.info(f"  Board after: {self.board.tolist()}")
        
        # Update the original too
        self.original.board = self.board
        self.original.borne_off_white = self.borne_off_white
        self.original.borne_off_black = self.borne_off_black
        
    def _execute_white_move(self, move):
        """Legacy method for compatibility"""
        from_pos, to_pos = move
        self._execute_move(from_pos, to_pos, is_white=True)

    def get_valid_moves(self, dice, current_player):
        logging.info(f"NardePatched.get_valid_moves: {dice=}, {current_player=}")
        
        # We need to handle each die separately to give more flexibility
        # Get valid moves for each die individually
        all_moves = []
        moves_by_die = {}
        
        for die in dice:
            # Get moves for this single die
            single_die_moves = self.original.get_valid_moves([die], current_player)
            moves_by_die[die] = single_die_moves
            all_moves.extend(single_die_moves)
        
        # Now get the standard moves from the original implementation
        standard_moves = self.original.get_valid_moves(dice, current_player)
        
        # Check if we found new valid moves by considering dice separately
        added_moves = [move for move in all_moves if move not in standard_moves]
        if added_moves:
            logging.info(f"  Added new valid moves by considering dice separately: {added_moves}")
            standard_moves.extend(added_moves)
        
        logging.info(f"  Valid moves from original: {standard_moves}")
        
        # For black player, we need to adjust the moves because we're dealing with
        # the black player's perspective from white's board
        if current_player == 'black' or current_player == -1:
            rotated_moves = []
            for move in standard_moves:
                from_pos, to_pos = move
                # Position 11 in white perspective is black's starting position
                if from_pos == 23:  # This is actually coming from white's perspective
                    from_pos = 11   # Change to black's starting position
                
                if to_pos == 'off':
                    rotated_move = (from_pos, to_pos)
                else:
                    # For black, we need to do point conversions
                    if to_pos >= 12:
                        to_pos = to_pos - 12
                    else:
                        to_pos = to_pos + 12
                        
                    # Ensure position is valid (0-23)
                    to_pos = to_pos % 24
                        
                rotated_moves.append((from_pos, to_pos))
            logging.info(f"  Rotated moves for black: {rotated_moves}")
            return rotated_moves
        
        return standard_moves