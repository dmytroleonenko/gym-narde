import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_narde.envs.narde import Narde

class NardeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        super(NardeEnv, self).__init__()

        self.game = Narde()
        self.current_player = 1  # Always white perspective
        
        self.render_mode = render_mode

        # New observation space: 24 (board) + 2 (dice) + 2 (borne_off) = 28 features
        # Define observation space components
        board_low = np.full(24, -15, dtype=np.float32)
        dice_low = np.zeros(2, dtype=np.float32)
        borne_off_low = np.zeros(2, dtype=np.float32)
        full_low = np.concatenate([board_low, dice_low, borne_off_low])
        
        board_high = np.full(24, 15, dtype=np.float32)
        dice_high = np.full(2, 6, dtype=np.float32)
        borne_off_high = np.full(2, 15, dtype=np.float32)
        full_high = np.concatenate([board_high, dice_high, borne_off_high], axis=0)
        
    
    def _get_obs(self):
        # Get board in current player's perspective
        board = self.game.get_perspective_board(self.current_player).astype(np.float32)
        
        # dice values (2 values)
        dice_array = np.array(self.dice, dtype=np.float32)
        
        # Borme_off counts (2 values)
        if self.current_player == 1:
            borne_off = np.array([self.game.borne_off_white, self.game.borne_off_black], dtype=np.float32)
        else:
            # For Black, invert the counts
            borne_off = np.array([self.game.borne_off_black, self.game.borne_off_white], dtype=np.float32)
        
        return np.concatenate([board, dice_array, borne_off]).astype(np.float32)
    def step(self, action):
        # Roll dice and store in self.dice
        dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
        self.dice = dice  # Store dice for observation computation
        # Roll two dice at the beginning of the turn
        dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
        
        valid_moves = self.game.get_valid_moves(dice, self.current_player)

        if len(valid_moves) == 0:
            # No legal moves; skip turn.
            done, reward = self._check_game_ended()
            if not done:
                self.current_player *= -1
            # New API: return observation, reward, terminated, truncated, info
            return self._get_obs(), reward, done, False, {}

        elif len(valid_moves) == 1:
            # Only one legal move—that is, using the higher die.
            self.game.execute_rotated_move(valid_moves[0], self.current_player)
        else:
            move1_code, move2_code = action
            from_pos1 = move1_code // 24
            to_pos1 = move1_code % 24
            
            # Check if this is a bearing off move (to_pos == 0 and from_pos in 0-5)
            if to_pos1 == 0 and 0 <= from_pos1 <= 5:
                move1 = (from_pos1, 'off')
            else:
                move1 = (from_pos1, to_pos1)
            
            from_pos2 = move2_code // 24
            to_pos2 = move2_code % 24
            
            # Check if second move is a bearing off move
            if to_pos2 == 0 and 0 <= from_pos2 <= 5:
                move2 = (from_pos2, 'off')
            else:
                move2 = (from_pos2, to_pos2)
            if move1 in valid_moves:
                self.game.execute_rotated_move(move1, self.current_player)
                # Recalculate valid moves for the remaining die after move1 is executed.
                # Try to determine which die was used for the first move
                try:
                    # Calculate the move distance
                    if move1[1] == 'off':
                        # Special case for bearing off
                        # For bearing off, we need a die >= (point_position + 1)
                        move_distance = move1[0] + 1
                    else:
                        move_distance = abs(move1[0] - move1[1])
                    
                    # Find matching die or closest die
                    temp_dice = dice.copy()
                    if move_distance in temp_dice:
                        temp_dice.remove(move_distance)
                    else:
                        # If no exact match, remove the first die
                        if temp_dice:
                            temp_dice.pop(0)
                    
                    # Check if there are any remaining dice
                    if temp_dice:
                        # Get valid moves for the remaining dice
                        new_valid_moves = self.game.get_valid_moves(temp_dice, self.current_player)
                        if move2 in new_valid_moves:
                            self.game.execute_rotated_move(move2, self.current_player)
                except Exception as e:
                    # If any exception occurs, just continue without executing the second move
                    pass

        # Check if game ended and compute reward
        done, reward = self._check_game_ended()

        # Switch players if game continues
        if not done:
            self.current_player *= -1
            
        # New API: return observation, reward, terminated, truncated, info
        return self._get_obs(), reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        # Initialize RNG if seed is provided
        if seed is not None:
            np.random.seed(seed)
            
        self.game = Narde()
        while True:
            white_roll = np.random.randint(1, 7)
            black_roll = np.random.randint(1, 7)
            if white_roll != black_roll:
                break
        # The winning player (strictly higher roll) becomes White; otherwise Black.
        self.current_player = 1 if white_roll > black_roll else -1
        
        # Return observation and info dict according to new API
        self.dice = [0, 0]  # Initialize to default
            
        self.game = Narde()
        while True:
            white_roll = np.random.randint(1, 7)
            black_roll = np.random.randint(1, 7)
            if white_roll != black_roll:
                break
        # The winning player (strictly higher roll) becomes White; otherwise Black.
        self.current_player = 1 if white_roll > black_roll else -1
        
        # Return observation and info dict according to new API
        self.dice = [0, 0]  # Initialize to default
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            board_str = ""
            for i in range(24):
                board_str += f"{self.game.board[i]:>3} "
                if (i + 1) % 6 == 0:
                    board_str += "\n"
            print(board_str)

    def close(self):
        pass
        
    def _check_game_ended(self):
        if self.current_player == 1 and self.game.borne_off_white == 15:
            reward = 1 if self.game.borne_off_black > 0 else 2
            return True, reward
        elif self.current_player == -1 and self.game.borne_off_black == 15:
            reward = 1 if self.game.borne_off_white > 0 else 2
            return True, reward
        return False, 0
