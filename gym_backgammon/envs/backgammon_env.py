import gym
import numpy as np
from gym import spaces

from gym_backgammon.envs.backgammon import Narde

class NardeEnv(gym.Env):
    def __init__(self):
        super(NardeEnv, self).__init__()

        self.game = Narde()
        self.current_player = 1  # Always white perspective

        self.observation_space = spaces.Box(low=-15, high=15, shape=(24,), dtype=np.int32)
        self.action_space = spaces.Tuple((
            spaces.Discrete(24 * 24),
            spaces.Discrete(24 * 24)
        ))

    def _get_obs(self):
        return self.game.get_perspective_board(self.current_player)

    def step(self, action):
        # Roll two dice at the beginning of the turn
        dice = [np.random.randint(1, 7), np.random.randint(1, 7)]
        
        valid_moves = self.game.get_valid_moves(dice, self.current_player)

        if len(valid_moves) == 0:
            # No legal moves; skip turn.
            done, reward = self._check_game_ended()
            if not done:
                self.current_player *= -1
            return self._get_obs(), reward, done, {}

        elif len(valid_moves) == 1:
            # Only one legal move—that is, using the higher die.
            self.game.execute_rotated_move(valid_moves[0], self.current_player)
        else:
            move1_code, move2_code = action
            move1 = (move1_code // 24, move1_code % 24)
            move2 = (move2_code // 24, move2_code % 24)
            if move1 in valid_moves:
                self.game.execute_rotated_move(move1, self.current_player)
                # Recalculate valid moves for the remaining die after move1 is executed.
                # (You need to select the remaining die value from dice; for example:)
                remaining_die = [d for d in dice if d != (abs(move1[0] - (move1[1] if move1[1] != 'off' else 0)))][0]
                new_valid_moves = self.game.get_valid_moves([remaining_die], self.current_player)
                if move2 in new_valid_moves:
                    self.game.execute_rotated_move(move2, self.current_player)

        # Check if game ended and compute reward
        done, reward = self._check_game_ended()

        # Switch players if game continues
        if not done:
            self.current_player *= -1
            
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.game = Narde()
        while True:
            white_roll = np.random.randint(1, 7)
            black_roll = np.random.randint(1, 7)
            if white_roll != black_roll:
                break
        # The winning player (strictly higher roll) becomes White; otherwise Black.
        self.current_player = 1 if white_roll > black_roll else -1
        return self._get_obs()

    def render(self, mode='human'):
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
