import gym
import numpy as np
from gym import spaces

from gym_backgammon.envs.backgammon import Backgammon

class BackgammonEnv(gym.Env):
    def __init__(self):
        super(BackgammonEnv, self).__init__()

        self.game = Backgammon()
        self.current_player = 1  # Always white perspective

        self.observation_space = spaces.Box(low=-15, high=15, shape=(24,), dtype=np.int)
        self.action_space = spaces.Discrete(24 * 24) # Representing all possible moves

    def _get_obs(self):
        return self.game.get_perspective_board(self.current_player)

    def step(self, action):
        # Action is always from current player's perspective
        move = (action // 24, action % 24) # Convert discrete action to move
        self.game.execute_rotated_move(move, self.current_player)

        # Switch perspective for next player
        self.current_player *= -1

        # Check winner from current perspective
        done = self._check_winner()
        reward = 1 if done else 0
        return self._get_obs(), reward, done, {}

    def _check_game_ended(self):
        # Check if either player has no checkers left
        current_has_won = np.sum(self.game.board > 0) == 0  # Current perspective's player
        opponent_has_won = np.sum(self.game.board < 0) == 0  # Other player
        return current_has_won or opponent_has_won

    def reset(self):
        self.game = Backgammon()
        self.current_player = 1
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
