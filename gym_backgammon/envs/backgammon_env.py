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
        move = (action // 24, action % 24)
        self.game.execute_rotated_move(move, self.current_player)

        done = self._check_game_ended()

        # Reward is always 1 for win, 0 otherwise (opponent's turn comes next)
        reward = 1 if done else 0

        # Switch players if game continues
        if not done:
            self.current_player *= -1

        return self._get_obs(), reward, done, {}

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
