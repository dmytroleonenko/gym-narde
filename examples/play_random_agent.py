import gym
import time
from itertools import count
import random
import numpy as np
from gym_narde.envs.narde import WHITE, BLACK, COLORS, TOKEN

env = gym.make('gym_narde:narde-v0')
# env = gym.make('gym_narde:narde-pixel-v0')

random.seed(0)
np.random.seed(0)


class RandomAgent:
    def __init__(self, color):
        self.color = color
        self.name = 'AgentExample({})'.format(self.color)

    def choose_best_action(self, env):
        return env.action_space.sample()


def make_plays():
    wins = {WHITE: 0, BLACK: 0}

    agents = {WHITE: RandomAgent(WHITE), BLACK: RandomAgent(BLACK)}

    observation = env.reset()
    agent_color = env.current_player

    agent = agents[agent_color]

    t = time.time()

    env.render(mode='human')

    for i in count():
        print("Current player={} ({} - {}) turn".format(agent.color, TOKEN[agent.color], COLORS[agent.color]))

        action = agent.choose_best_action(env)

        observation_next, reward, done, info = env.step(action)

        env.render(mode='human')

        if done:
            winner = env.current_player
            wins[winner] += 1

            tot = wins[WHITE] + wins[BLACK]
            tot = tot if tot > 0 else 1

            print("Game={} | Winner={} after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(1, winner, i,
                agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))

            break

        agent_color = env.current_player
        agent = agents[agent_color]
        observation = observation_next

    env.close()


if __name__ == '__main__':
    make_plays()
