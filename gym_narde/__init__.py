from gym.envs.registration import register

register(
    id='narde-v0',
    entry_point='gym_narde.envs:NardeEnv',
)
