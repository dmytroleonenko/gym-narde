from gymnasium.envs.registration import register

register(
    id='narde-v0',
    entry_point='gym_narde.envs:NardeEnv',
    max_episode_steps=1000,
)
