from gymnasium.envs.registration import register

register(
    id='gym_narde:narde-v0',
    entry_point='gym_narde.envs.narde_env:NardeEnv',
)
