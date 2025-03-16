from gymnasium.envs.registration import register

register(
    id='Narde-v0',
    entry_point='gym_narde.envs.narde_env:NardeEnv',
)
