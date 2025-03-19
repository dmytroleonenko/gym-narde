from gymnasium.envs.registration import register

register(
    id='Narde-v0',
    entry_point='gym_narde.envs:NardeEnv',
    max_episode_steps=1000,
)

# Register JAX-accelerated version
register(
    id='Narde-jax-v0',
    entry_point='gym_narde.envs:NardeEnvJAX',
    max_episode_steps=1000,
)
