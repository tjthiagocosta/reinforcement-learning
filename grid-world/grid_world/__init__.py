from gymnasium.envs.registration import register

register(
    id="grid_world/GridWorld-v1",
    entry_point="grid_world.envs:GridWorldEnv",
    max_episode_steps=300,
)
