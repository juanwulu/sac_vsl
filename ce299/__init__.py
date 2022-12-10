from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import register

_load_env_plugins()

# Register custom environment
# =========================================

register(
    id='CAVI80VSL_v0',
    entry_point="env.CAVI80VSLEnv",
    max_episode_steps=600,
)
