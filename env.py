import grid2op
try:
    from lightsim2grid import LightSimBackend
    bk_cls = LightSimBackend
except ImportError as exc:
    print(f"Error: {exc} when importing faster LightSimBackend")
    from grid2op.Backend import PandaPowerBackend
    bk_cls = PandaPowerBackend

import gymnasium
import numpy as np
from grid2op.gym_compat import GymEnv
    
def create_env(env_name = "l2rpn_neurips_2020_track1_small"):
    env_glop = grid2op.make(env_name, test=False, backend=bk_cls())
    env_gym = GymEnv(env_glop)
    print(f"The \"env_gym\" is a gym environment: {isinstance(env_gym, gymnasium.Env)}")
    obs_gym, info = env_gym.reset()
    return env_gym