from grid2op.Agent import BaseAgent
from grid2op.Runner import Runner
from env_wrapper import Grid2opEnvWrapper
from stable_baselines3 import PPO

class Grid2opAgentWrapper(BaseAgent):
    def __init__(self,
                 gym_env: Grid2opEnvWrapper,
                 trained_agent: PPO):
        self.gym_env = gym_env
        BaseAgent.__init__(self, gym_env._gym_env.init_env.action_space)
        self.trained_agent = trained_agent
        
    def act(self, obs, reward, done):
        # you can customize it here to call the NN policy `trained_agent`
        # only in some cases, depending on the observation for example
        gym_obs = self.gym_env._gym_env.observation_space.to_gym(obs)
        gym_act, _states = self.trained_agent.predict(gym_obs, deterministic=True)
        grid2op_act = self.gym_env._gym_env.action_space.from_gym(gym_act)
        return grid2op_act
    
    def seed(self, seed):
        # implement the seed function
        if seed is None:
            return
        seed_int = int(seed)
        if seed_int != seed:
            raise RuntimeError("Seed must be convertible to an integer")
        self.trained_agent.set_random_seed(seed_int)