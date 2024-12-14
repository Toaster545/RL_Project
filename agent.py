from stable_baselines3 import PPO
from grid2op.Runner import Runner
from agent_wrapper import Grid2opAgentWrapper
from env_wrapper import Grid2opEnvWrapper
from stable_baselines3.common.env_util import make_vec_env

model_path = "./ppo_grid2op_model"

log_dir = "./tensorboard_logs/"
vec_env = make_vec_env(lambda: Grid2opEnvWrapper(), n_envs=1)

sb3_algo2 = PPO.load(model_path, env=vec_env)
print("Model loaded successfully")

"""
custom_hyperparameters = {
    "learning_rate": 1e-5,
    "n_steps": 1024,
    "batch_size": 32,
    "n_epochs": 5,
    "gamma": 0.98,
    "gae_lambda": 0.9,
    "clip_range": 0.3,
    "ent_coef": 0.01,
}

# Initialize PPO with custom hyperparameters
sb3_algo2 = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    **custom_hyperparameters
)
"""
sb3_algo2.learn(total_timesteps=100000, tb_log_name="PPO_Grid2op")

# Save the model
sb3_algo2.save(model_path)

print(f"Model saved at {model_path}")



