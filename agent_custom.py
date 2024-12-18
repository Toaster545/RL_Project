from stable_baselines3 import PPO, DQN
from grid2op.Runner import Runner
from agent_wrapper import Grid2opAgentWrapper
from env_wrapper import Grid2opEnvWrapper
from grid2op.Reward import EpisodeDurationReward, L2RPNReward, EconomicReward
from stable_baselines3.common.env_util import make_vec_env

model_path = "./ppo_grid2op_model"

tb_log_name = "PPO_Grid2op"

log_dir = "./tensorboard_logs/"

PPO_hyperparameters = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
}


DNQ_hyperparameters = {
    "learning_rate": 1e-4,
    "buffer_size": int(1e6),
    "learning_starts": 100,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 10000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "max_grad_norm": 10,
}


RL_algo = PPO # DNQ

hyperparameters = PPO_hyperparameters

reward = EpisodeDurationReward

env_config = {
    # "backend_cls": ,
    # "backend_options": {},
    "env_name": "l2rpn_case14_sandbox", # "l2rpn_neurips_2020_track1_small"
    "env_is_test": False,
    "obs_attr_to_keep": ["gen_p", "p_or" ,"load_p", "rho", "line_status"], # "gen_q", "gen_v",
    "act_type": "discrete",
    "act_attr_to_keep": ["change_line_status", "set_line_status", "set_bus"],
    "reward_class": reward,

}


vec_env = make_vec_env(lambda: Grid2opEnvWrapper(env_config), n_envs=1)

#sb3_algo2 = RL_algo.load(model_path, env=vec_env)
#print("Model loaded successfully")


sb3_algo2 = RL_algo(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    **custom_hyperparameters
)

sb3_algo2.learn(total_timesteps=100000, tb_log_name=tb_log_name)

# Save the model
sb3_algo2.save(model_path)

print(f"Model saved at {model_path}")



