import numpy as np
import json

def read_episode_data(path):
    with open(path, "r") as file:
        data = json.load(file)

    max_time_steps = int(data['chronics_max_timestep'])
    cumulative_rew = data['cumulative_reward']
    survived_time_steps = data['nb_timestep_played']
    percentage_ep = (survived_time_steps/ max_time_steps)*100
    finished_ep = survived_time_steps == max_time_steps
    return cumulative_rew, survived_time_steps, percentage_ep, finished_ep


def read_episodes(files):

    survived=[]
    rewards = []
    percentage = []
    finished = []
   
    for file in files: # 1 file = 1 episode
        cumulative_rew, survived_time_steps, percentage_ep, finished_ep = read_episode_data(file)
        
        rewards.append(cumulative_rew)
        survived.append( survived_time_steps )
        percentage.append( percentage_ep )
        finished.append( finished_ep )

    return np.array(rewards), np.array(survived), np.array(percentage), np.array(finished)

def fpath(algo,prefix=''):
    model_name = algo.__name__
    tb_log_name = prefix + "_"+ f"{model_name}"
    model_path = "./" + prefix + "/" f"{model_name}"
    return model_name, model_path, tb_log_name