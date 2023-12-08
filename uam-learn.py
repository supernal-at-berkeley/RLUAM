import gym
from stable_baselines3 import PPO
import os
import time
from src.env import Env
from gym import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
# from HybridPPO.hybridppo import HybridPPO
from sb3_contrib import RecurrentPPO

models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
env = Env()
env.reset()


model = RecurrentPPO("MlpLstmPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=logdir,
                )


TIMESTEPS = 10000
for i in range(1, 100000000000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="ppo_lstm")
    #model.save(model_dir)
    model.save(f"{models_dir}/ppo_recurrent")
# env.close()