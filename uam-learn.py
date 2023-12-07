import gym
from stable_baselines3 import PPO
import os
import time
from src.env import Env

models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
env = Env()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 100000000000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    
    
# env.close()