from env import Env

env = Env()
episodes = 50
for episode in range(episodes):
    done = False
    obs = env.reset()
    while True:
        random_aciton = env.action_space.sample()
        print("aciton", random_aciton)
        obs, reward, done, _, info = env.step(random_aciton)
        print("reward", reward)