from env import Env

env = Env()
episodes = 50

for episode in range (episodes):
    done = False
    obs = env.reset()
    while True:
        random_action = env.action_space.sample()
        print("aciton", random_action)
        obs, reward, done, _, info = env.step(random_action)
        print("reward", reward)