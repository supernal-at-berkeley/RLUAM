import gym
from src.env import *
from utils.logger import logger






def main():
    # Create the CartPole environment
    # env = gym.make('CartPole-v1')

    # try:
    #     # Initialize the environment
    #     observation = env.reset()

    #     for time_step in range(100):
    #         # Render the environment (optional, for visualization)
    #         env.render()

    #         # Choose a random action (in this case, 0 or 1 for CartPole)
    #         action = env.action_space.sample()

    #         # Take the chosen action and observe the next state, reward, and done flag
    #         observation, reward, done, info = env.step(action)

    #         # Check if the episode is done (e.g., the pole fell or time limit exceeded)
    #         if done:
    #             print(f"Episode ended after {time_step+1} timesteps.")
    #             break
    # finally:
    #     # Close the environment
    #     env.close()
    two_vertiport_system = Env()

    # ob_dim = two_vertiport_system.observation_space.shape[0]
    # print(ob_dim)

    for i in range(5):
        action = two_vertiport_system.compute_action()
        two_vertiport_system.step(action)
        logger(two_vertiport_system)









if __name__ == "__main__":
    main()


