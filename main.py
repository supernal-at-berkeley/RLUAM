import gym
from src.env import *
from utils.logger import logger
from agents.pg_agent import PGAgent

import os
import time
import numpy as np
import torch
from infrastructure import pytorch_util as ptu
from infrastructure import utils
from infrastructure.action_noise_wrapper import ActionNoiseWrapper




def run_training_loop(args):
    two_vertiport_system = Env()

    ob_dim = two_vertiport_system.ob_dim
    ac_dim = two_vertiport_system.ac_dim
    discrete = True
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()
    max_ep_len = args.ep_len or 1000

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(two_vertiport_system, agent.actor, args.batch_size, max_ep_len)  # TODO
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # TODO: train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(trajs_dict['observation'],trajs_dict['action'], trajs_dict['reward'],
                                        trajs_dict['terminal'])

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                two_vertiport_system, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")


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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--action_noise_std", type=float, default=0)

    args = parser.parse_args()


    two_vertiport_system = Env()

    for i in range(5):
        action = two_vertiport_system.compute_action()
        two_vertiport_system.step(action)
        logger(two_vertiport_system)

    run_training_loop(args)









if __name__ == "__main__":
    main()


