import time
import os
import gymnasium as gym
import numpy as np
from buffer import ReplayBuffer
import datetime
from agent import SAC
from gym_robotics_custom import RoboGymObservationWrapper
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    replay_buffer_size = 10000000
    episodes = 10
    warmup = 20
    batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.15 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 512
    learning_rate = 0.0001
    max_episode_steps=500 # max episode steps
    env_name = "PointMaze_UMaze-v3"
    exploration_scaling_factor=0.01

    MEDIUM_MAZE_DIVERSE_GR = [[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, "C", 0, 1, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, "C", 1],
                        [1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1],
                        [1, "C", 1, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, "C", 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]]


    LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    env = gym.make(env_name, max_episode_steps=max_episode_steps, render_mode='human', maze_map=LARGE_MAZE)
    env = RoboGymObservationWrapper(env)

    # print(f"Obervation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    agent = SAC(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)

    agent.load_checkpoint(evaluate=True)

    agent.test(env=env, episodes=10, max_episode_steps=500)

    env.close()
