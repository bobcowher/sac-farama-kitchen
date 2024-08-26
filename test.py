import time
import os
import gymnasium as gym
import numpy as np
from buffer import ReplayBuffer
import datetime
from agent import Agent
from gym_robotics_custom import RoboGymObservationWrapper
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    replay_buffer_size = 10000000
    episodes = 2
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
    env_name = "FrankaKitchen-v1"

    # task = "top burner"
    tasks = ["microwave", "hinge cabinet", "top burner"]
    task_no_spaces = "omni"


    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks, render_mode='human')
    env = RoboGymObservationWrapper(env)

    # print(f"Obervation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation, info = env.reset()

    observation_size = observation.shape[0]

    print(observation_size)

    # # Agent
    agent = Agent(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha,
                target_update_interval=target_update_interval, hidden_size=hidden_size, 
                learning_rate=learning_rate, goal=task_no_spaces)

    agent.load_checkpoint(evaluate=True)

    agent.test(env=env, episodes=episodes, max_episode_steps=500)

    env.close()
