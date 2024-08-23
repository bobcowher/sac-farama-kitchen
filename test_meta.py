import time
import os
import gymnasium as gym
import numpy as np
from buffer import ReplayBuffer
import datetime
from agent import Agent
from gym_robotics_custom import RoboGymObservationWrapper
from torch.utils.tensorboard import SummaryWriter
from meta_agent import MetaAgent


if __name__ == '__main__':

    replay_buffer_size = 10000000
    episodes = 10
    warmup = 20
    batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.1 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 512
    learning_rate = 0.0001
    max_episode_steps=800 # max episode steps
    env_name = "FrankaKitchen-v1"
    generate_score = True
    live_test = True

    tasks = ['top burner', 'slide cabinet', 'microwave', 'hinge cabinet']
    tasks = ['microwave', 'slide cabinet', 'hinge cabinet', 'top burner']
    # tasks = ['top burner', 'microwave', 'hinge cabinet']

    if live_test:
        env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks, render_mode='human')
        env = RoboGymObservationWrapper(env)

        # print(f"Obervation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        observation, info = env.reset()

        observation_size = observation.shape[0]

        # # Agent
        meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)

        meta_agent.initialize_agents()

        meta_agent.test()

        env.close()

    if generate_score:
        print(f"Generating performance score")
        env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks)
        env = RoboGymObservationWrapper(env)

        # print(f"Obervation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        observation, info = env.reset()

        observation_size = observation.shape[0]

        # # Agent
        meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)

        meta_agent.initialize_agents()
        perf_score_epochs = 10
        total_score = 0

        for i in range(perf_score_epochs):
            score = meta_agent.test()
            total_score += score
        
        success_ratio = ((total_score / len(tasks)) / perf_score_epochs) * 100
        print(f"Success ratio {success_ratio:.2f}%")


    env.close()
