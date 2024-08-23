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
    episodes = 3000
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
    max_episode_steps=1500 # max episode steps
    env_name = "FrankaKitchen-v1"

    tasks = ['top burner', 'microwave', 'hinge cabinet']
    # tasks = ['microwave', 'slide cabinet', 'hinge cabinet']


    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks)
    env = RoboGymObservationWrapper(env)

    # print(f"Obervation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)

    meta_agent.initialize_memory(augment_data=True, augment_rewards=True, augment_noise_ratio=0.1)

    meta_agent.initialize_agents(learning_rate=learning_rate)

    # meta_agent.load_memory()

    meta_agent.train(episodes=episodes, summary_writer_name=f'meta_agent_a={alpha}')

    meta_agent.save_models()

    env.close()
