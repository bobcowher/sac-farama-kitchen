import time
import os
import gymnasium as gym
import numpy as np
from buffer import ReplayBuffer
import datetime
from agent import Agent
from gym_robotics_custom import RoboGymObservationWrapper
from torch.utils.tensorboard import SummaryWriter
import pygame
from controller import Controller


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
    env_name = "FrankaKitchen-v1"
    exploration_scaling_factor=0.01

    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=['microwave'], render_mode='human', )

    env = RoboGymObservationWrapper(env)

    print(env.env.env.env.env.model.opt.gravity)

    # print(f"Obervation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation, info = env.reset()


    controller = Controller()

    done = False


    state = env.reset()


    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = controller.get_action()
        if(not np.all(action == 0)):
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
        time.sleep(0.05)