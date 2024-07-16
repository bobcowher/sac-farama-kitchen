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
    episodes = 5
    warmup = 20
    batch_size = 64
    updates_per_step = 1
    gamma = 0.99
    tau = 0.005
    alpha = 0.15 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 256
    learning_rate = 0.0001
    max_episode_steps=100 # max episode steps
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

    agent.load_checkpoint()

    # Memory
    memory = ReplayBuffer(replay_buffer_size, input_size=observation_size, n_actions=env.action_space.shape[0])

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in range(episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()

        while not done and episode_steps < max_episode_steps:

            action = agent.select_action(state)  # Sample action from policy
            # print(f"Action: {action}")

            next_state, reward, done, _, _ = env.step(action)  # Step
            # print(next_state.shape)
            episode_steps += 1
            total_numsteps += 1
            if reward == 1:
                done = True
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == max_episode_steps else float(not done)

            state = next_state

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))



    env.close()
