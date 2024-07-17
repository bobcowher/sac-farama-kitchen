import time
import os
import gymnasium as gym
import numpy as np
from agent import SAC
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer



if __name__ == '__main__':

    replay_buffer_size = 10000000
    episodes = 10000
    warmup = 20
    batch_size = 64
    updates_per_step = 4
    gamma = 0.99
    tau = 0.005
    alpha = 0.2 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 512
    learning_rate = 0.0001
    env_name = "PointMaze_UMaze-v3"
    exploration_scaling_factor=1


    # Training Phase 1
    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
    
    max_episode_steps_phase_1 = 100

    env = gym.make(env_name, max_episode_steps=max_episode_steps_phase_1, maze_map=STRAIGHT_MAZE)
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    agent = SAC(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)
    
        # Memory
    memory = ReplayBuffer(replay_buffer_size, input_size=observation_size, n_actions=env.action_space.shape[0])

    agent.train(env=env, env_name=env_name, memory=memory, episodes=100, 
                batch_size=batch_size, updates_per_step=updates_per_step,
                summary_writer_name=f"small_maze_temp={alpha}_lr={learning_rate}_hs={hidden_size}",
                max_episode_steps=max_episode_steps_phase_1)

    # Training Phase 2
    LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    max_episode_steps_phase_2 = 500
    
    env = gym.make(env_name, max_episode_steps=max_episode_steps_phase_2, maze_map=LARGE_MAZE)
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    agent = SAC(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)

    agent.train(env=env, env_name=env_name, memory=memory, episodes=episodes, 
                batch_size=batch_size, updates_per_step=updates_per_step,
                summary_writer_name=f"large_maze_temp={alpha}_lr={learning_rate}_hs={hidden_size}",
                max_episode_steps=max_episode_steps_phase_2)

    env.close()