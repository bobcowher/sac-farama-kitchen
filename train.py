import time
import os
import gymnasium as gym
import numpy as np
from agent import Agent
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
    alpha = 0.1 # Temperature parameter.
    policy = "Gaussian"
    target_update_interval = 1
    automatic_entropy_tuning = False
    hidden_size = 512
    learning_rate = 0.0001
    env_name = "FrankaKitchen-v1"
    max_episode_steps = 500

    task = "slide cabinet"
    task_no_spaces = task.replace(" ", "_")

    # Training Phase 1

    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task])
    env = RoboGymObservationWrapper(env, goal=task)

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    agent = Agent(observation_size, env.action_space, gamma=gamma, tau=tau, 
                  alpha=alpha, target_update_interval=target_update_interval,
                  hidden_size=hidden_size, learning_rate=learning_rate, goal=task_no_spaces)
    
    # Memory
    memory = ReplayBuffer(replay_buffer_size, input_size=observation_size, 
                          n_actions=env.action_space.shape[0], augment_rewards=True, augment_data=True)

    memory.load_from_csv(filename=f'checkpoints/human_memory_{task_no_spaces}.npz')
    time.sleep(2)

    # Training Loop
    total_numsteps = 0
    updates = 0
    pretrain_noise_ratio = 0.1

    # Phase 1
    memory.expert_data_ratio = 0.5
    agent.train(env=env, env_name=env_name, memory=memory, episodes=100, 
                batch_size=batch_size, updates_per_step=updates_per_step,
                summary_writer_name=f"live_train_phase_1_{task_no_spaces}",
                max_episode_steps=max_episode_steps)

    # Phase 2
    memory.expert_data_ratio = 0.25
    agent.train(env=env, env_name=env_name, memory=memory, episodes=200, 
                batch_size=batch_size, updates_per_step=updates_per_step,
                summary_writer_name=f"live_train_phase_2_{task_no_spaces}",
                max_episode_steps=max_episode_steps)

    # Phase 3
    memory.expert_data_ratio = 0
    agent.train(env=env, env_name=env_name, memory=memory, episodes=1000, 
                batch_size=batch_size, updates_per_step=updates_per_step,
                summary_writer_name=f"live_train_phase_3_{task_no_spaces}",
                max_episode_steps=max_episode_steps)


    env.close()