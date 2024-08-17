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
    exploration_scaling_factor=0.1
    max_episode_steps = 500


    # Training Phase 1

    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=['microwave'])
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    agent = Agent(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)
    
    # Memory
    memory = ReplayBuffer(replay_buffer_size, input_size=observation_size, n_actions=env.action_space.shape[0], sad_robot=False)

    memory.load_from_csv(filename='checkpoints/human_memory.npz')
    time.sleep(2)

    # Training Loop
    total_numsteps = 0
    updates = 0
    pretrain_noise_ratio = 0.1

    # agent.pretrain_critic_with_human_data(memory=memory, epochs=500, batch_size=64,
    #                       summary_writer_name=f"critic_pretrain", noise_ratio=pretrain_noise_ratio)

    agent.pretrain_actor(memory=memory, epochs=50, batch_size=64, 
                         summary_writer_name=f"actor_pretrain_only", noise_ratio=pretrain_noise_ratio)

    agent.train(env=env, env_name=env_name, memory=memory, episodes=10000, 
                batch_size=batch_size, updates_per_step=updates_per_step,
                summary_writer_name=f"live_train_lr={learning_rate}_hs={hidden_size}_esp={exploration_scaling_factor}_a={alpha}_pre_train_actor_only",
                max_episode_steps=max_episode_steps)



    env.close()