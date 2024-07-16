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
    hidden_size = 256
    learning_rate = 0.0001
    max_episode_steps=100 # max episode steps
    env_name = "PointMaze_UMaze-v3"
    exploration_scaling_factor=1


    STRAIGHT_MAZE = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
    

    MEDIUM_MAZE_DIVERSE_GR = [[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, C, 0, 1, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, C, 1],
                        [1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1],
                        [1, C, 1, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, C, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]]

    env = gym.make(env_name, max_episode_steps=max_episode_steps, maze_map=MEDIUM_MAZE_DIVERSE_GR)
    env = RoboGymObservationWrapper(env)

    # print(f"Obervation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation, info = env.reset()

    observation_size = observation.shape[0]

    # # Agent
    agent = SAC(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, policy=policy,
                target_update_interval=target_update_interval, automatic_entropy_tuning=automatic_entropy_tuning,
                hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)

    # agent.load_checkpoint()

    # Tesnorboard
    writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_hidden_size={hidden_size}_lr={learning_rate}_batch_size={batch_size}_intrinsic_curiosity={exploration_scaling_factor}_ups={updates_per_step}')

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
            if warmup > i_episode:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if memory.can_sample(batch_size=batch_size):
                # Number of updates per step in environment
                for i in range(updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, prediction_error_loss, alpha = agent.update_parameters(memory,
                                                                                                         batch_size,
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('loss/prediction_error', prediction_error_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _, _ = env.step(action)  # Step
                
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == max_episode_steps else float(not done)

            memory.store_transition(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))
        if i_episode % 10 == 0:
            agent.save_checkpoint(env_name=env_name)


    env.close()
