import time
import os
import gymnasium as gym
import numpy as np
from buffer import *
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
    max_episode_steps=1200 # max episode steps
    env_name = "FrankaKitchen-v1"
    exploration_scaling_factor=0.01

    tasks = ['top burner', 'slide cabinet']
    tasks = ['kettle']
    # task = "top burner"
    tasks = ["hinge cabinet"]
    # tasks = ['microwave', 'top burner', 'hinge cabinet']
    # tasks = ['top burner']
    # tasks = ['microwave']
    # tasks = ["hinge cabinet", "microwave"]
    # tasks = ['top burner', 'bottom burner', 'slide cabinet']
    # tasks = ['bottom burner']
    # tasks = ['kettle', 'top burner', 'slide cabinet']
    tasks = ['slide cabinet', 'light switch']


    task_no_spaces = 'omni'

    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks, render_mode='human', autoreset=False)

    env = RoboGymObservationWrapper(env)

    print(env.env.env.env.env.model.opt.gravity)

    # print(f"Obervation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    state, info = env.reset()

    state_size = state.shape[0]

    print(f"Input size: ", state_size)
    print(f"n_actions: {env.action_space.shape[0]}")

    memory = ReplayBuffer(replay_buffer_size, input_size=state_size, n_actions=env.action_space.shape[0])

    memory.load_from_csv(filename=f'checkpoints/human_memory_{task_no_spaces}.npz')
    
    starting_memory_size = memory.mem_ctr
    
    print(f"Starting memory size is {starting_memory_size}")

    controller = Controller()


    while True: # Run until interrupted
        episode_steps = 0
        done = False
        state, info = env.reset()

        while not done and episode_steps < max_episode_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                # Check if CTRL+H is pressed
                    if event.key == pygame.K_h and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        # Trigger the key event in MuJoCo
                        env.render()  # Ensure the environment handles the key event
                    # Handle other key events for the controller
                    action = controller.get_action()

            action = controller.get_action()
            if(action is not None):
                next_state, reward, done, _, _ = env.step(action)
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                memory.store_transition(state, action, reward, next_state, mask) 
                print(f"Episode step: {episode_steps} Reward: , {reward} Successfully added {memory.mem_ctr - starting_memory_size} steps to memory. Total: {memory.mem_ctr}")
                state = next_state
                # print(state)
                episode_steps += 1
            time.sleep(0.05)

        
        memory.save_to_csv(filename=f'checkpoints/human_memory_{task_no_spaces}.npz')

        goal_counts = classify_records_in_replay_buffer(replay_buffer=memory,
                                                        goal_start_map=env.goal_start_map)

        print(f"Memory counts {goal_counts}")




