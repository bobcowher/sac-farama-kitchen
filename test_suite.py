import os
import gymnasium as gym
import numpy as np
from agent import Agent
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from meta_agent import MetaAgent

# Start test to ensure new memories are getting rolled onto the stack. 
buffer_size = 10
loop_size = 20

memory = ReplayBuffer(buffer_size, input_size=1, n_actions=1)

for i in range(loop_size):
    memory.store_transition(i, i, i, i, i)

# print("MemorySize:", memory.mem_size)
# print("Memory:", memory.state_memory)

print("Testing to ensure the first memory state is correct.")
assert memory.state_memory[0] == 10
print("Test Successful\n")

print("Testing to ensure the last memory state is correct.")
assert memory.state_memory[-1] == 19
print("Test Successful\n")
# Complete test to ensure new memories are getting rolled onto the stack. 

replay_buffer_size = 10000000
episodes = 10000
warmup = 20
batch_size = 64
updates_per_step = 4
gamma = 0.99
tau = 0.005
alpha = 0.12 # Temperature parameter.
policy = "Gaussian"
target_update_interval = 1
automatic_entropy_tuning = False
hidden_size = 512
learning_rate = 0.0001
env_name = "FrankaKitchen-v1"
exploration_scaling_factor=0.1
max_episode_steps = 500


# Training Phase 1
tasks = ['microwave', 'top burner']

env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks)
env = RoboGymObservationWrapper(env)

observation, info = env.reset()

observation_size = observation.shape[0]

# # Agent
agent = Agent(observation_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, 
              target_update_interval=target_update_interval, hidden_size=hidden_size, 
              learning_rate=learning_rate, goal='microwave')

# Memory
memory = ReplayBuffer(replay_buffer_size, input_size=observation_size, n_actions=env.action_space.shape[0], sad_robot=True)

memory.load_from_csv(filename='checkpoints/human_memory_microwave.npz')

rewards = [reward for reward in memory.reward_memory if reward > 0]

print(f"found {len(rewards)} rewards in the buffer")


print("Reward batch, no augmentation")
state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=16)
print(reward_batch)

print("Reward batch with augmentation")
memory.augment_data = True
state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=16)
print(reward_batch)

