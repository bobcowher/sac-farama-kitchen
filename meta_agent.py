import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import *
from model import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from buffer import ReplayBuffer
import time
from agent import Agent
from gym_robotics_custom import RoboGymObservationWrapper

class MetaAgent(object):
    def __init__(self, env : RoboGymObservationWrapper, goal_list=['microwave']):
        self.agent_dict = {}
        goal_list_no_spaces = [a.replace(" ", "_") for a in goal_list]
        self.goal_dict = dict(zip(goal_list_no_spaces, goal_list))
        self.env = env
        self.agent: Agent = None
        
        self.initialize_agents()


        
    
    def initialize_agents(self, gamma=0.99, tau=0.005, alpha=0.1, 
                          target_update_interval=2, hidden_size=512, 
                          learning_rate=0.0001):
        
        for goal in self.goal_dict:

            self.env.set_goal(self.goal_dict[goal])
            observation, info = self.env.reset()
            observation_size = observation.shape[0]

            agent = Agent(observation_size, self.env.action_space, gamma=gamma, tau=tau, alpha=alpha,
                          target_update_interval=target_update_interval, hidden_size=hidden_size, 
                          learning_rate=learning_rate, goal=goal)
            print(f"Loading checkpoint for {goal}")
            agent.load_checkpoint(evaluate=True)
            self.agent_dict[goal] = agent


    def test(self):
        action = None

        for goal in self.goal_dict:
            print(f"Attempting goal {goal}..")
            self.env.set_goal(self.goal_dict[goal])
            self.agent = self.agent_dict[goal]

            action = self.agent.test(env=self.env, episodes=1, max_episode_steps=500, prev_action=action)









