import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper


class RoboGymObservationWrapper(ObservationWrapper):

    def __init__(self, env, goal='microwave'):
        super(RoboGymObservationWrapper, self).__init__(env)
        env_model = env.env.env.env.model
        env_model.opt.gravity[:] = [0, 0, -1]
        self.goal = goal
        self.goal_dict_max_length = 17
        self.goal_start_map = {
                'microwave': 0,
                'top burner': 1,
                'bottom burner': 3,
                'light switch': 5,
                'slide cabinet': 7,
                'hinge cabinet': 8,
                'kettle': 10
            }
        

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        observation, info = self.env.reset()
        observation = self.process_observation(observation)
        return observation, info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = self.process_observation(observation)
        return observation, reward, done, truncated, info

    def process_observation(self, observation):
        obs_pos = observation['observation']
        obs_achieved_goal = observation['achieved_goal']
        obs_desired_goal = observation['desired_goal']

        obs_desired_goal = np.zeros(self.goal_dict_max_length)
        obs_achieved_goal = np.zeros(self.goal_dict_max_length)

        for key in observation['desired_goal']:
            counter = 0
            for val in observation['desired_goal'][key]:
                obs_desired_goal[self.goal_start_map[key] + counter] = val
                counter += 1

        for key in observation['achieved_goal']:
            counter = 0
            for val in observation['achieved_goal'][key]:
                obs_achieved_goal[self.goal_start_map[key] + counter] = val
                counter += 1
        
        obs_concatenated = np.concatenate((obs_pos, obs_desired_goal, obs_achieved_goal))

        return obs_concatenated

