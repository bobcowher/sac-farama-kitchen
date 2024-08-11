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

class Controller:
    def __init__(self):
        self.gripper_closed = False

        pygame.init()
        pygame.joystick.init()

        # Assuming only one joystick is connected
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_action(self):
        """
        Map PlayStation controller input to the robot's action space.
        """
        action = np.zeros(9)  # Assuming 9 action dimensions as specified
        # action = np.full(9, 0.1)

        # Map left joystick to panda0_joint1 and panda0_joint2 angular velocity
        action[0] = self.joystick.get_axis(0)  # Left stick horizontal
        action[1] = self.joystick.get_axis(1)  # Left stick vertical

        action[0] = action[0] * -1

        # Map right joystick to panda0_joint3 and panda0_joint4 angular velocity
        action[2] = self.joystick.get_axis(3)  # Right stick vertical
        action[2] = action[2]

        action[3] = self.joystick.get_axis(2)  # Right stick horizontal
        action[3] = action[3] * -1

        # # Map L2 and R2 triggers to panda0_joint5 and panda0_joint6 angular velocity
        # action[4] = joystick.get_axis(2)  # L2 trigger
        # action[5] = joystick.get_axis(5)  # R2 trigger

        # Map buttons or D-pad to gripper control
        if self.joystick.get_button(0):  # X button
            action[4] = -1
            print("Button 0 pressed")
        elif self.joystick.get_button(2):  # Circle button
            action[4] = 1
            print("Button 2 pressed")
        elif self.joystick.get_button(1):
            self.gripper_closed = True
            print("Button 1 pressed")
        elif self.joystick.get_button(3):
            self.gripper_closed = False
            print("Button 3 pressed")
        elif self.joystick.get_button(4):  # Circle button
            action[5] = 1
            print("Button 4 pressed")
        elif self.joystick.get_button(5):
            action[5] = -1
            print("Button 5 pressed")
        elif self.joystick.get_button(6):  # Circle button
            action[6] = 1
            print("Button 6 pressed")
        elif self.joystick.get_button(7):
            action[6] = -1
            print("Button 7 pressed")
        elif self.joystick.get_button(8):  # Circle button
            action[7] = 1
            print("Button 8 pressed")
        elif self.joystick.get_button(9):
            action[7] = -1
            print("Button 9 pressed")
        
        mask = np.abs(action) >= 0.1
        action = action * mask
        action = np.where(action == -0.0, 0.0, action)

        if self.gripper_closed == True:
            action[7] = -1.0  # Close gripper
            action[8] = -1.0
        elif self.gripper_closed == False:
            action[7] = 1.0  # Open gripper
            action[8] = 1.0

        return action


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

    # print(dir(env.env.env.env.model.opt.gravity))


    env = RoboGymObservationWrapper(env)

    env_model = env.env.env.env.env.model

    env.model.opt.gravity[:] = [0, 0, -1]

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
            # print(action)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
        time.sleep(0.05)