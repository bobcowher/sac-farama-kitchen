import numpy as np
import pygame

class Controller:
    def __init__(self):
        self.gripper_closed = None

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

        gripper_button_pressed = False

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
            gripper_button_pressed = True
            print("Button 1 pressed")
        elif self.joystick.get_button(3):
            self.gripper_closed = False
            gripper_button_pressed = True
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

        if np.all(action == 0) and gripper_button_pressed == False:
            action = None
        else:
            if self.gripper_closed == True:
                action[7] = -1.0  # Close gripper
                action[8] = -1.0
            elif self.gripper_closed == False:
                action[7] = 1.0  # Open gripper
                action[8] = 1.0

        return action