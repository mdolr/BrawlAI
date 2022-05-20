import os
import sys
import cv2
import gym
import pynput
import pytesseract
import numpy as np
from mss import mss
from PIL import Image


class ImageCaptureEnv(gym.Env):
    """
    A custom gym environment that works with continuous stream capture
    for each step in order to solve a task using direct vision data.
    """

    def __init__(self, width, height, model):
        super().__init__()

        # Initialize the keyboard and mouse controller for this environment
        # using pynput to input directly into the game simulating
        # an user input
        self.keyboard = pynput.keyboard.Controller()
        self.mouse = pynput.mouse.Controller()

        # The observation space corresponds to the dimension of our image with
        # a value ranging from 0 to 255 for each pixel
        # We are using Grayscale so only 1 dimension for each pixel
        # replace 1 by 3 for RGB
        self.observation_space = gym.spaces.Box(0, 255, [height, width, 1])

        # We are using a multidiscrete action space
        # The first one corresponds to the direction no-move, up, down, right, left
        # The other corresponds to a binary choice for pressing X C and V
        # gym.spaces.MultiDiscrete([5, 2, 2, 2])
        self.action_space = gym.spaces.Discrete(15)
        
        # The model that will evolve in this environment
        self.model = model(self)

        self.previous_observation = None

        # Screen capture parameters
        self.monitor = {'left': 0, 'top': 100, 'bottom':70, 'width': 3456, 'height': 2234}
        self.sct = None

        with mss() as sct:
            self.sct = sct

        self.direction = None
        self.last_health = 0
        self.reward = 0

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        print('Reset')
        sys.exit(1)

    def step(self, action):
        """
        This function runs at each step and executes the action
        then computes the reward and checks if the episode is over.
        """
        #print(action)
        if len(str(action)) >= 2:
          action = np.argmax(action)
        
        print(f'Action: {action}')

        if self.direction is not None:
            self.keyboard.release(self.direction)
            self.direction = None

        if action % 4 == 0:
            self.direction = pynput.keyboard.Key.left
        elif action % 4 == 1:
            self.direction = pynput.keyboard.Key.right
        elif action % 4 == 2:
            self.direction = pynput.keyboard.Key.up
        elif action % 4 == 3:
            self.direction = pynput.keyboard.Key.down

        self.keyboard.press(self.direction)

        if action >= 4 and action < 8:
            self.keyboard.tap('x')
        elif action >= 8 and action < 12:
            self.keyboard.tap('c')
        elif action >= 12 and action < 16:
            self.keyboard.tap('v')

        # reward = % de rouge + difference de vie

        sct_img = self.sct.grab(self.monitor)
        shot = np.array(Image.frombytes(
            'RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))
        
        gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        gray = gray[350:2050, 750:]
        gray = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)

        self_life_count_crop = shot[50:125, 3100:3160]
        opponent_life_count_crop = shot[50:125, 3270:3330]
    
        self_health_crop = shot[110, 3168]
        opponent_health_crop = shot[110, 3338]
        """
        self_life_count = pytesseract.image_to_string(self_life_count_crop, lang='eng',
                                                      config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        opponent_life_count = pytesseract.image_to_string(opponent_life_count_crop, lang='eng',
                                                          config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        """
        self_health = [abs(255 - health) for health in self_health_crop]
        self_health = (np.sum(self_health) / (255*3)) 

        #self_health = self_health_crop[2] / sum(self_health_crop)
        #opponent_health = opponent_health_crop[2] / sum(opponent_health_crop)
        """
        #print(
        #    f'PyTesseract pre-if self: {self_life_count} opponent: {opponent_life_count}')

        if self_life_count == '' or self_life_count == '-':
            self_life_count = 3

        if opponent_life_count == '' or opponent_life_count == '-':
            opponent_life_count = 3
        """
        #print(
        #    f'PyTesseract post-if self: {self_life_count} opponent: {opponent_life_count}')

        #reward = (int(self_life_count) - int(opponent_life_count)) * \
        #    255 + (self_health - opponent_health) * 255
        print(self_health)
        
        if self_health > self.last_health:
            self.reward += self_health
            self.last_health = self_health

        reward = self.reward

        print(f'Reward: {reward}')

        if self.previous_observation is not None:
            self.model.update_replay_memory(
                (self.previous_observation, action, reward, gray))

        self.previous_observation = gray

        """
        if self_life_count == 0 or opponent_life_count == 0:
            self.reset()
        """

        observation = gray

        return observation, action, reward, {}
