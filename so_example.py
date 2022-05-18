
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random
from collections import deque
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.models import Sequential
import time
import pytesseract
import sys
import pynput
from PIL import Image, ImageOps  # , ImageGrab
from mss import mss
import gym
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


keyboard = pynput.keyboard.Controller()
mouse = pynput.mouse.Controller()

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1  # 50_000  # How many last steps to keep for model training
# Minimum number of steps in a memory to start training
MIN_REPLAY_MEMORY_SIZE = 1  # 1_000
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'BOX'

N_ACTIONS = 16

# Exploration settings
ELIPSON_DECAY = 0.999988877665
MIN_EPSILON = 0.0001

# For stats
ep_rewards = [-200]

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Own Tensorboard class


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self, env):
        self.env = env

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self,):
        with tf.device('cpu:0'):
            model = Sequential()

            observation_space = 1, 3456, 2234, 1
            action_space = self.env.action_space.n

            model.add(Conv2D(32, (3, 3), activation='relu',
                      input_shape=observation_space[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            # this converts our 3D feature maps to 1D feature vectors
            model.add(Flatten())
            model.add(Dense(64))

            # ACTION_SPACE_SIZE = how many choices (9)
            model.add(Dense(action_space, activation='linear'))
            model.compile(loss="mse", optimizer=Adam(
                lr=0.001), metrics=['accuracy'])
            return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0]
                                  for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


# For more repetitive results
random.seed(1)
np.random.seed(1)


running = True

# To time the time it takes to generate an image
last_time = time.time()

# 0 150 3456 2234


class MyEnv(gym.Env):
    def __init__(self, sct):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([3456, 2234]))
        # use multidiscrete later # number of controls
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self.model = DQNAgent(self)
        self.previous_observation = None

        self.monitor = {'left': 0, 'top': 70, 'width': 1728, 'height': 1117}

        self.sct = sct

    def step(self, action):
        # Conditional logic for what to do with actions
        # an example
        if action % 4 == 0:
            keyboard.tap('left')
        elif action % 4 == 1:
            keyboard.tap('right')
        elif action % 4 == 2:
            keyboard.tap('up')
        elif action % 4 == 3:
            keyboard.tap('down')

        elif action >= 4 and action < 8:
            keyboard.tap('x')
        elif action >= 8 and action < 12:
            keyboard.tap('c')
        elif action >= 12 and action < 16:
            keyboard.tap('v')

        # reward = % de rouge + difference de vie

        sct_img = self.sct.grab(self.monitor)
        shot = np.array(Image.frombytes(
            'RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))
        gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)

        self_life_count_crop = shot[100:200, 3000:3100]
        opponent_life_count_crop = shot[100:200, 3200:3300]

        self_health_crop = shot[150, 3175]
        opponent_health_crop = shot[160, 3345]

        self_life_count = pytesseract.image_to_string(self_life_count_crop, lang='eng',
                                                      config='--psm 10 --oem 3 -c tessedit_char_whitelist=-10123456789')
        opponent_life_count = pytesseract.image_to_string(opponent_life_count_crop, lang='eng',
                                                          config='--psm 10 --oem 3 -c tessedit_char_whitelist=-10123456789')

        self_health = self_health_crop[2] / sum(self_health_crop)
        opponent_health = opponent_health_crop[2] / sum(opponent_health_crop)

        if self_life_count == '':
            self_life_count = 3

        if opponent_life_count == '':
            opponent_life_count = 3

        reward = (int(self_life_count) - int(opponent_life_count)) * \
            255 + (self_health - opponent_health) * 255

        if self.previous_observation is not None:
            self.model.update_replay_memory(
                (self.previous_observation, action, reward, gray))

        self.previous_observation = gray

        if self_life_count == 0 or opponent_life_count == 0:
            self.reset()

        observation = gray

        return observation, action, reward, {}

    def reset(self):
        print('RESET')
        sys.exit()
        # reset the game (re-open it, or something like that)


print('Setup starting...')
env = None

with mss() as sct:
    env = MyEnv(sct)

print('Setup complete...')
epsilon = 0.1
decay = 0.99998
min = 0.001
steps = 1

for x in range(10, 0, -1):
    print(x)
    time.sleep(1)

print('Go!')

for i in range(0, steps):
    if random.random() < epsilon:
        env.step(env.action_space.sample())
        epsilon *= decay
    else:
        env.step(env.model.get_qs(env.previous_observation))

    print(f'Step {i} processed in {time.time() - last_time} seconds')
    last_time = time.time()


env.reset()
# env.model.save('models/player.h5')
# close the game here
# ...
