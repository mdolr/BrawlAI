import os
import time
import random
import numpy as np
import tensorflow as tf
from collections import deque
# from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.models import Sequential
from custom_env.image_capture import ImageCaptureEnv
from utils.modified_tensorboard import ModifiedTensorBoard

os.environ['TF_KERAS'] = '1'

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Model parameters
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1  # 50_000  # How many last steps to keep for model training
# Minimum number of steps in a memory to start training
MIN_REPLAY_MEMORY_SIZE = 1  # 1_000
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'BOX'

ELIPSON_DECAY = 0.999988877665
MIN_EPSILON = 0.0001

ep_rewards = [-200]

HEIGHT = 2234
WIDTH = 3456

steps = 5


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

            observation_space = steps, WIDTH, HEIGHT, 1
            action_space = self.env.action_space.n

            model.add(Conv2D(32, (3, 3), activation='relu',
                      input_shape=observation_space[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            """
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            """

            # this converts our 3D feature maps to 1D feature vectors
            model.add(Flatten())

            # ACTION_SPACE_SIZE = how many choices (9)
            model.add(Dense(action_space, activation='linear'))
            model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
            return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        x = np.array(state).reshape(-1, *state.shape, 1)/255
        print(x.shape)
        return self.model.predict(x)[0]

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

epsilon = 0.1
decay = 0.99998
min = 0.001

env = ImageCaptureEnv(HEIGHT, WIDTH, DQNAgent)

for x in range(3, 0, -1):
    print(x)
    time.sleep(1)

print('Go!')
last_time = time.time()

for i in range(0, steps):
    print(f'Step {i}')
    rand = random.random()
    if rand < epsilon or env.previous_observation is None:
        print('Exploration')
        env.step(env.action_space.sample())
        epsilon *= decay
    else:
        print('Prediction')
        env.step(env.model.get_qs(env.previous_observation))

    print(f'Step {i} processed in {time.time() - last_time} seconds')
    last_time = time.time()


env.reset()
