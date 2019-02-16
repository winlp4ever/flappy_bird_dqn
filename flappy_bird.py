import logging

import numpy as np
import random
import os, sys
import gym
from gym.wrappers import Monitor
import gym_ple
import keras
from keras.models import Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.optimizers import Adam
import cv2
import argparse

import matplotlib.pyplot as plt

from skimage import color


from collections import deque

def to_gray(img):
    img =  0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    img[img < 80] = 0
    img[img >= 80] = 255
    img = 255 - img
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = img[:54,4:]
    return img

class Agent():
    def __init__(self, action_size, weight_backup=None, save_weight='./model/flappybird_weight.h5'):
        self.weight_backup = weight_backup
        self.save_weight = save_weight
        self.memory = deque(maxlen=50000)

        self.observe = 10000

        self.explore = 1e5
        self.INITIAL_EXPLORE = 0.1
        self.exploration_rate = self.INITIAL_EXPLORE
        self.exploration_min = 0.0001
        self.learning_rate = 1e-6
        self.gamma = 0.99

        self.action_size = action_size
        self.input_shape = (54, 32, 4)
        self.brain = self._build_model()



    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if self.weight_backup and os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min

        model.summary()
        return model


    def save_model(self):
        self.brain.save(self.save_weight)



    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        # network
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def replay(self, sample_batch_size):
        if len(self.memory) < self.observe:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                u = self.brain.predict(x = next_state)
                target = reward + self.gamma * np.amax(u[0])
            # network
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate -= (self.INITIAL_EXPLORE - self.exploration_min)/self.explore


class FlappyBird:
    def __init__(self, weight_backup=None):
        self.sample_batch_size = 32
        self.episodes = 200000
        self.env = gym.make('FlappyBird-v0')

        self.action_size = self.env.action_space.n

        self.agent = Agent(self.action_size, weight_backup)

    def run(self, render=False):
        try:
            plot = []
            max = 0
            avg = 0
            for index_episode in range(self.episodes):
                frame = self.env.reset()
                frame = to_gray(frame)
                state = np.stack((frame, frame, frame, frame), axis=2)
                state = state.reshape((1, 54, 32, 4))

                reward = 0
                acc_reward = 0
                done = False
                while not done:
                    acc_reward += reward

                    if render:
                        self.env.render()

                    action = self.agent.act(state)

                    next_frame, reward, done, _ = self.env.step(action)
                    next_frame = to_gray(next_frame)
                    next_frame = next_frame.reshape((1, 54, 32, 1))

                    next_state = np.append(next_frame, state[:,:,:,:3], axis=3)

                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state

                if acc_reward > max:
                    max = acc_reward
                avg = (avg * index_episode + acc_reward) / (index_episode + 1)
                plot.append(acc_reward)

                print("Episode {}# Score: {} - max: {} - average: {})".format(index_episode, acc_reward, max, avg), flush=True, end='\r')

                self.agent.replay(self.sample_batch_size)

                if index_episode%100 == 99:
                    self.agent.save_model()
        finally:
            self.env.close()
            print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained-weights', '-l', help='decide whether load trained weights or not.',
                        default=None, const='./model/flappybird_weight.h5', nargs='*')
    parser.add_argument('--render', '-r', help='decide whether render video or not.', type=bool,
                        default=False, const=True, nargs='*')

    args = parser.parse_args()

    flappy = FlappyBird(args.trained_weights);
    flappy.run(args.render)
