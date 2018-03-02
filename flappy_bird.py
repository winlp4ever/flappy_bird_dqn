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
from collections import deque

class Agent():
    def __init__(self, action_space, observation_space):
        self.weight_backup = "flappybird_weight.h5"
        self.memory = deque(maxlen=2000)

        self.exploration_rate = 0.5
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95

        self.action_space = action_space
        self.input_shape = np.shape(observation_space)
        self.brain = self._build_model()



    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.action_space.n, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min

        return model


    def save_model(self):
        self.brain.save(self.weight_backup)



    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return self.action_space.sample()
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class FlappyBird:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 1000
        self.env = gym.make('FlappyBird-v0')

        self.action_size = self.env.action_space.n
        self.agent = Agent(self.env.action_space, self.env.observation_space)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()

                index = 0
                done = False
                while not done:
                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

if __name__ == '__main__':
    flappy = FlappyBird();
    flappy.run()