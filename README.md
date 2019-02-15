## Deep Q-learning with Flappy bird

_Mini project: Implementation of Deep Q-learning algorithm to create an AI agent capable of scoring high on Flappy Bird game_

The algorithm implemented is fully described in this paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (_Deep Q-learning with experience replay_, page 5)

A demo video can be found here:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/8OcF7RKr5e4/0.jpg)](https://www.youtube.com/watch?v=8OcF7RKr5e4)

#### Package dependencies

* gym
* gym_ple
* Pygame-Development-Environment
* Pygame
* keras

#### Experience

To train from scratch, just run the command `python flappy_bird.py -r`

You can also download [pretrained weights]() to `./model` folder in the current dir and run the command `python flappy_bird.py -l  -r` to see how the trained AI plays.

The training is undertaken on a machine with GTX860 gpu. So you need a GPU to reproduce the results.
