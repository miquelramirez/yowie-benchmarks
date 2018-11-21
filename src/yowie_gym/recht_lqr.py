"""
    LQR benchmark problem

    Benjamin Recht, 2018

    Adapted by Miquel Ramirez, from the ARS code here
    https://github.com/modestyachts/ARS
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class LQR_Env(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.viewer = None

        self.A =  np.array([[1.01, 0.01, 0.0],[0.01, 1.01, 0.01], [0., 0.01, 1.01]])
        self.B = np.eye(3)

        self.d, self.p = self.B.shape

        self.R = np.eye(self.p) * 1000
        self.Q = np.eye(self.d)

        self.time = 0

        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(self.p,1),dtype=np.float64)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.d, 1),dtype=np.float64)

        self.state = np.random.normal(0,1,size = self.d)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):

        x = self.state

        cost = np.dot(x.T, np.dot(self.Q, x)) + np.dot(u.T, np.dot(self.R, u))
        new_x = np.dot(self.A, x) + np.dot(self.B, u) + self.np_random.normal(0,1,size = (self.d,1))

        self.state = new_x

        terminated = False
        if self.time > 300:
            terminated = True

        self.time += 1

        return self.get_obs(), - cost, terminated, {}

    def reset(self):
        self.state = self.np_random.normal(0, 1, size = (self.d,1))
        self.last_u = None
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.state

    def get_params(self):
        return self.A, self.B, self.Q, self.R

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer: self.viewer.close()
