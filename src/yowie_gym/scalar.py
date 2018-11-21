"""
    Simple Scalar Linear System

    example problem and system discussed in

    "Dynamic Programming and Optimal Control"
    Dmitri P. Bertsekas, 4th Edition, Athena Scientific, 2017
    Example 6.7.1, pp.398-399

    Adapted by Miquel Ramirez, 2018
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class Bertsekas671_Env(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):

        self.viewer = None

        self.a = 2.5
        self.b = 0.2

        self.d = 1
        self.p = 1
        self.time = 0

        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(self.p,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(self.d, ), dtype=np.float64)

        self.w_mu = kwargs['w_mu']
        self.w_sigma = kwargs['w_sigma']
        self.x0 = kwargs['x0']
        self.state = self.x0


        self.seed()


    @staticmethod
    def optimal(self):
        """
            Returns optimal control u for current state
        """
        return -1.0 * (self.a/self.b)*self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):

        x = self.state

        cost = np.power(x,2.0)
        # x_{k+1} = a x_k + b u_k + w_k
        new_x = self.a * x + self.b * u + self.np_random.normal(self.w_mu,self.w_sigma,size = self.d)
        self.state = new_x

        self.time += 1

        return self.get_obs(), - cost, self.time > 300, {}

    def reset(self):
        self.state = self.x0
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.state

    def get_params(self):
        return self.a, self.b

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer: self.viewer.close()
