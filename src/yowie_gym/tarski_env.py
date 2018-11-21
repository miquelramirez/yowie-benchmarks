"""
Tarski based environment
"""

import numpy as np

import tarski
from tarski.theories import Theory
from tarski.syntax import *
from tarski.io import rddl
from tarski.model import Model
from tarski.evaluators.simple import evaluate
from tarski.syntax.arithmetic import *
from tarski.syntax.arithmetic.special import *
from tarski.syntax.arithmetic.random import *

import gym
from gym import spaces

class Tarski_Env(gym.Env):

    metadata = None # needs to be defined by subclasses

    def __init__(self, **kwargs):

        # MRJ: visualization
        self.viewer = None
        self.language = None
        self.parameters = kwargs
        # state variables
        self.X = []
        # control inputs
        self.U = []
        # constants
        self.constants = []
        # constraints
        self.ode = []
        self.perturbation = []
        self.state_invariants = []
        self.control_constraints = []
        # cost function
        self.stage_cost = None
        self.terminal_cost = None
        self.M0 = None
        self.define_task()

        self.time = 0
        self.horizon = self.parameters['horizon']
        self.dt = kwargs['dt']
        self.d = len(self.X)
        self.p = len(self.U)
        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(self.p,1),dtype=np.float64)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.d, 1),dtype=np.float64)

        self.state = np.zeros((self.d, 1))


    def create_model(self):
        """
            Creates model for current language
        """
        assert self.language is not None
        m = Model(self.language)
        m.evaluator = evaluate
        return m

    def get_current_model(self):
        """
            Returns current model synchronised with current state
            and constants
        """
        m_s = self.create_model()
        for k, x_k in enumerate(self.X):
            m_s.setx(x_k, self.state[k])
        for c in self.constants:
            m_s.setx(c, self.M0[c].symbol)
        return m_s


    def set_initial_state(self, **kwargs):
        """
            Sets the initial state
        """
        self.state = np.random.normal(0,1,size = self.d)

    def define_task(self):
        raise NotImplementedError("Needs to be defined by subclasses")

    def optimal_control(self):
        """
            Returns optimal control for current state (when available) otherwise
            will raise NotImplementedError()
        """
        raise NotImplementedError("Needs to be defined by subclasses")

    def reset(self):
        self.time = 0
        self.set_initial_state()
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, u):
        pass

    def get_obs(self):
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer: self.viewer.close()
