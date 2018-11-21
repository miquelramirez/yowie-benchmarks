"""
    Simple Scalar Linear System

    example problem and system discussed in

    "Dynamic Programming and Optimal Control"
    Dmitri P. Bertsekas, 4th Edition, Athena Scientific, 2017
    Example 6.7.1, pp.398-399

    Adapted by Miquel Ramirez, 2018
"""

from envs.tarski_env import Tarski_Env

import tarski
from tarski.theories import Theory
from tarski.syntax import *
from tarski.io import rddl
from tarski.model import Model
from tarski.evaluators.simple import evaluate
from tarski.syntax.arithmetic import *
from tarski.syntax.arithmetic.special import *
from tarski.syntax.arithmetic.random import *

class Environment(Tarski_Env):

    def __init__(self, **kwargs):
        super(Environment,self).__init__(**kwargs)

    def define_task(self):
        self.language = tarski.language('bertsekas_671', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])
        self.a = self.language.function('a', self.language.Real)
        self.b = self.language.function('b', self.language.Real)
        self.x = self.language.function('x', self.language.Real)
        self.u = self.language.function('u', self.language.Real)
        self.mu_w = self.language.function('mu_w', self.language.Real)
        self.sigma_w = self.language.function('sigma_w', self.language.Real)

        self.M0 = self.create_model()

        self.stage_cost = self.x() * self.x()

        self.X = [self.x()]
        self.U = [self.u()]
        self.constants = [self.a(), self.b(), self.mu_w(), self.sigma_w()]

        self.M0.setx(self.a(), 2.5)
        self.M0.setx(self.b(), 0.2)
        self.M0.setx(self.mu_w(), self.parameters['mu_w'])
        self.M0.setx(self.sigma_w(), self.parameters['sigma_w'])

        # dynamics
        self.ode = [\
            self.a() * self.x() + self.b() * self.u()
        ]
        self.perturbation = [\
            normal(self.mu_w(), self.sigma_w())
        ]

    def set_initial_state(self):
        self.state[0] = self.parameters['x0']

    def optimal_control(self):
        return [-self.M0[self.a()/self.b()].symbol * self.state[0]]

    def step(self, u):
        current = self.get_current_model()
        for k, u_k in enumerate(self.U):
            current.setx(u_k, u[k])

        next_state = np.zeros((self.d,1))
        next_state[0] = self.state[0] + self.dt * current[self.ode[0]].symbol + current[self.perturbation[0]].symbol
        self.state = next_state
        self.time += 1

        reward = -current[self.stage_cost].symbol

        return self.state, reward, self.time > self.horizon, {}
