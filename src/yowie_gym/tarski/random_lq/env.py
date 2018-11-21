"""
    Linear System with Quadratic Cost
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

import dilithium as dp

class Environment(Tarski_Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):
        super(Environment, self).__init__(**kwargs)

    def define_task(self):
        d, p = self.parameters['fu'].shape
        self.language = tarski.language('random_lq', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])
        for i in range(d):
            x_i = self.language.function('x_{}'.format(i), self.language.Real)
            self.X += [x_i()]
        for i in range(p):
            u_i = self.language.function('u_{}'.format(i), self.language.Real)
            self.U += [u_i()]

        self.M0 = self.create_model()

        x = self.language.vector(self.X, self.language.Real)
        u = self.language.vector(self.U, self.language.Real)
        fx = self.language.matrix(self.parameters['fx'], self.language.Real)
        fu = self.language.matrix(self.parameters['fu'], self.language.Real)
        lxx = self.language.matrix(self.parameters['lxx'], self.language.Real)
        luu = self.language.matrix(self.parameters['luu'], self.language.Real)
        lf = self.language.matrix(self.parameters['lf'], self.language.Real)

        self.stage_cost = 0.5 * (transpose(x) * lxx * x + transpose(u) * luu * u)
        self.terminal_cost = transpose(x) * lf * x

        self.ode = [ fx * x + fu * u]
        bounds = []
        u_ub = []
        u_lb = []
        for i in range(p):
            bounds += [self.U[i] >= -1.0]
            bounds += [self.U[i] <= 1.0]
            u_lb.append([-1.0])
            u_ub.append([1.0])

        self.control_constraints = land(*bounds)

        # Dilithium stuff setup
        self.sys = dp.LTISystem(d,p)
        self.sys.set_matrices(self.parameters['fx'].tolist(), self.parameters['fu'].tolist())
        self.sys.set_control_bounds(u_lb, u_ub)

        self.cost = dp.QRCost(d,p)
        self.cost.Q = self.parameters['lxx'].tolist()
        self.cost.R = self.parameters['luu'].tolist()

        self.solver = dp.DDPLQRSolver(self.sys, self.cost)
        self.solver.T = 10
        self.solver.dt = self.parameters['dt']
        self.solver.max_iterations = 4
        self.solver.xG = np.zeros((d,1)).tolist()

    def set_initial_state(self):
        if self.parameters['random_init']:
            self.state = np.random.normal(0.0, 1.0, size=(self.observation_space.shape[0],1))
        else:
            self.state = self.parameters['x0']

    def mpc(self, x):
        self.solver.x0 = x.tolist()
        self.solver.start()
        self.solver.solve()
        X, U, J = self.solver.trajectory
        return np.array(U[0]).reshape(self.action_space.shape), J

    def step(self, u):
        current = self.get_current_model()
        for k, u_k in enumerate(self.U):
            current.setx(u_k, u[k,0])

        # MRJ: controls outside of the constraints are rejected
        if not current[self.control_constraints]:
            for u_k in self.U:
                current.setx(u_k, 0.0)

        # MRJ: Euler integration: TODO: use generic integrator
        u = u.reshape(self.action_space.shape)
        next = self.sys.next_state(self.dt, self.state.tolist(), u.tolist() )
        self.state = np.array(next).reshape(self.observation_space.shape)
        self.time += 1

        is_terminal = self.time > self.horizon

        if is_terminal:
            R = -current[self.terminal_cost][0,0].symbol
        else:
            #print(current[self.stage_cost])
            R = -current[self.stage_cost][0,0].symbol

        return self.state, R, is_terminal, {}
