"""
    UAV platooning, dynamics as described in

    Integrated Hybrid Planning and Programmed Control for Real-Time UAV Maneuvering
    Ramirez et al, AAMAS 2018
"""
import matplotlib.pyplot as plt

from envs.tarski_env import Tarski_Env
from envs.utils import huber

import tarski
from tarski.theories import Theory
from tarski.syntax import *
from tarski.io import rddl
from tarski.model import Model
from tarski.evaluators.simple import evaluate
from tarski.syntax.arithmetic import *
from tarski.syntax.arithmetic.special import *
from tarski.syntax.arithmetic.random import *

import dilithium as dl

class Environment(Tarski_Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):
        super(Environment, self).__init__(**kwargs)
        self.red_trace = []
        self.blue_trace = []

    def define_task(self):
        self.language = tarski.language('uav_platooning',  [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])

        # State variables
        uav = self.language.sort('uav', self.language.Object)
        b = self.language.constant('b', uav)
        r = self.language.constant('r', uav)

        x = self.language.function('x', uav, self.language.Real)
        y = self.language.function('y', uav,self.language.Real)
        z = self.language.function('z', uav,self.language.Real)

        v = self.language.function('v', uav,self.language.Real)

        # yaw, pitch, and bank
        psi = self.language.function('psi', uav, self.language.Real)
        theta = self.language.function('theta', uav, self.language.Real)
        phi = self.language.function('phi', uav, self.language.Real)

        g = self.parameters['g']

        # controls
        d_theta = self.language.function('delta_theta', uav, self.language.Real)
        d_phi = self.language.function('delta_phi', uav, self.language.Real)
        d_v = self.language.function('delta_v', uav, self.language.Real)

        self.X = [x(b), y(b), z(b), psi(b), theta(b), phi(b), v(b), \
            x(r), y(r), z(r), psi(r), theta(r), phi(r), v(r)]
        self.U = [d_theta(b), d_phi(b), d_v(b)]

        self.M0 = self.create_model()

        self.task = dl.DDP_Task("UAV Platooning")
        zero = self.language.constant(0.0, self.language.Real)

        self.task.add_dynamic_constraint(self.language, x(b), v(b) * cos(psi(b)))
        self.task.add_dynamic_constraint(self.language, y(b), v(b) * sin(psi(b)))
        self.task.add_dynamic_constraint(self.language, z(b), v(b) * sin(theta(b)))
        self.task.add_dynamic_constraint(self.language, psi(b), sin(1.6*sin(phi(b)) * (g/v(b))))
        self.task.add_dynamic_constraint(self.language, theta(b), d_theta(b))
        self.task.add_dynamic_constraint(self.language, phi(b), d_phi(b))
        self.task.add_dynamic_constraint(self.language, v(b), d_v(b))
        self.task.add_dynamic_constraint(self.language, x(r), v(r) * cos(psi(r)))
        self.task.add_dynamic_constraint(self.language, y(r), v(r) * sin(psi(r)))
        self.task.add_dynamic_constraint(self.language, z(r), v(r) * sin(theta(r)))
        self.task.add_dynamic_constraint(self.language, psi(r), sin(1.6*sin(phi(r)) * (g/v(r))))
        self.task.add_dynamic_constraint(self.language, theta(r), zero)
        self.task.add_dynamic_constraint(self.language, phi(r), zero)
        self.task.add_dynamic_constraint(self.language, v(r), zero)



        # dynamics
        fxu = [\
            v(b) * cos(psi(b)),
            v(b) * sin(psi(b)),
            v(b) * sin(theta(b)),
            asin(sin(phi(b)) * (g/v(b))),
            d_theta(b),
            d_phi(b),
            d_v(b),
            v(r) * cos(psi(r)),
            v(r) * sin(psi(r)),
            v(r) * sin(theta(r)),
            0.0,
            0.0,
            0.0,
            0.0
        ]

        self.ode = self.language.vector(fxu, self.language.Real)

        # constraints
        lb_u = self.parameters['lb_u']
        ub_u = self.parameters['ub_u']
        self.control_constraints = (d_theta(b) >= lb_u[0]) &\
            (d_theta(b) <= ub_u[0]) & (d_phi(b) >= lb_u[1]) &\
            (d_phi(b) <= ub_u[1]) & (d_v(b) >= lb_u[2]) &\
            (d_v(b) <= ub_u[2])

        self.task.add_input(self.language, d_theta(b), lb_u[0], ub_u[0])
        self.task.add_input(self.language, d_phi(b), lb_u[1], ub_u[1])
        self.task.add_input(self.language, d_v(b), lb_u[2], ub_u[2])

        # terminal cost
        dx = (x(b)-x(r))*(x(b)-x(r))
        dy = (y(b)-y(r))*(y(b)-y(r))
        dz = (z(b)-z(r))*(z(b)-z(r))
        zd = huber(sqrt(dx + dy + dz), 100.0)
        zv = huber(v(b)-v(r), 10.0)

        self.terminal_cost = 1e-4*zd + 1e-2*zv
        control_costs = 1e-3*d_theta(b)*d_theta(b) + 1e-3*d_phi(b)*d_phi(b) + 1e-3 *d_v(b)*d_v(b)
        self.stage_cost = 1e-5*zd + control_costs

        self.task.set_stage_cost(self.language, self.stage_cost)
        self.task.set_terminal_cost(self.language, self.terminal_cost)
        self.task.compile()

        self.solver = dl.DDPSolver(self.task.sys, self.task.cost_func, False)
        self.solver.T = 50
        self.solver.dt = self.parameters['dt']
        self.solver.max_iterations = 3
        self.solver.xG = np.zeros((len(self.X),1)).tolist()


    def set_initial_state(self):
        if self.parameters['random_init']:
            index = np.random.choice(np.arange(len(self.parameters['I'])))
            self.state = self.parameters['I'][index]
        else:
            self.state = self.parameters['I'][0]
        self.blue_trace = []
        self.red_trace = []

    def step(self, u):
        current = self.get_current_model()
        for k, u_k in enumerate(self.U):
            current.setx(u_k, u[k])

        self.blue_trace += [np.array([current[self.X[0]].symbol, current[self.X[1]].symbol])]
        self.red_trace += [np.array([current[self.X[7]].symbol, current[self.X[8]].symbol])]

        # MRJ: controls outside of the constraints are rejected
        if not current[self.control_constraints]:
            for u_k in self.U:
                current.setx(u_k, 0.0)

        u = u.reshape(self.action_space.shape)
        next = self.task.sys.next_state(self.dt, self.state.tolist(), u.tolist() )
        self.state = np.array(next).reshape(self.observation_space.shape)
        # simple perturbations
        self.state[-3,0] += np.random.normal(0.0, 1e-2)
        self.state[-2,0] += np.random.normal(0.0, 1e-2)
        self.state[-1,0] += np.random.normal(0.0, 1e-2)
        self.time += 1

        lb_x = np.array(self.parameters['lb_x']).reshape(self.observation_space.shape)
        ub_x = np.array(self.parameters['ub_x']).reshape(self.observation_space.shape)
        is_terminal = (self.time > self.horizon)\
            or np.any(np.less(self.state, lb_x))\
            or np.any(np.greater(self.state, ub_x))
        if is_terminal:
            R = -current[self.terminal_cost].symbol
        else:
            R = -current[self.stage_cost].symbol

        return self.state, R, is_terminal, {}

    def render(self, mode, close=False):
        return np.array([])
