"""
    "Simple" Car model, car parking task
"""
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

from envs.errors import NumericError

class Environment(Tarski_Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):
        super(Environment, self).__init__(**kwargs)

    def define_task(self):
        d, _ = self.parameters['G'].shape
        p = self.parameters['gains_u'].shape
        self.language = tarski.language('car parking', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])

        # State variables
        x = self.language.function('x', self.language.Real)
        y = self.language.function('y', self.language.Real)
        theta = self.language.function('theta', self.language.Real)
        v = self.language.function('v', self.language.Real)

        # Control inputs
        omega = self.language.function('omega', self.language.Real)
        a = self.language.function('a', self.language.Real)

        # constants
        length = self.parameters['d']

        self.X = [x(), y(), theta(), v()]
        self.U = [omega(), a()]

        self.M0 = self.create_model()
        # back wheels to center of gravity
        B = 0.3 * length
        # front wheels to center of gravity
        A = 0.7 * length
        self.task = dl.DDP_Task("Car Parking")
        self.task.add_dynamic_constraint(self.language,
            x(), (cos(theta()) - sin(B * 1.6 * (sin(omega())/length))*sin(theta()))*v())
        self.task.add_dynamic_constraint(self.language,
            y(), (sin(theta()) - sin(B * 1.6 * (sin(omega())/length))*cos(theta()))*v())
        self.task.add_dynamic_constraint(self.language,
            theta(), sin(1.6* (v()/length) * sin(omega())))
        self.task.add_dynamic_constraint(self.language,
            v(), a())

        # MRJ: this is not strictly necessary
        dynamics = [\
            (cos(theta()) - sin(1.6* B * (sin(omega())/length))*sin(theta()))*v(),\
            (sin(theta()) - sin(1.6 * B * (sin(omega())/length))*cos(theta()))*v(),\
            sin(1.6 * (v()/length) * sin(omega())),\
            a()
        ]
        self.ode = self.language.vector(dynamics, self.language.Real)

        # constraints
        lb_u = self.parameters['lb_u']
        ub_u = self.parameters['ub_u']
        self.control_constraints = (omega() >= lb_u[0]) &\
            (omega() <= ub_u[0]) & (a() >= lb_u[1]) & (a() <= ub_u[1])

        self.task.add_input(self.language, omega(), lb_u[0], ub_u[0])
        self.task.add_input(self.language, a(), lb_u[1], ub_u[1])

        # MRJ: @TODO: Initializer properly control bounds
        #for j in range(self.action_space.shape[0]):
        #    self.action_space.low[j,0] = lb_u[j]
        #    self.action_space.high[j,0] = lb_u[j]


        # cost function
        # cost function
        zxp = huber(x(), 0.01)#sqrt(x()*x() + 0.01*0.01) - 0.01
        zyp = huber(y(), 0.01)#sqrt(y()*y() + 0.01*0.01) - 0.01
        zthetap = huber(theta(), 0.01)#sqrt(theta()*theta() + 0.01*0.01 ) - 0.01
        zvp = huber(v(), 1.0)#sqrt(v()*v() + 1.0) - 1.0

        self.terminal_cost = 0.1* (zxp + zyp) + zthetap + 0.3 * zvp
        self.task.set_terminal_cost(self.language,  self.terminal_cost)
        zxp_t = huber(x(), 0.1)
        zyp_t = huber(y(), 0.1)
        self.state_cost = 0.001 * (zxp_t + zyp_t)
        self.control_cost = (0.01 * omega()*omega()) + (1e-4 * a()*a())
        self.stage_cost = self.state_cost + self.control_cost
        self.task.set_stage_cost(self.language, self.state_cost + self.control_cost )

        self.task.compile()
        self.solver = dl.DDPSolver(self.task.sys, self.task.cost_func, False)
        self.solver.T = 50
        self.solver.dt = self.parameters['dt']
        self.solver.max_iterations = 3
        self.solver.xG = np.zeros((4,1)).tolist()

        # initial control initialisation
        self.U0 = []
        for k in range(self.solver.T):
            self.U0 += [np.clip(0.1*np.random.normal([0.0, 0.0], [1.0, 1.0]).reshape(2,1), lb_u, ub_u).tolist()]
        #print(U0[-1])


    def set_initial_state(self):
        if self.parameters['random_init']:
            index = np.random.choice(np.arange(len(self.parameters['I'])))
            self.state = self.parameters['I'][index]
            #print(self.state)
        else:
            self.state = self.parameters['I'][0]

    def reset(self):
        s = super().reset()
        self.solver.start()
        return s

    def mpc(self, x):
        self.solver.initialise_control(self.U0)
        self.solver.x0 = x.tolist()
        self.solver.start()
        self.solver.solve()
        X, U, J, i = self.solver.trajectory
        self.U0 = U[1:] + [U[-1]]
        return np.array(U[0]).reshape(self.action_space.shape), J

    def step(self, u):
        current = self.get_current_model()
        for k, u_k in enumerate(self.U):
            current.setx(u_k, u[k,0])

        # MRJ: controls outside of the constraints are rejected
        if not current[self.control_constraints]:
            for u_k in self.U:
                current.setx(u_k, 0.0)
        #print (self.dt, self.state.tolist(), u.tolist())
        u = u.reshape(self.action_space.shape)
        next = self.task.sys.next_state(self.dt, self.state.tolist(), u.tolist() )
        self.state = np.array(next).reshape(self.observation_space.shape)
        self.time += 1
        lb_x = np.array(self.parameters['lb_x']).reshape(self.observation_space.shape)
        ub_x = np.array(self.parameters['ub_x']).reshape(self.observation_space.shape)
        is_terminal = (self.time > self.horizon)\
            or np.any(np.less(self.state, lb_x))\
            or np.any(np.greater(self.state, ub_x))

        if is_terminal:
            try:
                R = -current[self.terminal_cost].symbol
            except ValueError as e:
                orig_msg = str(e)
                msg = 'car_park.Environment.step(): Numeric error evaluating terminal cost:\n'
                msg += 'see original exception message:\n{}'.format(orig_msg)
                msg += 'Current interpretation of variables:\n'
                for x in self.X:
                    msg += '{} = {}\n'.format(x, current[x].symbol)
                for u in self.U:
                    msg += '{} = {}\n'.format(u, current[u].symbol)
                msg += 'State:\n{}'.format(self.state)

                raise NumericError(msg)
        else:
            try:
                R = -current[self.stage_cost].symbol
            except ValueError as e:
                orig_msg = str(e)
                msg = 'car_park.Environment.step(): Numeric error evaluating stage cost:\n'
                msg += 'see original exception message:\n{}'.format(orig_msg)
                msg += 'Current interpretation of variables:\n'
                for x in self.X:
                    msg += '{} = {}\n'.format(x, current[x].symbol)
                for u in self.U:
                    msg += '{} = {}\n'.format(u, current[u].symbol)
                msg += 'State:\n{}'.format(self.state)
                raise NumericError(msg)

        return self.state, R, is_terminal, {}
