"""
    AUV recovery task, after Fernandez-Gonzalez et al 2018
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

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):
        super(Environment, self).__init__(**kwargs)

    def define_task(self):
        self.language = tarski.language('random_lq', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])

        # Sorts
        vehicle = self.language.sort('vehicle', self.language.Object)
        auv = self.language.sort('AUV', vehicle)

        # Objects
        auv1 = self.language.constant('AUV_1', auv)

        # State variables
        x = self.language.function('x', vehicle, self.language.Real)
        y = self.language.function('y', vehicle, self.language.Real)
        b = self.language.function('b', vehicle, self.language.Real)

        # Control inputs
        vx = self.language.function('vx', vehicle, self.language.Real)
        vy = self.language.function('vy', vehicle, self.language.Real)

        # constants
        k = self.parameters['k']
        v_max = self.parameters['v_max']

        self.X = [x(auv1), y(auv1), b(auv1)]
        self.U = [vx(auv1), vy(auv1)]

        self.M0 = self.create_model()

        # dynamics
        fxu = [\
            vx(auv1),
            vy(auv1),
            -k * sqrt(vx(auv1)*vx(auv1) + vy(auv1)*vy(auv1))
        ]
        self.ode = self.language.vector(fxu, self.language.Real)
        
        # constraints
        lb_u = self.parameters['lb_u']
        ub_u = self.parameters['ub_u']
        self.control_constraints = (vx(auv1) >= lb_u[0]) &\
            (vx(auv1) <= ub_u[0]) & (vy(auv1) >= lb_u[1]) & (vy(auv1) <= ub_u[1])

        vx2 = vx(auv1)*vx(auv1)
        vy2 = vy(auv1)*vy(auv1)

        self.state_invariants = ((vx2 + vy2) <= v_max) & (b(auv1) >= 0)

        # const function
        # cost function
        zxp = sqrt(x(auv1)*x(auv1) + 0.1*0.1) - 0.1
        zyp = sqrt(y(auv1)*y(auv1) + 0.1*0.1) - 0.1
        zvxp = sqrt(vx2 + 1.0) - 1.0
        zvyp = sqrt(vy2 + 1.0) - 1.0
        self.terminal_cost = zxp + zyp + zvxp + zvyp

        state_cost = 0.01 * (zxp + zyp)
        control_cost = vx(auv1)*vx(auv1) + vy(auv1)*vy(auv1)
        self.stage_cost = state_cost + control_cost

    def set_initial_state(self):
        self.state = self.parameters['I']

    def step(self, u):
        current = self.get_current_model()
        for k, u_k in enumerate(self.U):
            current.setx(u_k, u[k])

        # MRJ: controls outside of the constraints are rejected
        if not current[self.control_constraints]:
            for u_k in self.U:
                current.setx(u_k, 0.0)

        # MRJ: Euler integration: TODO: use generic integrator
        sym_ode = current[self.ode]
        values = np.zeros(sym_ode.shape)
        for i in range(sym_ode.shape[0]):
            for j in range(sym_ode.shape[1]):
                values[i, j] = sym_ode[i, j].symbol
        self.state = self.state + self.dt * values
        self.time += 1

        is_terminal = self.time > self.horizon

        if is_terminal:
            R = -current[self.terminal_cost].symbol
        else:
            R = -current[self.stage_cost].symbol

        return self.state, R, is_terminal, {}

    def render(self, mode, close=False):
        return np.array([])
