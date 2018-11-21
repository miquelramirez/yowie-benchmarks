from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from gym import spaces

import numpy as np
from .reacher import Reacher

import tarski
from tarski.theories import Theory
from tarski.syntax import *
from tarski.io import rddl
from tarski.model import Model
from tarski.evaluators.simple import evaluate
from tarski.syntax.arithmetic import *
from tarski.syntax.arithmetic.special import *
from tarski.syntax.arithmetic.random import *

from yowie_gym.utils import huber

class ReacherBulletEnv(MJCFBaseBulletEnv):
    def __init__(self):
        self.robot = Reacher()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.dt = 0.01 # @TODO: this should be available from the robot model
        self._cam_dist = 0.5
        self.parameters = {}
        self.language = tarski.language('reacher 2D', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])

        fp = self.language.constant('fp', self.language.Object)
        t = self.language.constant('t', self.language.Object)

        # State variables
        x = self.language.function('x', self.language.Object, self.language.Real)
        y = self.language.function('y', self.language.Object, self.language.Real)
        z = self.language.function('z', self.language.Object, self.language.Real)
        theta = self.language.function('theta', self.language.Real)
        gamma = self.language.function('gamma', self.language.Real)
        theta_dot = self.language.function('theta_dot', self.language.Real)
        gamma_dot = self.language.function('gamma_dot', self.language.Real)

        central_joint_torque = self.language.function('cj', self.language.Real)
        elbow_joint_torque = self.language.function('ej', self.language.Real)

        self.X = [x(fp), y(fp), z(fp), x(t), y(t), z(t), theta(), gamma(), theta_dot(), gamma_dot()]
        self.U = [central_joint_torque(), elbow_joint_torque() ]

        self.parameters['lb_u'] = [-1, -1]
        self.parameters['ub_u'] = [1, 1]
        self.parameters['lb_x'] = np.array([-10.,-10.,-1.,-0.27,-0.27,-0.1,-np.pi,-np.pi,-np.pi,-np.pi])
        self.parameters['ub_x'] = np.array([10.,10.,1.,0.27,0.27,0.1,np.pi,np.pi,np.pi,np.pi])
        self.parameters['dt'] = self.dt

        dxt = huber(x(fp)-x(t), 0.1)
        dyt = huber(y(fp)-y(t), 0.1)
        dzt = huber(z(fp)-z(t), 0.1)
        self.terminal_cost = dxt + dyt + dzt

        cu00 = central_joint_torque()*theta_dot()
        cu01 = elbow_joint_torque()*gamma_dot()
        cu0 = (cu00 * cu00) + (cu01 * cu01)
        cu1 = 0.01 * (central_joint_torque()*central_joint_torque() + elbow_joint_torque() * elbow_joint_torque())
        control_cost = cu0 + cu1
        self.stage_cost = (0.01 * self.terminal_cost) + control_cost

        self.d = len(self.X)
        self.p = len(self.U)
        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(self.p,1),dtype=np.float64)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.d, 1),dtype=np.float64)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def _step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        self.state = self.robot.calc_state()  # sets self.to_target_vec

        # electricity cost
        cu00 = a[0] * self.robot.theta_dot
        cu01 = a[1] * self.robot.gamma_dot
        cu0 = (cu00 * cu00) + (cu01 * cu01)
        cu1 = 0.01 * (a[0]*a[0] + a[1]*a[1])
        lu = cu0 + cu1

        lx = self.robot.stage_cost()
        self.reward = -(lx + lu)
        self.HUD(self.state, a, False)
        lb_x = np.array(self.parameters['lb_x']).reshape(10,1)
        ub_x = np.array(self.parameters['ub_x']).reshape(10,1)
        is_terminal = np.any(np.less(self.state, lb_x))\
            or np.any(np.greater(self.state, ub_x))
        return self.state, self.reward, False, {}

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
