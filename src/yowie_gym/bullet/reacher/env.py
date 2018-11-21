from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
from .reacher import Reacher


class ReacherBulletEnv(MJCFBaseBulletEnv):
	def __init__(self):
		self.robot = Reacher()
		MJCFBaseBulletEnv.__init__(self, self.robot)

	def create_single_player_scene(self, bullet_client):
		return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

	def _step(self, a):
		assert (not self.scene.multiplayer)
		self.robot.apply_action(a)
		self.scene.global_step()

		state = self.robot.calc_state()  # sets self.to_target_vec

		potential_old = self.potential
		self.potential = self.robot.calc_potential()

		electricity_cost = (
			-0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot))  # work torque*angular_velocity
			- 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
		)
		stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
		self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
		self.HUD(state, a, False)
		return state, sum(self.rewards), False, {}

	def camera_adjust(self):
		x, y, z = self.robot.fingertip.pose().xyz()
		x *= 0.5
		y *= 0.5
		self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
