from pybullet_envs.robot_bases import MJCFBasedRobot
import numpy as np
import os

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))

class Reacher(MJCFBasedRobot):
	TARG_LIMIT = 0.27

	def __init__(self):
		MJCFBasedRobot.__init__(self, os.path.join(_CURRENT_DIR_, 'reacher.xml'), 'body0', action_dim=2, obs_dim=8)

	def robot_specific_reset(self, bullet_client):
		self.jdict["target_x"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.jdict["target_y"].reset_current_position(
			self.np_random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
		self.fingertip = self.parts["fingertip"]
		self.target = self.parts["target"]
		self.central_joint = self.jdict["joint0"]
		self.elbow_joint = self.jdict["joint1"]
		self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

	def apply_action(self, a):
		assert (np.isfinite(a).all())
		self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
		self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

	def calc_state(self):
		self.theta, self.theta_dot = self.central_joint.current_relative_position()
		self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
		target_x, _ = self.jdict["target_x"].current_position()
		target_y, _ = self.jdict["target_y"].current_position()
		self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())

		effector_pos = np.array(self.fingertip.pose().xyz())
		target_pos = np.array(self.target.pose().xyz())
		angles = np.array([self.theta, self.gamma, self.theta_dot, self.gamma_dot])

		return np.hstack( (effector_pos, target_pos, angles) ).reshape(10,1)

	def stage_cost(self):
		return np.linalg.norm(self.to_target_vec)
