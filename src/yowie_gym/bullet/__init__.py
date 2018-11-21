import numpy as np
import os
import logging
import itertools

from gym.envs.registration import register

register(
	id='Simloc-Reacher-v0',
	entry_point='yowie_gym.bullet.reacher.env:ReacherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
	)


register(
	id='Simloc-Reacher-v1',
	entry_point='yowie_gym.bullet.reacher2.env:ReacherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
	)
