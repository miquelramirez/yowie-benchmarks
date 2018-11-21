"""
    AUV joint operations task

    ScottyActivity: Mixed Discrete-Continuous Planning with Convex Optimization
    Fernandez-Gonzalez, Williams and Karpas
    JAIR 2018
"""

import numpy as np
import os
import logging
import itertools

from gym.envs.registration import register
_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))

h = 0.5

task_name = 'AUV-Navigation-v1'

register(
    id=task_name,
    entry_point='envs.tarski.auv_navigation.env:Environment',
    max_episode_steps=1000,
    kwargs={
        'I': np.array([1000.0, 500.0, 80.0]).reshape((3,1)),
        'lb_u': np.array([-2.0, -2.0]),
        'ub_u': np.array([2.0, 2.0]),
        'horizon': 1000,
        'k': 0.01,
        'v_max': 5.14, # 10 knots
        'dt': h
    }
)
