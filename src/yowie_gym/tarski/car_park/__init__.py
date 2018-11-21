"""
    Car Parking tasks, following closely the discussion in

    Control-Limited Differential Dynamic Programming
    Yuval Tassa, Nicolas Mansard and Emo Todorov
    Proceedings of ICRA-14
"""

import numpy as np
import os
import logging
import itertools

from gym.envs.registration import register
_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))

h = 0.03

X0 = np.linspace(-4.0, 4.0, num=9)
Y0 = X0
theta0 = np.array([0.0, 0.5*np.pi, np.pi, 1.5*np.pi])
v0 = np.linspace(-1.0, 1.0, num=5)

task_names = []
initial_states = []

# MRJ: v1 domains are instances where the initial state is fixed
task_name_template = 'CarPark-{}{}{}{}-v1'

for x, y, theta, v in itertools.product(X0,Y0,theta0,v0):
    x_idx = np.where(X0==x)[0][0]
    y_idx = np.where(Y0==y)[0][0]
    theta_idx = np.where(theta0==theta)[0][0]
    v_idx = np.where(v0==v)[0][0]
    task_name = task_name_template.format(x_idx, y_idx, theta_idx, v_idx)
    x0 = np.array([x, y, theta, v]).reshape((4,1))
    initial_states += [x0]
    register(\
        id=task_name,
        entry_point='yowie_gym.tarski.car_park.env:Environment',
        max_episode_steps=1000,
        kwargs={
            'I': [x0],
            'G': np.zeros((4,1)),
            'lb_x': np.array([-10.0, -10.0, -2.*np.pi, -5.0]),
            'ub_x': np.array([10.0, 10.0, 2.*np.pi, 5.0]),
            'lb_u': np.array([-0.5, -2.0]),
            'ub_u': np.array([0.5, 2.0]),
            'd': 2,
            'gains_u': np.array([0.01, 1e-4]),
            'random_init': False,
            'horizon': 1000,
            'dt':h
        }
    )
    task_names += [task_name]

# MRJ: v2 are instances where the initial state is set randomly
register(\
    id='CarPark-v2',
    entry_point='yowie_gym.tarski.car_park.env:Environment',
    max_episode_steps=1000,
    kwargs={
        'I': initial_states,
        'G': np.zeros((4,1)),
        'lb_x': np.array([-10.0, -10.0, -2.*np.pi, -5.0]),
        'ub_x': np.array([10.0, 10.0, 2.*np.pi, 5.0]),
        'lb_u': np.array([-0.5, -2.0]),
        'ub_u': np.array([0.5, 2.0]),
        'd': 2,
        'gains_u': np.array([0.01, 1e-4]),
        'random_init': True,
        'horizon': 1000,
        'dt': h
    }
)
