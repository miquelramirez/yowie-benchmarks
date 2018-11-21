"""
    UAV Platooning task

    The task requires a UAV to intercept another UAV flying on a straight line,
    and keep station behind it indefinitely.
"""
import numpy as np
import h5py
import os
import logging
import itertools

from gym.envs.registration import register

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))

h = 0.1

X0 = np.linspace(-2000, 2000, num=9)
Y0 = X0
psi0 = np.array([0.0, 0.5*np.pi, np.pi, 1.5*np.pi])
V0 = np.linspace(100, 250, num=4)

task_names = []
initial_states = []

task_name_template = 'UAV-Platooning-{}{}{}{}-v1'

for x, y, psi, v in itertools.product(X0, Y0, psi0, V0):
    x_idx = np.where(X0==x)[0][0]
    y_idx = np.where(Y0==y)[0][0]
    psi_idx = np.where(psi0==psi)[0][0]
    v_idx = np.where(V0==v)[0][0]
    task_name = task_name_template.format(x_idx, y_idx, psi_idx, v_idx)
    x0 = np.array([x, y, 10000, psi, 0, 0, v, 0, 0, 10000, 0, 0, 0, v]).reshape((14,1))
    initial_states += [x0]
    register(\
        id=task_name,
        entry_point='envs.tarski.uav_platoon.env:Environment',
        max_episode_steps=1000,
        kwargs={
            'I': [x0],
            'target_x0': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0]),
            'g': 9.81,
            'lb_u': np.array([-0.5, -0.174533, -10 ]),
            'ub_u': np.array([0.5, 0.174533, 10]),
            'lb_x': np.array([-50000.0, -50000.0, 0.0, -2.0 *np.pi, -0.5*np.pi, -1.22173, 50.0,
                        -50000.0, -50000.0, 0.0, -2.0 *np.pi, -0.5*np.pi, -1.22173, 50.0]),
            'ub_x': np.array([50000.0, 50000.0, 20000.0, 2.0 *np.pi, 0.5*np.pi, 1.22173, 680.0,
                        50000.0, 50000.0, 20000.0, 2.0 *np.pi, 0.5*np.pi, 1.22173, 680.0]),
            'gains_u': np.ones((4,1)),
            'random_init': False,
            'horizon': 1000,
            'dt':h
        }
    )
    task_names += [task_name]

# MRJ: v2 are instances where the initial state is set randomly
register(\
    id='UAV-Platooning-v2',
    entry_point='envs.tarski.uav_platoon.env:Environment',
    max_episode_steps=1000,
    kwargs={
        'I': initial_states,
        'target_x0': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0]),
        'g': 9.81,
        'lb_u': np.array([-0.5, -0.5, -10 ]),
        'ub_u': np.array([0.5, 0.5, 10]),
        'lb_x': np.array([-50000.0, -50000.0, 0.0, -2.0 *np.pi, -0.5*np.pi, -1.22173, 50.0,
                        -50000.0, -50000.0, 0.0, -2.0 *np.pi, -0.5*np.pi, -1.22173, 50.0]),
        'ub_x': np.array([50000.0, 50000.0, 20000.0, 2.0 *np.pi, 0.5*np.pi, 1.22173, 680.0,
                        50000.0, 50000.0, 20000.0, 2.0 *np.pi, 0.5*np.pi, 1.22173, 680.0]),
        'gains_u': np.ones((4,1)),
        'random_init': True,
        'horizon': 1000,
        'dt': h
    }
)
