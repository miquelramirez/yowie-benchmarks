"""
    Random LQ problems, as defined in

    Control-Limited Differential Dynamic Programming
    Yuval Tassa, Nicolas Mansard and Emo Todorov
    Proceedings of ICRA-14
"""
import numpy as np
import h5py
import os
import logging

from gym.envs.registration import register

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))

seed = 1337
N = 20
h = 0.01
c_u = 0.01 # MRJ: control cost coefficient not given in the paper
task_names = []

np.random.seed(seed)

def generate_random_linear_system():
    # ... state dimension ... was drawn uniformly from [10,100]
    d = np.random.randint(10, 100)

    # ... control dimension ... was drawn from [1, d/2]
    p = np.random.randint(1, d/2)

    # dynamic matrices
    fx = np.eye(d,d) + h * np.random.normal(0.0, 1.0, size=(d,d))
    fu = h * np.random.normal(0.0, 1.0, size=(d,p))

    # cost matrices
    lxx = h * np.eye(d,d)
    lf = lxx
    luu = c_u *  h * np.eye(p,p)

    # initial state
    x0 = np.random.normal(p, 1.0, size=(d,1))

    return fx, fu, lxx, luu, lf, x0

def get_filename(index):
    return 'random_lq_{:03d}.h5'.format(index)

def write_system(index, fx, fu, lxx, luu, lf, x0):
    path = os.path.join(_CURRENT_DIR_, get_filename(index))
    with h5py.File(path, 'w') as df:
        fx_ds = df.create_dataset('fx', data=fx)
        fu_ds = df.create_dataset('fu', data=fu)
        lxx_ds = df.create_dataset('lxx', data=lxx)
        luu_ds = df.create_dataset('luu', data=luu)
        lf_ds = df.create_dataset('lf', data=lf)
        x0_ds = df.create_dataset('x0', data=x0)

def read_system(index):
    logging.info("Reading parameters of system #{}".format(index))
    path = os.path.join(_CURRENT_DIR_, get_filename(index))
    with h5py.File(path, 'r') as df:
        fx = df['fx'][()]
        fu = df['fu'][()]
        lxx = df['lxx'][()]
        luu = df['luu'][()]
        lf = df['lf'][()]
        x0 = df['x0'][()]
    return fx, fu, lxx, luu, lf, x0

for i in range(N):
    if not os.path.exists(os.path.join(_CURRENT_DIR_, get_filename(i))):
        fx, fu, lxx, luu, lf, x0 = generate_random_linear_system()
        write_system(i, fx, fu, lxx, luu, lf, x0)
    else:
        fx, fu, lxx, luu, lf, x0 = read_system(i)

    task_name = 'RandomLQ-{:03d}-v1'.format(i)
    register(\
        id=task_name,
        entry_point='yowie_gym.tarski.random_lq.env:Environment',
        max_episode_steps=200,
        kwargs={
            'fx': fx,
            'fu': fu,
            'lxx': lxx,
            'luu': luu,
            'lf': lf,
            'x0': x0,
            'dt': h,
            'horizon': 200,
            'random_init': False})
    task_names.append(task_name)
    task_name = 'RandomLQ-{:03d}-v2'.format(i)
    register(\
        id=task_name,
        entry_point='yowie_gym.tarski.random_lq.env:Environment',
        max_episode_steps=200,
        kwargs={
            'fx': fx,
            'fu': fu,
            'lxx': lxx,
            'luu': luu,
            'lf': lf,
            'x0': x0,
            'dt': h,
            'horizon': 200,
            'random_init': True})
    task_names.append(task_name)
