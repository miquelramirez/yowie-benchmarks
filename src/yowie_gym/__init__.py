# -*- coding: utf-8 -*-
# pylint: disable=

from gym.envs.registration import register
import envs.tarski.random_lq
import envs.tarski.car_park
import envs.tarski.uav_platoon
import envs.bullet

register(
    id='Bertsekas671-000-v1',
    entry_point='envs.scalar:Bertsekas671_Env',
    kwargs = {
        'w_mu': 0.0,
        'w_sigma': 1.0,
        'x0': 75.0,
        'horizon': 100
    })

register(
    id='Bertsekas671-000-v2',
    entry_point='envs.tarski.bertsekas_671:Environment',
    kwargs = {
        'mu_w': 0.0,
        'sigma_w': 1.0,
        'x0': 75.0,
        'horizon': 100,
        'dt': 0.1
    }
)

register(
    id='RechtLQR-v0',
    entry_point = 'envs.recht_lqr:LQR_Env',
    max_episode_steps=1000,
    kwargs = {}
)
