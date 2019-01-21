"""
    2D Multi-units LQR Navigation problems

"""

import argparse
import os
import numpy as np
from tqdm import tqdm

import tarski
from tarski.theories import Theory
from tarski.syntax import *
from tarski.io import rddl
from tarski.model import Model
from tarski.evaluators.simple import evaluate
from tarski.syntax.arithmetic import *
from tarski.syntax.arithmetic.special import *
from tarski.syntax.arithmetic.random import *
from tarski.rddl import Task

def parse_arguments():
    parser = argparse.ArgumentParser(description='2D Multi-units LQR problem generators')
    parser.add_argument('--num-instances', default=10, type=int, help='Number of random instances to generate')
    parser.add_argument('--seed', required=True, type=int, help='Seed for the random number generator')
    parser.add_argument('--output-prefix', required=False, default='.', help='Prefix path for output files')
    parser.add_argument('--horizon', required=False, default=50, type=int, help='Horizon of the MDP')

    args = parser.parse_args()

    if not os.path.exists(args.output_prefix):
        print("Creating output directory '{}'...".format(args.output_prefix))
        os.makedirs(args.output_prefix)

    return args


def make_task(k, args):

    lang = tarski.language('lqr_nav_2d_multi_unit', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])
    the_task = Task( lang, 'lqr_nav_2d_multi_unit', 'instance_{0:06d}'.format(k))

    the_task.requirements = [rddl.Requirements.CONTINUOUS, rddl.Requirements.REWARD_DET]

    the_task.parameters.discount = 1.0
    the_task.parameters.horizon = args.horizon
    the_task.parameters.max_nondef_actions = 'pos-inf'

    vehicle = lang.sort('vehicle', lang.Object)

    # variables
    x = lang.function('x', vehicle, lang.Real)
    y = lang.function('y', vehicle, lang.Real)
    vx = lang.function('vx', vehicle, lang.Real)
    vy = lang.function('vy', vehicle, lang.Real)
    ux = lang.function('ux', vehicle, lang.Real)
    uy = lang.function('uy', vehicle, lang.Real)
    t = lang.function('t', lang.Real)

    # objects
    v001 = lang.constant('v001', vehicle)
    v002 = lang.constant('v002', vehicle)

    # non fluents
    dt = lang.function('dt', lang.Real)
    gx = lang.function('gx', lang.Real)
    gy = lang.function('gy', lang.Real)
    H = lang.function('H', lang.Real)

    # Perturbation distribution parameters
    mu_w = lang.function('mu_w', lang.Real)
    sigma_w = lang.function('sigma_w', lang.Real)

    # logical variable
    v = lang.variable('v', vehicle)

    # cpfs
    the_task.add_cpfs(t(), t() + 1.0)
    the_task.add_cpfs(vx(v), vx(v) + dt() * ux(v))
    the_task.add_cpfs(vy(v), vy(v) + dt() * uy(v))
    the_task.add_cpfs(x(v), x(v) + dt() * vx(v))
    the_task.add_cpfs(y(v), y(v) + dt() * vy(v))

    # constraints
    the_task.add_constraint( forall(v, ux(v) >= -1.0), rddl.ConstraintType.ACTION)
    the_task.add_constraint( forall(v, ux(v) <= 1.0), rddl.ConstraintType.ACTION)
    the_task.add_constraint( forall(v, uy(v) >= -1.0), rddl.ConstraintType.ACTION)
    the_task.add_constraint( forall(v, uy(v) <= 1.0), rddl.ConstraintType.ACTION)
    the_task.add_constraint( forall(v, (sqrt(vx(v)*vx(v) + vy(v)*vy(v)) <= 5.0)), rddl.ConstraintType.STATE)
    the_task.add_constraint( forall(v, x(v) >= -100.0), rddl.ConstraintType.STATE)
    the_task.add_constraint( forall(v, x(v) <= 100.0), rddl.ConstraintType.STATE)
    the_task.add_constraint( forall(v, y(v) >= -100.0), rddl.ConstraintType.STATE)
    the_task.add_constraint( forall(v, y(v) <= 100.0), rddl.ConstraintType.STATE)

    # cost function
    Q = sumterm(v, ((x(v) - gx()) * (x(v) - gx())) + ((y(v) - gy()) * (y(v) - gy())))
    # MRJ: RDDL does not support the abs() algebraic construct
    zero = lang.constant(0.0, lang.Real)
    R = sumterm(v, ite(t() < H(), (ux(v) * ux(v) * 0.01) + (uy(v) * uy(v) * 0.01), zero))

    V = sumterm(v, ite((sqrt(vx(v)*vx(v) + vy(v)*vy(v)) <= 5.0), 5.0 - abs(sqrt(vx(v)*vx(v) + vy(v)*vy(v))), lang.constant(0.0, lang.Real)))
    X = sumterm(v, ite(abs(x(v)) <= 100.0, 100.0 - abs(x(v)), lang.constant(0.0, lang.Real)))
    Y = sumterm(v, ite(abs(y(v)) <= 100.0, 100.0 - abs(y(v)), lang.constant(0.0, lang.Real)))
    
    #R = u() * u() * 0.01
    the_task.reward = -1.0 * (Q + R + V + X + Y)    

    # definitions

    the_task.x0.setx(ux(v001), 0.0)
    the_task.x0.setx(uy(v001), 0.0)
    the_task.x0.setx(t(), 0.0)
    the_task.x0.setx(dt(), 0.5)
    the_task.x0.setx(mu_w(), 0.0)
    the_task.x0.setx(sigma_w(), 0.05)
    the_task.x0.setx(H(), float(args.horizon))

    # fluent metadata
    the_task.declare_state_fluent(x(v), 0.0)
    the_task.declare_state_fluent(y(v), 0.0)
    the_task.declare_state_fluent(t(), 0.0)
    the_task.declare_state_fluent(vx(v), 0.0)
    the_task.declare_state_fluent(vy(v), 0.0)
    the_task.declare_action_fluent(ux(v), 0.0)
    the_task.declare_action_fluent(uy(v), 0.0)
    the_task.declare_non_fluent(dt(), 0.0)
    the_task.declare_non_fluent(gx(), 0.0)
    the_task.declare_non_fluent(gy(), 0.0)
    the_task.declare_non_fluent(mu_w(), 0.0)
    the_task.declare_non_fluent(sigma_w(), 0.0)
    the_task.declare_non_fluent(H(), 0.0)

    return the_task

def main(args):

    np.random.seed(args.seed)

    for k in tqdm(range(args.num_instances)):
        task_k = make_task(k, args)

        if not os.path.exists(os.path.join(args.output_prefix, task_k.domain_name)):
            os.makedirs(os.path.join(args.output_prefix, task_k.domain_name))

        ux = task_k.L.get('ux')
        uy = task_k.L.get('uy')

        x = task_k.L.get('x')
        y = task_k.L.get('y')

        vx = task_k.L.get('vx')
        vy = task_k.L.get('vy')

        v001 = task_k.L.get('v001')
        v002 = task_k.L.get('v002')

        gx = task_k.L.get('gx')
        gy = task_k.L.get('gy')

        task_k.x0.setx(ux(v001), 0)
        task_k.x0.setx(uy(v001), 0)
        task_k.x0.setx(ux(v002), 0)
        task_k.x0.setx(uy(v002), 0)

        task_k.x0.setx(x(v001), np.random.normal(0, 100.0))
        task_k.x0.setx(y(v001), np.random.normal(0, 100.0))
        task_k.x0.setx(vx(v001), np.random.normal(0, 1.0))
        task_k.x0.setx(vy(v001), np.random.normal(0, 1.0))

        task_k.x0.setx(x(v002), np.random.normal(0, 100.0))
        task_k.x0.setx(y(v002), np.random.normal(0, 100.0))
        task_k.x0.setx(vx(v002), np.random.normal(0, 1.0))
        task_k.x0.setx(vy(v002), np.random.normal(0, 1.0))

        task_k.x0.setx(gx(), np.random.normal(0, 100.0))
        task_k.x0.setx(gy(), np.random.normal(0, 100.0))

        the_writer = rddl.Writer(task_k)
        rddl_filename = os.path.join(args.output_prefix,\
            task_k.domain_name,\
            '{}.rddl'.format(task_k.instance_name))
        the_writer.write_model(rddl_filename)

if __name__ == '__main__':
    print('lqr2dmu.py')
    args = parse_arguments()
    main(args)
