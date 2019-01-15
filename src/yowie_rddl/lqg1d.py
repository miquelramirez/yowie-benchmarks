"""
    1D LQR Navigation problems

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
    parser = argparse.ArgumentParser(description='1D LQG problem generators')
    parser.add_argument('--num-instances', default=10, type=int, help='Number of random instances to generate')
    parser.add_argument('--seed', required=True, type=int, help='Seed for the random number generator')
    parser.add_argument('--output-prefix', required=False, default='.', help='Prefix path for output files')
    parser.add_argument('--horizon', required=False, default=20, type=int, help='Horizon of the MDP')

    args = parser.parse_args()

    if not os.path.exists(args.output_prefix):
        print("Creating output directory '{}'...".format(args.output_prefix))
        os.makedirs(args.output_prefix)

    return args


def make_task(k, args):

    lang = tarski.language('lqg_nav_1d', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL, Theory.RANDOM])
    the_task = Task( lang, 'lqg_nav_1d',  'instance_{0:06d}'.format(k))

    the_task.requirements = [rddl.Requirements.CONTINUOUS, rddl.Requirements.REWARD_DET]

    the_task.parameters.discount = 1.0
    the_task.parameters.horizon = args.horizon
    the_task.parameters.max_nondef_actions = 'pos-inf'

    # variables
    x = lang.function('x', lang.Real)
    v = lang.function('v', lang.Real)
    u = lang.function('u', lang.Real)
    t = lang.function('t', lang.Real)

    # non fluents
    dt = lang.function('dt', lang.Real)
    gx = lang.function('gx', lang.Real)
    H = lang.function('H', lang.Real)
    # Perturbation distribution parameters
    mu_w = lang.function('mu_w', lang.Real)
    sigma_w = lang.function('sigma_w', lang.Real)

    # cpfs
    the_task.add_cpfs(t(), t() + 1.0)
    the_task.add_cpfs(v(), v() + dt() * u() + normal(mu_w(), sigma_w()))
    the_task.add_cpfs(x(), x() + dt() * v())

    # constraints
    # constraints
    the_task.add_constraint( u() >= -1.0, rddl.ConstraintType.ACTION)
    the_task.add_constraint( u() <= 1.0, rddl.ConstraintType.ACTION)
    the_task.add_constraint( v() >= -5.0, rddl.ConstraintType.STATE)
    the_task.add_constraint( v() <= 5.0, rddl.ConstraintType.STATE)
    the_task.add_constraint( x() >= -100.0, rddl.ConstraintType.STATE)
    the_task.add_constraint( x() <= 100.0, rddl.ConstraintType.STATE)

    # cost function
    Q = (x()-gx()) * (x()-gx())
    # MRJ: RDDL does not support the abs() algebraic construct
    zero = lang.constant(0.0, lang.Real)
    R = ite(t() < H(), u() * u() * 0.01, zero)
    
    V = ite(abs(v()) <= 5.0, 5.0 - abs(v()), lang.constant(0.0, lang.Real))
    X = ite(abs(x()) <= 100.0, 100.0 - abs(x()), lang.constant(0.0, lang.Real))
    
    #R = u() * u() * 0.01
    #MRJ: flip sign to turn minimisation into maximisation problem
    the_task.reward = -1.0 * (Q + R + V + X)

    # fluent metadata
    the_task.declare_state_fluent(x(), 0.0)
    the_task.declare_state_fluent(t(), 0.0)
    the_task.declare_state_fluent(v(), 0.0)
    the_task.declare_action_fluent(u(), 0.0)
    the_task.declare_non_fluent(dt(), 0.0)
    the_task.declare_non_fluent(gx(), 0.0)
    the_task.declare_non_fluent(mu_w(), 0.0)
    the_task.declare_non_fluent(sigma_w(), 0.0)    
    the_task.declare_non_fluent(H(), 0.0)

    # definitions
    the_task.x0.setx(x(), 0.0)
    the_task.x0.setx(v(), 0.0)
    the_task.x0.setx(u(), 0.0)
    the_task.x0.setx(t(), 0.0)
    the_task.x0.setx(dt(), 0.5)
    the_task.x0.setx(mu_w(), 0.0)
    the_task.x0.setx(sigma_w(), 0.05)    
    the_task.x0.setx(H(), float(args.horizon))

    return the_task

def main(args):

    np.random.seed(args.seed)

    for k in tqdm(range(args.num_instances)):
        task_k = make_task(k, args)

        if not os.path.exists(os.path.join(args.output_prefix, task_k.domain_name)):
            os.makedirs(os.path.join(args.output_prefix, task_k.domain_name))

        x = task_k.L.get('x')
        v = task_k.L.get('v')
        gx = task_k.L.get('gx')

        task_k.x0.setx(x(), np.clip(np.random.normal(0, 100.0), -100.0, 100.0))
        task_k.x0.setx(v(), np.clip(np.random.normal(0, 1.0), -5.0, 5.0))
        task_k.x0.setx(gx(), np.clip(np.random.normal(0, 100.0), -100.0, 100.0))

        the_writer = rddl.Writer(task_k)
        rddl_filename = os.path.join(args.output_prefix,\
            task_k.domain_name,\
            '{}.rddl'.format(task_k.instance_name))
        the_writer.write_model(rddl_filename)



if __name__ == '__main__':
    print('lqg1d.py')
    args = parse_arguments()
    main(args)
