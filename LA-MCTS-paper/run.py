# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

# ---------------------------------------------------------------------------------------
# FAQ:

# 1. How to retrieve every f(x) during the search?

#   During the optimization, the function will create a folder to store the f(x) trace; and
# the name of the folder is in the format of function name + function dimensions, e.g. Ackley10.

#   Every 100 samples, the function will write a row to a file named results + total samples, e.g. result100 
# mean f(x) trace in the first 100 samples.

#   Each last row of result file contains the f(x) trace starting from 1th sample -> the current sample.
# Results of previous rows are from previous experiments, as we always append the results from a new experiment
# to the last row.

# Here is an example to interpret a row of f(x) trace.
#   [5, 3.2, 2.1, ..., 1.1]
# The first sampled f(x) is 5, the second sampled f(x) is 3.2, and the last sampled f(x) is 1.1 

# 2. How to improve the performance?
# Tune Cp, leaf_size, and improve BO sampler with others.
# ---------------------------------------------------------------------------------------

import os
from time import perf_counter
from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')

FUNCTION_MAP: Dict[str, Any] = {
    'ackley': Ackley,
    'levy':  Levy,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'lunar': Lunarlanding,
    'swimmer': Swimmer,
    'hopper': Hopper,
}


def main(args):
    print('-' * 80)
    if args.func not in FUNCTION_MAP:
        raise ValueError(f"Not defined function: {args.func}")

    func_cls = FUNCTION_MAP[args.func]
    f = func_cls(dims=args.dims)

    assert args.iterations > 0, "Number of iterations should be > 0"

    start_time = perf_counter()
    agent = MCTS(
        lb = f.lb,              # the lower bound of each problem dimensions
        ub = f.ub,              # the upper bound of each problem dimensions
        dims = f.dims,          # the problem dimensions
        ninits = f.ninits,      # the number of random samples used in initializations 
        func = f,               # function object to be optimized
        Cp = f.Cp,              # Cp for MCTS
        leaf_size = f.leaf_size, # tree leaf size
        kernel_type = f.kernel_type, #SVM configruation
        gamma_type = f.gamma_type,    #SVM configruation
        solver_type = args.solver_type,
    )
    samples = agent.search(iterations=args.iterations)
    fxs = [sample[1] for sample in samples]
    elapsed_time = perf_counter() - start_time

    expr_name = f"{args.func}_{args.dims}_{args.iterations}_{args.solver_type}"
    with open(os.path.join(LOG_DIR, expr_name + '_data.log'), 'w') as f:
        for x, fx in samples:
            x = str(list(x))
            f.write(f'{fx}, {x}\n')
    with open(os.path.join(LOG_DIR, expr_name + '.log'), 'w') as f:
        f.write(f'Best Function value: {np.max(fxs):.4f}\n')
        f.write(f'Elapsed time: {elapsed_time:3f} seconds\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--func', '-f', default="levy", help='specify the test function')
    parser.add_argument('--dims', '-d', type=int, default=10, help='specify the problem dimensions')
    parser.add_argument('--iterations', '-n', type=int, default=1000, help='specify the iterations to collect in the search')
    parser.add_argument("--solver_type", '-s', type=str, default='bo', help="Type of local sampler")
    args = parser.parse_args()

    main(args)
