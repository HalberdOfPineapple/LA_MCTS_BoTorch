import os
import argparse
import numpy as np
from time import perf_counter
from functions.functions import *
from functions.mujoco_functions import *

from lamcts import MCTS
from lamcts.turbo_1.turbo_1 import Turbo1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
FUNCTION_MAP: Dict[str, Any] = {
    'ackley': Ackley,
    'levy':  Levy,
    'lunar': Lunarlanding,
    'swimmer': Swimmer,
    'hopper': Hopper,
}

def generate_random_samples(nums_samples, dims, lb, ub):
    x = np.random.uniform(lb, ub, size = (nums_samples, dims))
    return x


def main(args):
    lb = -5
    ub = 10
    num_init = 40
    num_evals = args.iterations
    expr_name = f"turbo_{args.func}_{args.dims}_{args.iterations}"

    if args.func not in FUNCTION_MAP:
        raise ValueError(f"Not defined function: {args.func}")

    func_cls = FUNCTION_MAP[args.func]
    if args.func == 'ackley' or args.func == 'levy':
        if args.dims <= 0:
            raise ValueError(f'Dimentionality should be >0 when function is {args.func}')
        func = func_cls(dims=args.dims)
    else:
        func = func_cls()

    assert args.iterations > 0, "Number of iterations should be > 0"
    
    start_time = perf_counter()
    X_init = generate_random_samples(num_init, args.dims, lb, ub)
    turbo1 = Turbo1(
            f  = func,              # Handle to objective function
            lb = func.lb,           # Numpy array specifying lower bounds
            ub = func.ub,           # Numpy array specifying upper bounds
            n_init = num_init,            # Number of initial bounds from an Latin hypercube design
            max_evals  = num_evals, # Maximum number of evaluations
            batch_size = 1,         # How large batch size TuRBO uses
            verbose=True,           # Print information from each batch
            use_ard=True,           # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
            n_training_steps=50,    # Number of steps of ADAM to learn the hypers
            min_cuda=1024,          #  Run on the CPU for small datasets
            device="cpu",           # "cpu" or "cuda"
            dtype="float32",        # float64 or float32
            X_init = X_init,
        )
    xs, fxs = turbo1.optimize()
    elapsed_time = perf_counter() - start_time

    with open(os.path.join(LOG_DIR, expr_name + '.log'), 'w') as f:
        for x, fx in zip(xs, fxs):
            x = str(list(x))
            f.write(f'{fx}, {x}\n')
    print(f"Total time: {elapsed_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--func', '-f', default="levy", help='specify the test function')
    parser.add_argument('--dims', '-d', type=int, default=10, help='specify the problem dimensions')
    parser.add_argument('--iterations', '-n', type=int, default=1000, help='specify the iterations to collect in the search')
    args = parser.parse_args()

    main(args)