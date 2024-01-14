import os
import yaml
import torch 
import argparse
from time import perf_counter

from mcts import MCTS
from utils import init_logger, get_logger, LOG_DIR, set_expr_name, print_log, IBEX_LOG_DIR
from optimizer import OPTIMIZER_MAP
from classifier import CLASSIFIER_MAP
from turbo_1 import TuRBO as PlainTuRBO

from botorch.test_functions import Ackley, Rosenbrock
from benchmarks import BENCHMARK_MAP

dtype = torch.double
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def run_turbo_expr(expr_name: str, params: dict, num_runs: int, logger):
    turbo_params = {
        'obj_func': params['obj_func'],
        'bounds': params['bounds'],
        'batch_size': params['optimizer_params']['batch_size'],
        'n_init': params['num_init'],
        'num_evals': num_runs,
        'seed': params['seed'],
        'acqf_func': params['optimizer_params'].get('acqf', 'ts'),
        'max_cholesky_size': params['optimizer_params'].get('max_cholesky_size', float('inf')),
    }

    print_log(f"Start experiment with plain-TuRBO: {expr_name}")
    print_log('=' * 50)
    print_log(f"Config:")
    for k, v in turbo_params.items():
        print_log(f"{k}: {v}")
    print_log('=' * 50)

    # --------------------------------------
    # Optimization
    start_time = perf_counter()

    turbo = PlainTuRBO(logger=logger, **turbo_params)
    xs, fxs = turbo.optimize()

    elapsed_time = perf_counter() - start_time
    # --------------------------------------

    logger.info("-" * 50)
    logger.info(f"Best function value: {fxs.max().item()}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    log_dir = IBEX_LOG_DIR if args.ibex else LOG_DIR
    data_file_name = f"{expr_name}_data.log"
    xs, fxs = xs.detach().cpu().numpy(), fxs.detach().cpu().numpy()
    with open(os.path.join(log_dir, data_file_name), 'w') as f:
        for x, fx in zip(xs, fxs):
            x = str(list(x))
            f.write(f"{fx}, {x}\n")


def run_expr(benchmark_name: str, num_runs: int, method: str):
    expr_name = f"{method}_{benchmark_name}_{num_runs}"
    init_logger(expr_name, args.ibex)
    set_expr_name(expr_name)
    logger = get_logger()

    benchmark, kwargs = BENCHMARK_MAP[benchmark_name.lower()]
    benchmark = benchmark(**kwargs)
    params: dict = benchmark.to_dict()
    if method.lower() == 'turbo':
        run_turbo_expr(expr_name, params, num_runs, logger)
        return

    print_log(f"Start experiment: {expr_name}")
    print_log('=' * 50)
    print_log(f"Config:")
    for k, v in params.items():
        print_log(f"{k}: {v}")
    print_log('=' * 50)


    start_time = perf_counter()
    mcts = MCTS(**params)
    xs, fxs, best_X, best_Y = mcts.optimize(num_evals=num_runs)
    elapsed_time = perf_counter() - start_time

    logger.info("-" * 50)
    logger.info(f"Best X: {best_X}")
    logger.info(f"Best Y: {best_Y}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    log_dir = IBEX_LOG_DIR if args.ibex else LOG_DIR
    data_file_name = f"{expr_name}_data.log"
    xs, fxs = xs.detach().cpu().numpy(), fxs.detach().cpu().numpy()
    with open(os.path.join(log_dir, data_file_name), 'w') as f:
        for x, fx in zip(xs, fxs):
            x = str(list(x))
            f.write(f"{fx}, {x}\n")

if __name__ == "__main__":
    # test_main()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--benchmark_name', '-e', type=str, default='ackley20d')
    argparser.add_argument('--num_runs', '-n', type=int, default=1000)
    argparser.add_argument('--method', '-m', type=str, default='mcts')
    argparser.add_argument("--ibex", "-i", action="store_true", help="in Ibex environment")
    args = argparser.parse_args()

    run_expr(args.benchmark_name, args.num_runs, args.method)
