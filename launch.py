import os
import yaml
import torch 
import argparse
from time import perf_counter

from mcts import MCTS
from utils import init_logger, get_logger, LOG_DIR
from optimizer import OPTIMIZER_MAP
from classifier import CLASSIFIER_MAP
from turbo_1 import TuRBO as PlainTuRBO

from botorch.test_functions import Ackley, Rosenbrock
from benchmarks import BENCHMARK_MAP

dtype = torch.double
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def build_test_params():
    test_params = {}
    test_params['seed'] = 0
    test_params['Cp'] = 1.
    test_params['leaf_size'] = 20
    test_params['node_selection_type'] = 'UCB'
    test_params['initial_sampling_method'] = 'Sobol'

    dim = 20
    test_params['bounds'] = torch.tensor([
        [-5.] * dim, [10.] * dim]).to(dtype=dtype, device=device)
    test_params['obj_func'] = Ackley(
        dim=dim, negate=True, 
        bounds=[(-5., 10.) for _ in range(dim)]).to(dtype=dtype, device=device)
    # test_params['obj_func'] = Rosenbrock(
    #     dim=dim, negate=True, 
    #     bounds=[(-5., 10.)] * dim).to(dtype=dtype, device=device)

    test_params['num_init'] = 40

    test_params['optimizer_type'] = 'turbo'
    test_params['optimizer_params'] = {
        'batch_size': 4, # Note this is the "local" batch size
        'acqf': 'ts',
        'num_restarts': 10,
        'raw_samples': 512,
    }

    test_params['classifier_type'] = 'SVM'
    test_params['classifier_params'] = {
        'kernel_type': 'rbf',
        'gamma_type': 'auto',
    }

    return test_params

def test_main():
    expr_name = 'test_expr'
    num_runs = 1000

    init_logger(expr_name)
    logger = get_logger()

    test_params = build_test_params()

    print(f"Start experiment: {expr_name}")
    print('=' * 80)
    print(f"Config:")
    print(f"Number of samples: {num_runs}")
    for k, v in test_params.items():
        print(f"{k}: {v}")
    print('=' * 80)
    print()

    mcts = MCTS(**test_params)
    xs, fxs, best_X, best_Y = mcts.optimize(num_evals=num_runs)
    logger.info("-" * 50)
    logger.info(f"Best X: {best_X}")
    logger.info(f"Best Y: {best_Y}")


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

    print(f"Start experiment with plain-TuRBO: {expr_name}")
    print('=' * 50)
    print(f"Config:")
    for k, v in turbo_params.items():
        print(f"{k}: {v}")
    print('=' * 50)

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

    data_file_name = f"{expr_name}_data.log"
    xs, fxs = xs.detach().cpu().numpy(), fxs.detach().cpu().numpy()
    with open(os.path.join(LOG_DIR, data_file_name), 'w') as f:
        for x, fx in zip(xs, fxs):
            x = str(list(x))
            f.write(f"{fx}, {x}\n")


def run_expr(benchmark_name: str, num_runs: int, method: str):
    expr_name = f"{method}_{benchmark_name}_{num_runs}"
    init_logger(expr_name)
    logger = get_logger()

    benchmark, kwargs = BENCHMARK_MAP[benchmark_name.lower()]
    benchmark = benchmark(**kwargs)
    params: dict = benchmark.to_dict()
    if method.lower() == 'turbo':
        run_turbo_expr(expr_name, params, num_runs, logger)
        return

    print(f"Start experiment: {expr_name}")
    print('=' * 50)
    print(f"Config:")
    for k, v in params.items():
        print(f"{k}: {v}")
    print('=' * 50)


    start_time = perf_counter()
    mcts = MCTS(**params)
    xs, fxs, best_X, best_Y = mcts.optimize(num_evals=num_runs)
    elapsed_time = perf_counter() - start_time

    logger.info("-" * 50)
    logger.info(f"Best X: {best_X}")
    logger.info(f"Best Y: {best_Y}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    data_file_name = f"{expr_name}_data.log"
    xs, fxs = xs.detach().cpu().numpy(), fxs.detach().cpu().numpy()
    with open(os.path.join(LOG_DIR, data_file_name), 'w') as f:
        for x, fx in zip(xs, fxs):
            x = str(list(x))
            f.write(f"{fx}, {x}\n")

if __name__ == "__main__":
    # test_main()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--benchmark_name', '-e', type=str, default='ackley20d')
    argparser.add_argument('--num_runs', '-n', type=int, default=1000)
    argparser.add_argument('--method', '-m', type=str, default='mcts')
    args = argparser.parse_args()

    run_expr(args.benchmark_name, args.num_runs, args.method)
