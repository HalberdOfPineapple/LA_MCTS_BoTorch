import os
import math
import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List

import botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, standardize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

from optimizer import TuRBO

seed = 0
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dtype = torch.double

def get_initial_points(dim, n_init, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_init).to(dtype=dtype, device=device)
    return X_init

def test_ackley_20():
    # To minimize the function => maximize its negation
    func = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)

    func.bounds[0, :].fill_(-5)
    func.bounds[1, :].fill_(10)

    dimension = func.dim
    bounds = func.bounds
    bounds.to(dtype=dtype, device=device)

    batch_size = 4
    n_init = 2 * dimension
    max_cholesky_size = float("inf")  # Always use Cholesky

    # from botorch.utils.transforms includes standardize, normalize and unnormalize
    # https://botorch.org/api/_modules/botorch/utils/transforms.html
    # In this function's case, the input X is assumed to be normalized in [0, 1]
    # def eval_objective(x: torch.Tensor):
    #     return func(unnormalize(x, func.bounds))
    def eval_objective(x: torch.Tensor):
        # Assume input x has been unnormalized
        return func(x)
    
    X_init = get_initial_points(dim=dimension, n_init=n_init, seed=seed)
    Y_init = torch.tensor(
        [eval_objective(x) for x in X_init], dtype=dtype, device=device,
    ).unsqueeze(-1)

    turbo = TuRBO(
        obj_func=eval_objective,
        bounds=bounds,
        batch_size=batch_size,
        seed=seed,
        optimizer_params={
            'num_restarts': 10,
            'raw_samples': 512,
            'acqf': 'ts',
            'max_cholesky_size': max_cholesky_size,
        }
    )
    turbo.optimize(X_init, Y_init, num_evals=500)

if __name__ == "__main__": 
    test_ackley_20()