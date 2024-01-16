import os
import math
import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List

import botorch
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, standardize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

from utils import print_log

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

ACQFS = {"ts", "ei"}
def get_initial_points(dim, n_init, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_init).to(dtype=dtype, device=device)
    return X_init

class GPEI:
    def __init__(
        self, 
        obj_func: Callable, 
        bounds: torch.Tensor,
        batch_size: int, 
        n_init: int=20, 
        seed: int=0, 
        max_cholesky_size:float=float("inf"),
        **kwargs,
    ):
        self.obj_func = obj_func

        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.X = torch.empty((0, self.dimension), dtype=dtype, device=device)
        self.Y = torch.empty((0, 1), dtype=dtype, device=device)

        self.batch_size = batch_size
        self.n_init = n_init
        self.num_calls = 0

        self.seed = seed
        self.max_cholesky_size = max_cholesky_size

        self.acqf_func = qExpectedImprovement
    
    def init_samples(self, num_init: int):
        X_init = get_initial_points(self.dimension, num_init, self.seed)
        Y_init = torch.tensor(
            [self.obj_func(unnormalize(x, self.bounds)) for x in X_init], dtype=dtype, device=device,
        ).unsqueeze(-1)

        self.num_calls += len(X_init)
        return X_init, Y_init

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_restarts = 10 if not SMOKE_TEST else 2
        raw_samples = 512 if not SMOKE_TEST else 4
        n_candidates =  min(5000, max(2000, 200 * self.dimension)) if not SMOKE_TEST else 4

        torch.manual_seed(self.seed)

        num_init = min(self.n_init, num_evals - self.num_calls)
        X_sampled, Y_sampled = self.init_samples(num_init)
        
        while self.num_calls < num_evals:
            batch_size = min(self.batch_size, num_evals - self.num_calls)

            train_Y = standardize(Y_sampled)

            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(X_sampled, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            ei = self.acqf_func(model=model, best_f=train_Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds =torch.stack([
                    torch.zeros(self.dimension, dtype=dtype, device=device),
                    torch.ones(self.dimension, dtype=dtype, device=device),
                ]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            Y_next = torch.tensor(
                [self.obj_func(unnormalize(x, self.bounds)) for x in X_next], dtype=dtype, device=device,
            ).unsqueeze(-1)

            X_sampled = torch.cat((X_sampled, X_next), dim=0)
            Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
            self.num_calls += len(X_next)
        
            log_msg = (
                    f"Sample {len(X_sampled) + len(self.X)} | "
                    f"Best value: {Y_sampled.max().item():.2f} |"
            )
            print_log(log_msg)

        self.X = torch.cat((self.X, X_sampled), dim=0)
        self.Y = torch.cat((self.Y, Y_sampled), dim=0)
        best_x = self.X[self.Y.argmax()]
        best_y = self.Y.max()
        return self.X, self.Y, best_x, best_y

    
def test_ackley_20():
    # To minimize the function => maximize its negation
    func = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
    func.bounds[0, :].fill_(-5)
    func.bounds[1, :].fill_(10)
    dim = func.dim

    num_evals = 1000
    batch_size = 4
    n_init = 2 * dim
    max_cholesky_size = float("inf")  # Always use Cholesky

    # from botorch.utils.transforms includes standardize, normalize and unnormalize
    # https://botorch.org/api/_modules/botorch/utils/transforms.html
    # In this function's case, the input X is assumed to be normalized in [0, 1]
    gp_ei = GPEI(
        obj_func=func,
        num_evals=num_evals,
        bounds=func.bounds,
        batch_size=batch_size,
        n_init=n_init,
        seed=0,
        max_cholesky_size=max_cholesky_size,
        acqf_func="ei",
    )
    xs, fxs = gp_ei.optimize()

if __name__ == "__main__":
    from utils import init_logger
    init_logger("gp_ei_ackley20d", False)
    test_ackley_20()