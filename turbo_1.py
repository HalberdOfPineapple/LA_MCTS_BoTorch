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

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

ACQFS = {"ts", "ei"}

def get_logger(logger_name: str):
    import logging
    import os

    # Ensure the logs directory exists
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Setup logging
    log_filename = os.path.join(logs_dir, f'{logger_name}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the logger
    return logging.getLogger(__name__)

def get_initial_points(dim, n_init, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_init).to(dtype=dtype, device=device)
    return X_init

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # to be post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # paper's version: 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state: TurboState, Y_next: torch.Tensor):
    
    # Note that `tensor(bool)`` can directly be used for condition eval
    if max(Y_next) > state.best_value: 
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    
    state.best_value = max(state.best_value, max(Y_next).item())

    # "Whenever L falls below a given minimum threshold L_min, we discard 
    #  the respective TR and initialize a new one with side length L_init"
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def generate_batch(
    state: TurboState,
    model: botorch.models.model.Model,
    X: torch.Tensor, # Evaluated points 
    Y: torch.Tensor, # Function values of Evaluated points
    batch_size: int,
    n_candidates:int = None,
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: str = "ts",
):
    assert acqf in ACQFS
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()

    # Length scales for all dimensions
    # weights - (dim, )
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    # Clamps all elements in input into the range [min, max]
    # tr_lbs, tr_ubs - (1, dim)
    tr_lbs = torch.clamp(x_center - state.length / 2 * weights, 0.0, 1.0)
    tr_ubs = torch.clamp(x_center + state.length / 2 * weights, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)

        # pert - (n_candidates, dim)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lbs + (tr_ubs - tr_lbs) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate set 
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)
    elif acqf == "ei":
        ei = qExpectedImprovement(model=model, best_f=Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lbs, tr_ubs]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    
    return X_next

class TuRBO:
    def __init__(
        self, 
        obj_func: Callable, 
        num_evals: int,
        bounds: torch.Tensor,
        batch_size: int, 
        n_init: int=20, 
        seed: int=0, 
        max_cholesky_size:float=float("inf"),
        acqf_func:str = "ts",
        logger = None
    ):
        self.obj_func = obj_func

        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.X = torch.empty((0, self.dimension), dtype=dtype, device=device)
        self.Y = torch.empty((0, 1), dtype=dtype, device=device)

        self.batch_size = batch_size
        self.n_init = n_init
        self.num_evals = num_evals
        self.num_calls = 0

        self.seed = seed
        self.max_cholesky_size = max_cholesky_size

        if acqf_func not in ACQFS:
            raise ValueError(f"Acquisition function {acqf_func} not supported")
        self.acqf_func = acqf_func

        self.logger = logger
    
    def init_samples(self, num_init: int):
        X_init = get_initial_points(self.dimension, num_init, self.seed)
        Y_init = torch.tensor(
            [self.obj_func(unnormalize(x, self.bounds)) for x in X_init], dtype=dtype, device=device,
        ).unsqueeze(-1)

        self.num_calls += len(X_init)
        return X_init, Y_init

    def optimize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num_restarts = 10 if not SMOKE_TEST else 2
        raw_samples = 512 if not SMOKE_TEST else 4
        n_candidates =  min(5000, max(2000, 200 * self.dimension)) if not SMOKE_TEST else 4

        torch.manual_seed(self.seed)

        restart_counter = 0
        while self.num_calls < self.num_evals:
            print('-' * 80)
            print(f"Restart {restart_counter}:")

            num_init = min(self.n_init, self.num_evals - self.num_calls)
            X_sampled, Y_sampled = self.init_samples(num_init)
            if self.num_calls >= self.num_evals: 
                self.X = torch.cat((self.X, X_sampled), dim=0)
                self.Y = torch.cat((self.Y, Y_sampled), dim=0)
                break
            state = TurboState(dim=self.dimension, batch_size=self.batch_size)

            while not state.restart_triggered: # Run until TuRBO converges
                train_Y = standardize(Y_sampled)
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=Interval(0.005, 4.0),
                    ),
                )

                model = SingleTaskGP(
                    X_sampled, train_Y, covar_module=covar_module, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    fit_gpytorch_mll(mll)
                    batch_size = min(self.batch_size, self.num_evals - self.num_calls)
                    X_next = generate_batch(
                        state=state,
                        model=model,
                        X=X_sampled, 
                        Y=train_Y,
                        batch_size=self.batch_size,
                        n_candidates=n_candidates,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        acqf=self.acqf_func,
                    )
                
                Y_next = torch.tensor(
                    [self.obj_func(unnormalize(x, self.bounds)) for x in X_next], dtype=dtype, device=device,
                ).unsqueeze(-1)

                state = update_state(state, Y_next)
                X_sampled = torch.cat((X_sampled, X_next), dim=0)
                Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
                self.num_calls += len(X_next)

                log_msg = (
                    f"[Restart {restart_counter}] {len(self.X) + len(X_sampled)}) "
                    f"Best value: {state.best_value:.2e} | TR length: {state.length:.2e} | "
                    f"num. restarts: {state.failure_counter}/{state.failure_tolerance} | "
                    f"num. successes: {state.success_counter}/{state.success_tolerance}"
                )
                print(log_msg)
                if self.logger is not None:
                    self.logger.info(log_msg)

                if self.num_calls >= self.num_evals: break

            self.X = torch.cat((self.X, X_sampled), dim=0)
            self.Y = torch.cat((self.Y, Y_sampled), dim=0)
            restart_counter += 1

        return self.X, self.Y
    
def test_ackley_20():
    # To minimize the function => maximize its negation
    func = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
    func.bounds[0, :].fill_(-5)
    func.bounds[1, :].fill_(10)
    dim = func.dim
    lbs, ubs = func.bounds

    num_evals = 1000
    batch_size = 4
    n_init = 2 * dim
    max_cholesky_size = float("inf")  # Always use Cholesky

    # from botorch.utils.transforms includes standardize, normalize and unnormalize
    # https://botorch.org/api/_modules/botorch/utils/transforms.html
    # In this function's case, the input X is assumed to be normalized in [0, 1]
    def eval_objective(x: torch.Tensor):
        return func(x)
    
    turbo = TuRBO(
        obj_func=eval_objective,
        bounds = func.bounds,
        num_evals=num_evals,
        dimension=dim,
        batch_size=batch_size,
        n_init=n_init,
        max_cholesky_size=max_cholesky_size,
    )
    xs, fxs = turbo.optimize()


if __name__ == "__main__": 
    test_ackley_20()