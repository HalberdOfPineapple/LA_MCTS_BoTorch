import os
import math
import torch
import numpy as np
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union, List, Dict

import botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, standardize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from node import Node
from utils import get_logger, print_log

def contrained_acqf(
        acqf: botorch.acquisition.AcquisitionFunction, 
        path: List[Node], 
        X: torch.Tensor, # (batch_size, dimension)
    ) -> torch.Tensor:
        results = torch.full((X.shape[0], ), float('-inf'), dtype=X.dtype, device=X.device)

        choices = Node.path_filter(path, X) # (batch_size, )
        results[choices] = acqf(X[choices])

        return results

class BaseOptimizer:
    @abstractmethod
    def optimize(self, X_in_region: torch.Tensor, Y_in_region: torch.Tensor, num_evals: int, path: List[Node]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

# ================================================================
# TuRBO
# ================================================================
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
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

ACQFS = {"ts", "ei"}
DEFAULT_TURBO_PARAMS = {
    'acqf': 'ts',
    'num_restarts': 10,
    'raw_samples': 512,
}
class TuRBO(BaseOptimizer):
    def __init__(self, 
        obj_func: Callable,
        bounds: torch.Tensor, # (2, dimension)
        num_init: int,
        seed: int, 
        optimizer_params: Dict
    ):
        global logger
        logger = get_logger()

        torch.manual_seed(seed)
        self.seed = seed
        self.obj_func = obj_func
    
        self.acqf = optimizer_params['acqf']
        if not self.acqf in ACQFS:
            raise ValueError(f"Acquisition function {self.acqf} not supported")
        
        self.num_init = num_init # Note that num_init for local TuRBO is not the same as the global one
        self.init_bounding_box_length = optimizer_params.get('init_bounding_box_length', 0.0005)
        
        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.batch_size = optimizer_params['batch_size']
        
        self.max_cholesky_size = optimizer_params.get('max_cholesky_size', float('inf'))
        self.num_restarts = optimizer_params.get('num_restarts', 10) # 10 if not SMOKE_TEST else 2
        self.raw_samples = optimizer_params.get('raw_samples', 512) # 512 if not SMOKE_TEST else 4
        self.n_candidates = min(5000, max(2000, 200 * self.dimension)) # if not SMOKE_TEST else 4

        self.dtype = bounds.dtype
        self.device = bounds.device

        self.num_calls = 0
        self.best_vals: List[float] = []
    
    def generate_samples_in_region(
        self, 
        num_samples: int,
        path: List[Node],
        region_center: torch.Tensor, # (1, dimension)
        weights: torch.Tensor = None, # (dimension, )
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_init: torch.Tensor = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        bounding_box_length = self.init_bounding_box_length
        weights = weights if weights is not None else \
                    torch.ones(self.dimension, dtype=self.dtype, device=self.device)
    
        sobol_num_samples = 2 * num_samples
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)

        # sobol_samples - (sobol_num_samples, dim)
        sobol_samples = sobol.draw(sobol_num_samples).to(dtype=self.dtype, device=self.device)
        

        while X_init.shape[0] < num_samples and bounding_box_length < 1.0:
            # bounding_box_lbs, bounding_box_ubs - (1, dim)
            bouning_box_lbs = torch.clamp(region_center - bounding_box_length / 2 * weights, 0.0, 1.0)
            bouning_box_ubs = torch.clamp(region_center + bounding_box_length / 2 * weights, 0.0, 1.0)

            sobol_cands = sobol_samples * (bouning_box_ubs - bouning_box_lbs) + bouning_box_lbs
            in_region = Node.path_filter(path, sobol_cands) # (num_in_region_samples, )

            X_init = torch.cat((X_init, sobol_cands[in_region]), dim=0)
            if X_init.shape[0] < num_samples:
                bounding_box_length *= 2
        
        if X_init.shape[0] > num_samples:
            X_init = X_init[:num_samples]
        elif X_init.shape[0] < num_samples:
            # if not enough samples are generated within the path-defined region, generate the rest randomly
            num_rand_samples = num_samples - X_init.shape[0]
            rand_samples = sobol.draw(num_rand_samples).to(dtype=self.dtype, device=self.device)
            X_init = torch.cat((X_init, rand_samples), dim=0)

        Y_init = torch.tensor(
            [self.obj_func(unnormalize(x, self.bounds)) for x in X_init], dtype=self.dtype, device=self.device,
        ).unsqueeze(-1)
        self.best_vals = [max(Y_init).item()] * num_samples

        return X_init, Y_init
        

    def generate_batch(self, 
        state: TurboState,
        model: botorch.models.model.Model, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
    ) -> torch.Tensor:
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

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

        if self.acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True, seed=self.seed)

            # pert - (n_candidates, dim)
            pert = sobol.draw(self.n_candidates).to(dtype=self.dtype, device=self.device)
            pert = tr_lbs + (tr_ubs - tr_lbs) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(self.n_candidates, dim, dtype=self.dtype, device=self.device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cands = x_center.expand(self.n_candidates, dim).clone()
            X_cands[mask] = pert[mask]
            
            # Sample on the candidate set 
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cands, num_samples=self.batch_size)
        elif self.acqf == "ei":
            ei = qExpectedImprovement(model=model, best_f=Y.max())

            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lbs, tr_ubs]),
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        
        return X_next

    def optimize(self, X_in_region: torch.Tensor, Y_in_region: torch.Tensor, num_evals: int, path: List[Node]) -> Tuple[torch.Tensor, torch.Tensor]:
        num_init = min(self.num_init, num_evals)
        region_center: torch.Tensor = X_in_region[torch.argmax(Y_in_region), :].clone()
        X_init, Y_init = self.generate_samples_in_region(
            num_samples=self.num_init,
            path=path,
            region_center=region_center,
        )
        self.num_calls += num_init
        print(f"[TuRBO] Start local modeling with {num_init} data points")
        
        state = TurboState(dim=self.dimension, batch_size=self.batch_size)
        X_sampled = X_init # torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        Y_sampled = Y_init # torch.empty((0, 1), dtype=self.dtype, device=self.device)
        while not state.restart_triggered and self.num_calls < num_evals:
            train_X = X_sampled
            train_Y = standardize(Y_sampled)

            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=Interval(0.005, 4.0),
                ),
            )

            model = SingleTaskGP(
                train_X, train_Y, covar_module=covar_module, likelihood=likelihood
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                try:
                    fit_gpytorch_mll(mll)
                except Exception as ex:
                    print_log(f"[TuRBO] {ex} happens when fitting model, restart.")
                X_next = self.generate_batch(
                    state=state, model=model, X=train_X, Y=train_Y)
            X_next = X_next[:min(self.batch_size, num_evals - self.num_calls)]
            Y_next = torch.tensor(
                [self.obj_func(unnormalize(x, self.bounds)) for x in X_next], 
                dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)

            X_sampled = torch.cat((X_sampled, X_next), dim=0)
            Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)

            # Statistics Update
            self.num_calls += len(X_next)
            self.best_vals.append(max(Y_next).item())
            state = update_state(state, Y_next)

            print_log(
                f"[{len(X_sampled)} in node {path[-1].id}] "
                f"Best value: {state.best_value:.3f} | TR length: {state.length:.3f} | "
                f"num. failures: {state.failure_counter}/{state.failure_tolerance} | "
                f"num. successes: {state.success_counter}/{state.success_tolerance}"
            )

        if state.restart_triggered:
            print_log(f"[TuRBO] Local Modelling converges (TR length below threshold) after {self.num_calls} evaluations")

        return X_sampled, Y_sampled

OPTIMIZER_MAP = {
    'turbo': TuRBO,
}