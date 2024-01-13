import torch
import gym 
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

dtype = torch.double
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class AckleyBenchmark:
    def __init__(self, **kwargs):
        self.seed = 0
        self.Cp: float = 1.
        self.leaf_size: int = 20
        self.node_selection_type: str = 'UCB'
        self.initial_sampling_method: str = 'Sobol'
        self.save_path: bool = kwargs.get('save_path', False)

        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print(f"Using Ackley {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -5.)
        self.ub: float = kwargs.get('ub', 10.)
        self.bounds: torch.tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=dtype, device=device)
        self.obj_func = Ackley(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=dtype, device=device)

        self.num_init = kwargs.get('num_init', 2 * self.dim)
        
        # Optimizer settings
        self.optimizer_type: str = 'turbo'
        self.optimizer_params: dict =  {
            'batch_size': 4, # Note this is the "local" batch size
            'acqf': 'ts',
            'num_restarts': 10,
            'raw_samples': 512,
        }

        # Classifier settings
        self.classifier_type: str = 'SVM'
        self.classifier_params: dict = {
            'kernel_type': 'rbf',
            'gamma_type': 'auto',
        }

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            'seed': self.seed,
            'Cp': self.Cp,
            'leaf_size': self.leaf_size,
            'node_selection_type': self.node_selection_type,
            'initial_sampling_method': self.initial_sampling_method,
            'save_path': self.save_path,
            'bounds': self.bounds,
            'num_init': self.num_init,
            'obj_func': self.obj_func,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
        }

class RosenbrockBenchmark:
    def __init__(self, **kwargs):
        self.seed = 0
        self.Cp: float = 1.
        self.leaf_size: int = 20
        self.node_selection_type: str = 'UCB'
        self.initial_sampling_method: str = 'Sobol'

        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print(f"Using Rosenbrock {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -9.)
        self.ub: float = kwargs.get('ub', 10.)
        self.bounds: torch.tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=dtype, device=device)
        self.obj_func = Rosenbrock(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=dtype, device=device)

        # Optimizer settings
        self.num_init = kwargs.get('num_init', 2 * self.dim)
        self.optimizer_type: str = 'turbo'
        self.optimizer_params: dict =  {
            'batch_size': 4, # Note this is the "local" batch size
            'acqf': 'ts',
            'num_restarts': 10,
            'raw_samples': 512,
            'max_cholesky_size': 2000,
        }

        # Classifier settings
        self.classifier_type: str = 'SVM'
        self.classifier_params: dict = {
            'kernel_type': "poly",
            'gamma_type': 'auto',
        }


    def to_dict(self):
        return {
            'seed': self.seed,
            'Cp': self.Cp,
            'leaf_size': self.leaf_size,
            'node_selection_type': self.node_selection_type,
            'initial_sampling_method': self.initial_sampling_method,
            'bounds': self.bounds,
            'num_init': self.num_init,
            'obj_func': self.obj_func,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
        }

class LevyBenchmark:
    def __init__(self, **kwargs):
        self.seed = 0
        self.Cp: float = 1.
        self.leaf_size: int = 20
        self.node_selection_type: str = 'UCB'
        self.initial_sampling_method: str = 'Sobol'

        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print(f"Using Levy {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -10.)
        self.ub: float = kwargs.get('ub', 10.)
        self.bounds: torch.tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=dtype, device=device)
        self.obj_func = Levy(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=dtype, device=device)

        # Optimizer settings
        self.num_init = kwargs.get('num_init', 2 * self.dim)
        self.optimizer_type: str = 'turbo'
        self.optimizer_params: dict =  {
            'batch_size': 4, # Note this is the "local" batch size
            'acqf': 'ts',
            'num_restarts': 10,
            'raw_samples': 512,
        }

        # Classifier settings
        self.classifier_type: str = 'SVM'
        self.classifier_params: dict = {
            'kernel_type': 'poly',
            'gamma_type': 'auto',
        }

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        print('[LevyBenchmark] Levy Initialized')

    def to_dict(self):
        return {
            'seed': self.seed,
            'Cp': self.Cp,
            'leaf_size': self.leaf_size,
            'node_selection_type': self.node_selection_type,
            'initial_sampling_method': self.initial_sampling_method,
            'bounds': self.bounds,
            'num_init': self.num_init,
            'obj_func': self.obj_func,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
        }

class RastriginBenchmark:
    def __init__(self, **kwargs):
        self.seed = 0
        self.Cp: float = 1.
        self.leaf_size: int = 20
        self.node_selection_type: str = 'UCB'
        self.initial_sampling_method: str = 'Sobol'

        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print(f"Using Rastrigin {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -5.12)
        self.ub: float = kwargs.get('ub', 5.12)
        self.bounds: torch.tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=dtype, device=device)
        self.obj_func = Rastrigin(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=dtype, device=device)

        self.num_init = kwargs.get('num_init', 2 * self.dim)
        # Optimizer settings
        self.optimizer_type: str = 'turbo'
        self.optimizer_params: dict =  {
            'batch_size': 4, # Note this is the "local" batch size
            'acqf': 'ts',
            'num_restarts': 10,
            'raw_samples': 512,
        }

        # Classifier settings
        self.classifier_type: str = 'SVM'
        self.classifier_params: dict = {
            'kernel_type': 'rbf',
            'gamma_type': 'auto',
        }
        
        print('[Rastrigin] Rastrigin Initialized')

    def to_dict(self):
        return {
            'seed': self.seed,
            'Cp': self.Cp,
            'leaf_size': self.leaf_size,
            'node_selection_type': self.node_selection_type,
            'initial_sampling_method': self.initial_sampling_method,
            'bounds': self.bounds,
            'num_init': self.num_init,
            'obj_func': self.obj_func,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
        }



class Lunarlanding:
    def __init__(self):
        self.dims = 12
        self.lb   = np.zeros(12)
        self.ub   = 2 * np.ones(12)
        self.counter = 0
        self.env = gym.make('LunarLander-v2')
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 50
        self.leaf_size   = 10
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type  = "scale"
        
        self.render      = False
        
        
    def heuristic_controller(self, state, weights):
        angle_targ = state[0] * weights[0] + state[2] * weights[1]
        if angle_targ > weights[2]:
            angle_targ = weights[2]
        if angle_targ < -weights[2]:
            angle_targ = -weights[2]
        hover_targ = weights[3] * np.abs(state[0])

        angle_todo = (angle_targ - state[4]) * weights[4] - (state[5]) * weights[5]
        hover_todo = (hover_targ - state[1]) * weights[6] - (state[3]) * weights[7]

        if state[6] or state[7]:
            angle_todo = weights[8]
            hover_todo = -(state[3]) * weights[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > weights[10]:
            a = 2
        elif angle_todo < -weights[11]:
            a = 3
        elif angle_todo > +weights[11]:
            a = 1
        return a
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
    
        total_rewards = []
        for i in range(0, 3): # controls the number of episode/plays per trial
            state, _ = self.env.reset()
            rewards_for_episode = []
            num_steps = 2000
        
            for step in range(num_steps):
                if self.render:
                    self.env.render()
                received_action = self.heuristic_controller(state, x)
                next_state, reward, done, _, info = self.env.step(received_action)
                rewards_for_episode.append( reward )
                state = next_state
                if done:
                     break
                        
            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append( np.sum(rewards_for_episode) )
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean( total_rewards )
        
        return mean_rewards * -1

class LunarLandingBench:
    def __init__(self, **kwargs):
        self.seed = 0
        self.Cp: float = 1.
        self.leaf_size: int = 20
        self.node_selection_type: str = 'UCB'
        self.initial_sampling_method: str = 'Sobol'

        # Function settings
        self.lunar_env = Lunarlanding()
        self.dim: int = self.lunar_env.dims
        self.bounds: torch.tensor = torch.tensor(
            [[self.lunar_env.lb[0]] * self.dim, [self.lunar_env.ub[0]] * self.dim]
        ).to(dtype=dtype, device=device)

        def obj_func(x: torch.Tensor):
            # x: (num_samples, dim)
            reward = self.lunar_env(x.cpu().numpy())
            return reward
        self.obj_func = obj_func

        # Optimizer settings
        self.num_init = kwargs.get('num_init', 2 * self.dim)
        self.optimizer_type: str = 'turbo'
        self.optimizer_params: dict =  {
            'batch_size': 4, # Note this is the "local" batch size
            'acqf': 'ts',
            'num_restarts': 10,
            'raw_samples': 512,
        }

        # Classifier settings
        self.classifier_type: str = 'SVM'
        self.classifier_params: dict = {
            'kernel_type': 'rbf',
            'gamma_type': 'auto',
        }

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        print('[LunarLanding] LunarLanding Initialized')

    def to_dict(self):
        return {
            'seed': self.seed,
            'Cp': self.Cp,
            'leaf_size': self.leaf_size,
            'node_selection_type': self.node_selection_type,
            'initial_sampling_method': self.initial_sampling_method,
            'bounds': self.bounds,
            'num_init': self.num_init,
            'obj_func': self.obj_func,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
        }



BENCHMARK_MAP = {
    'ackley2d': (AckleyBenchmark, {'dim': 2, 'lb': -10., 'save_path': True}),
    'ackley20d': (AckleyBenchmark, {'dim': 20}),
    'ackley50d': (AckleyBenchmark, {'dim': 50}),
    'ackley100d': (AckleyBenchmark, {'dim': 100}),
    'rosenbrock20d': (RosenbrockBenchmark, {'dim': 20}),
    'rosenbrock50d': (RosenbrockBenchmark, {'dim': 50}),
    'rosenbrock100d': (RosenbrockBenchmark, {'dim': 100}),
    'levy20d': (LevyBenchmark, {'dim': 20}),
    'levy50d': (LevyBenchmark, {'dim': 50}),
    'levy100d': (LevyBenchmark, {'dim': 100}),
    'rastrigin20d': (RastriginBenchmark, {'dim': 20}),
    'rastrigin50d': (RastriginBenchmark, {'dim': 50}),
    'rastrigin100d': (RastriginBenchmark, {'dim': 100}),
    'lunarlanding': (LunarLandingBench, {}),
}

if __name__ == '__main__':
    print(AckleyBenchmark.optimizer_type)
    AckleyBenchmark.optimizer_type = 'test'
    print(AckleyBenchmark.optimizer_type)