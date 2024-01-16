import torch
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

dtype = torch.double
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

PROPERTY_SET = {
    'seed', 'Cp', 'leaf_size', 'node_selection_type', 'initial_sampling_method',
    'bounds', 'num_init', 'obj_func', 'optimizer_type', 'optimizer_params',
    'classifier_type', 'classifier_params',
}
class BaseBenchmark:
    def __init__(self, **kwargs):
        self.seed = 0
        self.Cp: float = 1.
        self.leaf_size: int = 20
        self.node_selection_type: str = 'UCB'
        self.initial_sampling_method: str = 'Sobol'
        self.save_path: bool = kwargs.get('save_path', False)

        # Optimizer settings
        self.optimizer_type: str = 'turbo'
        self.optimizer_params: dict =  {
            'batch_size': 4, # Note this is the "local" batch size
            'acqf': 'ts',
            'num_restarts': 10,
            'raw_samples': 512,
        }


        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        res_dict = {}
        for k, v in self.__dict__.items():
            if k in PROPERTY_SET:
                res_dict[k] = v
        return res_dict

class AckleyBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # Classifier settings
        self.classifier_type: str = 'SVM'
        self.classifier_params: dict = {
            'kernel_type': 'rbf',
            'gamma_type': 'auto',
        }

class RosenbrockBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print(f"Using Rosenbrock {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -10.)
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

class LevyBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        print('[LevyBenchmark] Levy Initialized')


class RastriginBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
}

if __name__ == '__main__':
    print(AckleyBenchmark.optimizer_type)
    AckleyBenchmark.optimizer_type = 'test'
    print(AckleyBenchmark.optimizer_type)