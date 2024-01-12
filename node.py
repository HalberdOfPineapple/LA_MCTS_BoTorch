import torch
import math
import numpy as np

from typing import Dict, Tuple, List
from classifier import CLASSIFIER_MAP, BaseClassifier

class Node:
    obj_counter: int = 0

    def __init__(
        self, 
        parent: 'Node', 
        label: int,
        bounds: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        classifier_type: str, 
        classifier_params: Dict,
        seed: int,
    ):
        self.id = Node.obj_counter
        Node.obj_counter += 1

        self.parent = parent
        self.label = label
        if parent is not None:
            self.parent.children.append(self)

        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.dtype = dtype
        self.device = device

        self.seed = seed
        self.classifier: BaseClassifier = CLASSIFIER_MAP[classifier_type.lower()](
            bounds=bounds,
            seed=self.seed, 
            classifier_params=classifier_params)

        self.sample_bag: List[torch.Tensor, torch.Tensor] = [
            torch.empty((0, self.dimension), dtype=dtype, device=device),
            torch.empty((0, 1), dtype=dtype, device=device),
        ]

        self.children: List[Node] = []
    
    @property
    def num_visits(self) -> int:
        return len(self.sample_bag[0])
    
    def get_UCB(self, Cp: float) -> float:
        if self.parent is None:
            return 0.
        
        exploit_value = torch.mean(self.sample_bag[1]).detach().cpu().item()
        exploration_value = 2. * Cp * \
            math.sqrt(2. * math.log(self.parent.num_visits) / self.num_visits)
        
        return exploit_value + exploration_value


    def update_sample_bag(self, X: torch.Tensor, Y: torch.Tensor):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows but got {} and {}".format(X.shape[0], Y.shape[0]))
        
        self.sample_bag[0] = torch.cat((self.sample_bag[0], X), dim=0)
        self.sample_bag[1] = torch.cat((self.sample_bag[1], Y), dim=0)
    
    
    def fit(self
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int]:
        if self.sample_bag[0].shape[0] == 0:
            raise ValueError("Fit is called when the current node's sample bag is empty")

        # labels = self.classifier.fit(self.sample_bag[0], self.sample_bag[1])
        self.classifier.fit(self.sample_bag[0], self.sample_bag[1])

        # Let the trained SVM determine the positive and negative samples instead of KMeans
        # One thing for sure is that pos_data will be classified to pos_label while neg_data will be classified to 1 - pos_label
        labels = self.classifier.predict(self.sample_bag[0])
        pos_data = (
            self.sample_bag[0][labels == self.classifier.pos_label],
            self.sample_bag[1][labels == self.classifier.pos_label])
        neg_data = (
            self.sample_bag[0][labels != self.classifier.pos_label],
            self.sample_bag[1][labels != self.classifier.pos_label])

        return pos_data, neg_data, self.classifier.pos_label

    def predict_label(self, X: torch.Tensor) -> np.array: 
        # labels - (num_samples, )
        labels = self.classifier.predict(X)
        return labels
    
    
