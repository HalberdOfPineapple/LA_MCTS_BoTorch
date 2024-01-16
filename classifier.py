import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List

from sklearn.cluster import KMeans
from sklearn.svm import SVC

class BaseClassifier:
    @abstractmethod
    def cluster(self, X: torch.Tensor, Y: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, save_path: str):
        raise NotImplementedError
    
    @property
    def pos_label(self) -> int:
        raise NotImplementedError

class SVMClassifier(BaseClassifier):
    def __init__(self, bounds: torch.Tensor, seed: int, classifier_params: Dict) -> None:
        self.bounds = bounds

        self.seed = seed
        self.kmeans = KMeans(n_clusters=2, n_init='auto', random_state=seed)
        self.svm = SVC(
            kernel=classifier_params['kernel_type'], 
            gamma=classifier_params['gamma_type'],
            random_state=seed)
    
        self._pos_label: int = None

    @property
    def pos_label(self) -> int:
        return self._pos_label

    def cluster(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """

        # X_and_Y - (num_samples, dimension + 1)
        X_and_Y = torch.cat((X, Y), dim=1)
        X_and_Y = X_and_Y.detach().cpu().numpy()

        self.kmeans = self.kmeans.fit(X_and_Y)
        labels = self.kmeans.predict(X_and_Y) # (num_samples, )
        # print(f"Cluster labels: {labels}")

        one_mean = Y[labels == 1].mean(axis=0) # (1, )
        zero_mean = Y[labels == 0].mean(axis=0) # (1, )
        
        # self.pos_label indicate the label of the positive (good) class
        self._pos_label = 1 if zero_mean < one_mean else 0

        return labels

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)
            Y: torch.Tensor - (num_samples, 1)
        """
        # Because the data essentially with association to the performance is the unnormalized ones,
        # we need to denormalize the data before feeding into the KMeans and SVM
        
        # X_scaled - (num_samples, dimension)        
        X_scaled = X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        # labels: ndarray with shape (num_samples, )
        labels = self.cluster(X_scaled, Y)
        self.svm = self.svm.fit(X_scaled.detach().cpu().numpy(), labels)

    
    def predict(self, X: torch.Tensor) -> np.array:
        """
        Args:
            X: torch.Tensor - (num_samples, dimension)

        Returns:
            labels: ndarray with shape (num_samples, )
        """
        X_scaled = X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return self.svm.predict(X_scaled.detach().cpu().numpy())

    def save(self, save_path: str):
        import pickle
        with open(save_path,'wb') as f:
            pickle.dump(self.classifier.svm, f)
    
CLASSIFIER_MAP = {
    'svm': SVMClassifier,
}