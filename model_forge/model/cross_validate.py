import numpy as np
from abc import ABC, abstractmethod


class BaseCrossValidator(ABC):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    @abstractmethod
    def split(self, X):
        pass

    @staticmethod
    def _num_samples(X):
        return X.shape[0]


class SimpleCrossValidator(BaseCrossValidator):
    def split(self, X):
        indices = np.arange(self._num_samples(X))

        for test_index in self._iter_test_indices(X):
            train_index = np.setdiff1d(indices, test_index, assume_unique=True)
            yield train_index, test_index

    def _iter_test_indices(self, X):
        n_samples = self._num_samples(X)
        indices = np.arange(n_samples)

        for test_index in np.array_split(indices, self.n_splits):
            yield test_index


class GroupCrossValidator(BaseCrossValidator):
    def split(self, X: np.array, group: np.array = None):
        if group is None:
            raise ValueError("Groups must be provided for GroupCrossValidator")
        unique_groups = np.unique(group)
        n_samples = self._num_samples(X)
        indices = np.arange(n_samples)

        for test_group in np.array_split(unique_groups, self.n_splits):
            test_index = np.nonzero(np.isin(group, test_group))[0]
            train_index = np.setdiff1d(indices, test_index, assume_unique=True)
            yield train_index, test_index
