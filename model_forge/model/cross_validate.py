import numpy as np
from abc import ABC, abstractmethod


class BaseCrossValidator(ABC):
    """
    Base class for cross-validation strategies.

    Parameters:
    -----------
    n_splits : int, optional (default=5)
        Number of folds. Determines the number of times the data will be split.
    """

    def __init__(self, n_splits=5, random_state=1):
        self.n_splits = n_splits
        self.random_state = random_state

    @abstractmethod
    def split(self, X):
        """
        Generate indices to split data into training and test sets.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        splits : generator
            Yields the indices of the training and test sets for each fold.
        """
        raise NotImplementedError("split method must be implemented")

    @staticmethod
    def _num_samples(X):
        """
        Return the number of samples in the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        num_samples : int
            The number of samples in the input data.
        """
        return X.shape[0]


class SimpleCrossValidator(BaseCrossValidator):
    """
    A simple cross-validator implementation.

    This class provides a basic implementation of a cross-validator
    for splitting data into training and testing sets.

    Parameters:
    -----------
    n_splits : int, default=5
        The number of splits to generate.

    Methods:
    --------
    split(X)
        Generate indices to split data into training and test sets.

    """

    def split(self, X):
        indices = np.arange(self._num_samples(X))
        if self.random_state is not None:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        for test_index in self._iter_test_indices(indices):
            train_index = np.setdiff1d(indices, test_index, assume_unique=True)
            yield train_index, test_index

    def _iter_test_indices(self, indices):
        for test_index in np.array_split(indices, self.n_splits):
            yield test_index


class GroupCrossValidator(BaseCrossValidator):
    """
    Cross-validator for splitting data into training and test sets based on groups.

    Parameters:
    -----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    Attributes:
    -----------
    n_splits : int
        The number of folds.

    Methods:
    --------
    split(X: np.array, group: np.array = None)
        Generate indices to split data into training and test sets.

    """

    def split(self, X: np.array, group: np.array = None):
        """
        Generate indices to split data into training and test sets.

        Parameters:
        -----------
        X : np.array
            The input data array.

        group : np.array, optional
            The array containing group labels for each sample. If not provided, an error will be raised.

        Returns:
        --------
        train_index : np.array
            The indices of the training set.

        test_index : np.array
            The indices of the test set.

        Raises:
        -------
        ValueError
            If `group` is not provided.

        """

        if group is None:
            raise ValueError("Groups must be provided for GroupCrossValidator")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            np.random.shuffle(group)

        unique_groups = np.unique(group)
        n_samples = self._num_samples(X)
        indices = np.arange(n_samples)
        if len(unique_groups) < self.n_splits:
            raise ValueError(
                f"Number of groups ({len(unique_groups)}) is less than the number of splits ({self.n_splits})"
            )

        # Split the data based on the unique groups
        for test_group in np.array_split(unique_groups, self.n_splits):
            # Get the indices of the test group
            test_index = np.nonzero(np.isin(group, test_group))[0]
            # Get the indices of the training group
            train_index = np.setdiff1d(indices, test_index, assume_unique=True)
            yield train_index, test_index
