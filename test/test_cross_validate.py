import numpy as np
import pytest

from model_forge.model.cross_validate import (
    BaseCrossValidator,
    SimpleCrossValidator,
    GroupCrossValidator,
)


class TestSimpleCrossValidator:
    def test_split(self):
        validator = SimpleCrossValidator(n_splits=3)

        # Don't use the same number twice so we can leverage the set meganism to check if split is correct
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

        splits = list(validator.split(X))

        assert len(splits) == 3
        tot_left_out = []
        for split in splits:
            assert len(split) == 2
            assert len(set(split[0]).intersection(set(split[1]))) == 0
            tot_left_out.extend(list(split[1]))
        # Make sure all data has been part of the left out set
        assert len(tot_left_out) == np.unique(tot_left_out).shape[0]
        assert len(tot_left_out) == X.shape[0]


class TestGroupCrossValidator:
    def test_split_with_groups(self):
        validator = GroupCrossValidator(n_splits=3)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]] * 100)
        groups = np.array([1, 1, 2, 2, 3] * 100)

        splits = list(validator.split(X, groups))

        assert len(splits) == 3
        for split in splits:
            assert len(split) == 2
            assert len(set(groups[split[0]]).intersection(set(groups[split[1]]))) == 0

    def test_split_with_groups_not_enough_groups(self):
        validator = GroupCrossValidator(n_splits=3)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        groups = np.array([1, 1, 2, 2, 2])

        with pytest.raises(ValueError):
            list(validator.split(X, groups))

    def test_split_without_groups(self):
        validator = GroupCrossValidator(n_splits=3)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

        with pytest.raises(ValueError):
            list(validator.split(X))


class TestBaseCrossValidator:
    def test_split_not_implemented(self):
        class NewCrossValidator(BaseCrossValidator):
            def split(self, X):
                return super().split(X)

        validator = NewCrossValidator(n_splits=3)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

        with pytest.raises(NotImplementedError):
            list(validator.split(X))

    def test_basecrossvalidator_usage(self):
        with pytest.raises(TypeError):
            _ = BaseCrossValidator(n_splits=3)
