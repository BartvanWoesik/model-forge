import pytest
import numpy as np

from sklearn.metrics import get_scorer
from sklearn.linear_model import LogisticRegression

from model_forge.model.model_evaluator import ModelEvaluator
from model_forge.model.cross_validate import SimpleCrossValidator


@pytest.mark.unit
class TestModelEvaluator:
    """
    A test class for evaluating a model using various metrics.
    """

    metric_names = ["accuracy", "precision", "recall"]

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = np.array(
            [[1, 1, 1], [1, 1, 1], [10, 10, 10], [10, 10, 10], [10, 10, 10]] * 100
        )
        self.y = np.array([0, 0, 1, 1, 1] * 100)
        self.model = LogisticRegression()
        self._metrics = {metric: get_scorer(metric) for metric in self.metric_names}
        self.evaluator = ModelEvaluator(metrics=self._metrics)

    def test_evaluate(self):
        expected_results = {
            "accuracy": 1,
            "precision": 1,
            "recall": 1,
        }

        results = self.evaluator.evaluate(
            model=self.model, cv_strat=SimpleCrossValidator(2), X=self.X, y=self.y
        )

        assert results["accuracy"] == expected_results["accuracy"]
        assert results["precision"] == expected_results["precision"]
        assert results["recall"] == expected_results["recall"]

    def test_metrics_property(self):
        assert self.evaluator.metrics == self.metric_names
