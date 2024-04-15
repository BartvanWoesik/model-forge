import pytest
import numpy as np

from sklearn.metrics import get_scorer
from sklearn.linear_model import LogisticRegression

from model_forge.model.model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """
    A test class for evaluating a model using various metrics.
    """
    metric_names = ["accuracy", "precision", "recall"]

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = np.array([[1, 1, 1], [1, 1, 1], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        self.y = np.array([0, 0, 1, 1, 1])
        self.model = LogisticRegression()
        self._metrics = {metric: get_scorer(metric) for metric in self.metric_names}
        self.evaluator = ModelEvaluator(metrics=self._metrics, cv=2)

    def test_evaluate(self):
        expected_results = {
            "accuracy": [1.0, 1.0],
            "precision": [1.0, 1.0],
            "recall": [1.0, 1.0],
        }

        results = self.evaluator.evaluate(model=self.model, X=self.X, y=self.y)

        assert results["test_accuracy"].tolist() == expected_results["accuracy"]
        assert results["test_precision"].tolist() == expected_results["precision"]
        assert results["test_recall"].tolist() == expected_results["recall"]

    def test_metrics_property(self):
        assert self.evaluator.metrics == self.metric_names
