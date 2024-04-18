from typing import Dict, Callable, List

import numpy as np

from sklearn.model_selection import cross_validate

from model_forge.model.model_orchastrator import CustomPipeline


class ModelEvaluator:
    """
    A class for evaluating machine learning models using cross-validation and multiple metrics.

    Attributes:
    - sklearn_metrics (List[str]): A list of sklearn metrics to be used for evaluation.
    - custom_scorers (dict, optional): A dictionary of custom scorers. Defaults to None.
    - cv (int, optional): The number of cross-validation folds. Defaults to 5.
    """

    def __init__(self, metrics: dict[str, Callable], cv: int = 5) -> None:
        """
        Initialize the MetricEvaluator class.

        Parameters:
        - sklearn_metrics (List[str]): A list of sklearn metrics to be used for evaluation.
        - metrics (dict, optional): A dictionary of metric scorers. Defaults to an empty dict.
        - cv (int, optional): The number of cross-validation folds. Defaults to 5.
        """
        self._metrics = metrics
        self.cv = cv

    @property
    def metrics(self) -> List[str]:
        """
        Get a list of the evaluation metrics.

        Returns:
        - metric (Dict[str, Callable]): A mapping of evaluation metrics.
        """
        return list(self._metrics.keys())

    def evaluate(
        self, model: CustomPipeline, X: np.array, y: np.array
    ) -> Dict[str, float]:
        """
        Evaluate the model using cross-validation and multiple metrics.

        Parameters:
        - model: The machine learning model to be evaluated.
        - X (array-like): The input features.
        - y (array-like): The target variable.

        Returns:
            A dictionary containing the evaluation results for each metric.
        """
        # TODO: re-write cross_validate to minimize dependence on sklearn
        return cross_validate(
            estimator=model, X=X, y=y, scoring=self._metrics, cv=self.cv
        )
