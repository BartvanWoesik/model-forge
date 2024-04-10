from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
import numpy as np
from typing import Dict, List

from model_forge.model.modelorchastrator import CustomPipeline


class ModelEvaluator:
    """
    A class for evaluating machine learning models using cross-validation and multiple metrics.

    Attributes:
    - sklearn_metrics (List[str]): A list of sklearn metrics to be used for evaluation.
    - custom_scorers (dict, optional): A dictionary of custom scorers. Defaults to None.
    - cv (int, optional): The number of cross-validation folds. Defaults to 5.
    """


    def __init__(self, sklearn_metrics: List[str], custom_scorers: dict = None, cv: int = 5) -> None:
        """
        Initialize the MetricEvaluator class.

        Parameters:
        - sklearn_metrics (List[str]): A list of sklearn metrics to be used for evaluation.
        - custom_scorers (dict, optional): A dictionary of custom scorers. Defaults to None.
        - cv (int, optional): The number of cross-validation folds. Defaults to 5.
        """
        self.sklearn_metrics = sklearn_metrics
        if custom_scorers is None:
            self.custom_scorers = {}
        else:
            self.custom_scorers = custom_scorers
        self.cv = cv

    @property
    def metrics(self) -> List[str]:
        """
        Get the list of evaluation metrics.

        Returns:
        - metric_list (List[str]): A list of evaluation metrics.
        """
        metric_list = self.sklearn_metrics
        if self.custom_scorers:
            metric_list.append(list(self.custom_scorers.keys()))
        return metric_list

    def evaluate(self, model: CustomPipeline,  X: np.array, y: np.array) -> Dict[str, float]:
        """
        Evaluate the model using cross-validation and multiple metrics.

        Parameters:
        - model: The machine learning model to be evaluated.
        - X (array-like): The input features.
        - y (array-like): The target variable.

        Returns:
        - results (dict): A dictionary containing the evaluation results for each metric.
        """
        scoring = {}
        for metric in self.sklearn_metrics:
            scoring[metric] = get_scorer(metric)
        for metric in list(self.custom_scorers.keys()):
            scoring[metric] = self.custom_scorers[metric]

        results = cross_validate(model, X, y, scoring=scoring, cv=self.cv)
        return results
