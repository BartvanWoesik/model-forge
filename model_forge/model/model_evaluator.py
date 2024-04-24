from typing import Dict, Callable, List

import numpy as np


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
        self, model: CustomPipeline, cv_strat, X: np.array, y: np.array
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

        scores_dict = {metric: [] for metric in self._metrics.keys()}

        X = np.array(X)
        y = np.array(y)
        for train_index, val_index in cv_strat.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)

            kwargs = {"estimator": model, "X": X_val, "y_true": y_val}
            for metric, func in self._metrics.items():
                score = func(**kwargs)
                scores_dict[metric].append(score)

        for metric in scores_dict.keys():
            scores_dict[metric] = np.mean(scores_dict[metric])

        return scores_dict
