class Predictions(dict):
    """
    A dictionary-like class for storing predictions generated by a model on different data splits.

    Inherits from the built-in `dict` class.

    Methods:
    - create_from_model_dataset: Creates a `Predictions` object from a model and dataset.

    Attributes:
    - pred: A dictionary that stores the predictions for each data split.

    """

    @classmethod
    def create_from_model_dataset(cls, model, dataset):
        """
        Creates a `Predictions` object from a model and dataset.

        Args:
        - model: The model object used for making predictions.
        - dataset: The dataset object containing the data splits.

        Returns:
        - A `Predictions` object containing the predictions for each data split.

        Raises:
        - AttributeError: If the model does not have a 'predict' attribute.

        """
        pred = {}
        for split, (X, _) in dataset:
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                pred[split] = (model.predict(X), model.predict_proba(X))
            elif hasattr(model, "predict"):
                pred[split] = (model.predict(X), None)
            else:
                raise AttributeError("Model does not have 'predict' attribute.")
        return cls(pred)