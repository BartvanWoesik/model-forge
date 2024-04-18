class Predictions(dict):
    @classmethod
    def create_from_model_dataset(cls, model, dataset):
        pred = {}
        for split, (X, _) in dataset:
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                pred[split] = (model.predict(X), model.predict_proba(X))
            elif hasattr(model, "predict"):
                pred[split] = (model.predict(X), None)
            else:
                raise AttributeError("Model does not have 'predict' attribute.")
        return cls(pred)
