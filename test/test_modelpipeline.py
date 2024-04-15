import pytest
import numpy as np

from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression

from model_forge.model.model_orchastrator import ModelPipeline


class TestModelPipeline:
    """
    A test class for evaluating a model using various metrics.
    """

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = np.array([[1, 1, 1], [1, 1, 1], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        self.y = np.array([0, 0, 1, 1, 1])
        self.model = LogisticRegression()
        model_step = OmegaConf.create({"_target_": "sklearn.linear_model.LogisticRegression", "C": 0.1})
        self.cfg = OmegaConf.create({
            "model": {"model_steps":  [{"m": model_step}]},
        })

    def test_modelpipeline(self):
        
        pipe = ModelPipeline.create_from_config(self.cfg)
        assert pipe.stepnames == ["m"]
 