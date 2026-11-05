from .protocol import NeuralModel
from .regression import RegressionModel

MODELS = {"regression": RegressionModel}

__all__ = ["MODELS", "NeuralModel", "RegressionModel"]
