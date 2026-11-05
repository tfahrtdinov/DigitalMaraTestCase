from .protocol import NeuralModel
from .regression import RegressionModel
from .rnn import RNNModel

MODELS = {"regression": RegressionModel, "rnn": RNNModel}

__all__ = ["MODELS", "NeuralModel", "RegressionModel", "RNNModel"]
