from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """Base of all model configurations. `to_dict` is what hyperparameter search and logging read."""

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LGBMConfig(ModelConfig):
    objective: str = "l1"
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 0.0


@dataclass
class TorchConfig(ModelConfig):
    net: str = "mlp"  # key of models.nets.NETS
    net_params: dict = field(default_factory=dict)
    model: str = (
        "regression"  # key of models.neural.MODELS: loss, prediction, optimiser
    )
    model_params: dict = field(default_factory=dict)  # e.g. lr, weight_decay
    epochs: int = 30
    batch_size: int = 1024
    threads: int = 1
