"""Model configurations. The torch-based parts live in `models.nets` and `models.neural`
and are imported from there, so this package loads without torch installed."""

from .config import LGBMConfig, ModelConfig, TorchConfig

__all__ = ["LGBMConfig", "ModelConfig", "TorchConfig"]
