import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config import LGBMConfig


def fit(config: LGBMConfig, X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    """Trains a LightGBM regressor with the parameters of `config`; NaN in X is allowed."""
    params = {k: v for k, v in config.to_dict().items() if k != "seed"}
    model = lgb.LGBMRegressor(
        **params, random_state=config.seed, subsample_freq=1, verbose=-1
    )
    return model.fit(X, y)


def predict(model: lgb.LGBMRegressor, X: pd.DataFrame) -> NDArray[np.float64]:
    """Predictions in the units of the training target, one per row of X."""
    return np.asarray(model.predict(X))
