import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class RF1std:
    def __init__(self):
        # When predictions are parallelized outside scikit-learn (e.g. joblib over bundles),
        # using n_jobs=-1 here can massively oversubscribe CPUs.
        # Override if desired: export RUAL_RF_N_JOBS=-1
        n_jobs = _get_int_env("RUAL_RF_N_JOBS", 1)
        self.model = RandomForestRegressor(n_jobs=n_jobs)
        self.model.fit(np.random.randint(2, size=[1000, 2048]), np.random.randint(10, size=1000))

    def predict(self, X):
        return self.model.predict(X)
    

class NN1:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=[200, 200],
                                  tol=0.1,
                                  early_stopping=True,
                                  warm_start=True,
                                  alpha=0.01
                                  )
        self.model.fit(np.random.randint(2, size=[1000, 2048]), np.random.randint(10, size=1000))

    def predict(self, X):
        return self.model.predict(X)
