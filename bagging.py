from copy import deepcopy

import numpy as np

from logistic_regression import accuracy


class BaggingClasifier:
    def __init__(self, base_model, sample_size=100, iters=50):
        self.sample_size = sample_size
        self.iters = iters
        self.base_model = base_model
        self.models = []
        self.weights = []

    def fit(self, x, y):
        models = []
        indicies = np.arange(x.shape[0])
        for i in range(self.iters):
            data = np.concatenate([x, y.reshape((len(y), 1))],
                                  axis=1)
            sample_indicies = np.random.choice(indicies, size=self.sample_size,
                                               replace=False)
            sample = data[sample_indicies, :]
            model = deepcopy(self.base_model)
            model.fit(sample[:, :-1], sample[:, -1])
            models.append(model)
        self.models = models

    def predict(self, x):
        res = []
        for model in self.models:
            res.append(model.predict(x))
        res = np.stack(res, axis=1)
        return np.mean(res, axis=1) > 0.5
