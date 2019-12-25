import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def h(theta, x):
    return sigmoid(x @ theta).T


def predict(theta, x):
    return h(theta, x) > 0.5


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def get_features(x, power):
    # new_x = np.zeros(shape=(x.shape[0], x.shape[1] * x.shape[1] * power * power))
    # row = 0
    new_x = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            for p1 in range(power + 1):
                for p2 in range(power + 1):
                    if p1 + p2 <= power:
                        # print(f"x{i}^{p1} * x{j}^{p2}")
                        new_x.append(
                            np.power(x[:, i], p1) * np.power(x[:, j], p2))
    return np.array(new_x).T


def normalize(d):
    for i in range(len(d[0])):
        col = d[:, i]
        dif = col.max() - col.min()
        if dif != 0:
            d[:, i] = (col - col.min()) / dif
    return d


class LogisticRegression:
    def __init__(self, learning_rate=3e-1, iters=10000):
        self.learning_rate = learning_rate
        self.iters = iters
        self.theta = None

    def grad(self, x, y):
        m = x.shape[0]
        self.theta = self.theta - (self.learning_rate / m) * (
                (h(self.theta, x) - y) @ x).T

    def fit(self, x, y):
        self.theta = np.zeros(shape=(x.shape[1], 1))
        for i in range(self.iters):
            self.grad(x, y)

    def predict(self, x):
        p = predict(self.theta, x)
        return p.reshape((p.shape[1],))
