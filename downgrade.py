import numpy as np


def pca(x):
    x_c = x - x.mean(axis=0)
    c = (x_c.T @ x_c) / (x.shape[0] - 1)
    l, v = np.linalg.eig(c)
    l = l.real
    v = v.real
    idx = np.argsort(l)[::-1]
    l = l[idx]
    v = v[:, idx]
    r = np.where(np.cumsum(l / l.sum()) > 0.99)
    return v[:, :r[0][0]]
