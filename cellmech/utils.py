import numpy as np


def symmetric_gaussian(v: np.ndarray[int, int]):
    r2 = v[0]**2 + v[1]**2
    r = np.sqrt(r2)
    A = 1 * np.exp(-10*r2)
    cos = -v[0]/r
    sin = -v[1]/r
    return np.array([A*cos, A*sin])


def simple_unit_force(v: np.ndarray[int, int]):
    return v/(np.sqrt(v[0]**2 + v[1]**2))
    # return np.array([v[1], -v[0]])/(np.sqrt(v[0]**2 + v[1]**2))