from unittest import TestCase
from t2b.likelihood import *
import numpy as np
import matplotlib.pyplot as plt
from t2b.constants import Nb_dots


class Test(TestCase):
    def test_spawn_grid0(self):
        dtype = np.uint16
        res = spawn_grid(np.zeros(8, dtype=dtype))
        assert res.shape == (np.prod(Nb_dots), 2)
        assert np.all(res == 0)

    def test_spawn_grid1(self):
        dtype = np.uint16
        estimate = np.array([0, Nb_dots[0] - 1, Nb_dots[0] - 1, 0, 0, 0, Nb_dots[1] - 1, Nb_dots[1] - 1], dtype=dtype)
        res = spawn_grid(estimate)
        expectation = np.array(list(product(*[np.arange(i, dtype=dtype) for i in Nb_dots]))).astype(dtype)
        plt.scatter(*res.T)
        error = ~(res == expectation)
        assert np.all(~error)
        return

    def test_spawn_grid2(self):
        dtype = np.uint16
        estimate = np.array([0, Nb_dots[0], Nb_dots[0], 0, 0, 0, Nb_dots[1] * 2, Nb_dots[1]], dtype=dtype)
        res = spawn_grid_float(estimate)
        res_int = spawn_grid(estimate)
        assert np.all(np.round(res + 1e-6).astype(np.int) == res_int)
        plt.scatter(*res.T)
        return
