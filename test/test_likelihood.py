from glob import glob
from os.path import abspath, dirname, join
from unittest import TestCase

import cv2

from t2b.likelihood import *
import numpy as np
import matplotlib.pyplot as plt
from t2b.constants import Nb_dots
from t2b.main import *


class Test(TestCase):
    def test_spawn_grid0(self):
        res = spawn_grid(np.zeros(8, dtype=estimate_dtype))
        assert res.shape == (np.prod(Nb_dots), 2)
        assert np.all(res == 0)

    def test_spawn_grid1(self):
        dtype = estimate_dtype
        estimate = np.array([0, Nb_dots[0] - 1, Nb_dots[0] - 1, 0, 0, 0, Nb_dots[1] - 1, Nb_dots[1] - 1], dtype=dtype)
        res = spawn_grid(estimate)
        expectation = np.array(list(product(*[np.arange(i, dtype=dtype) for i in Nb_dots]))).astype(dtype)
        plt.scatter(*res.T)
        error = ~(res == expectation)
        assert np.all(~error)
        return

    def test_spawn_grid2(self):
        dtype = estimate_dtype
        estimate = np.array([0, Nb_dots[0], Nb_dots[0], 0, 0, 0, Nb_dots[1] * 2, Nb_dots[1]], dtype=dtype)
        res = spawn_grid_float(estimate)
        res_int = spawn_grid(estimate)
        assert np.all(np.round(res + 1e-6).astype(np.int) == res_int)
        plt.scatter(*res.T)
        return

    @property
    def default_estimate(self):
        return np.array([50, 800, 800, 50, 50, 50, 600, 600], dtype=estimate_dtype)

    def get_test_image(self, estimate=None, sigma=3):
        if estimate is None:
            estimate = self.default_estimate
        coord = spawn_grid(estimate.astype(estimate_dtype))
        max = coord.max(0)
        shape = max * 1.1
        res = np.zeros(tuple(shape.astype(int)))
        res[coord[:, 0], coord[:, 1]] = 255
        res_blur = cv2.GaussianBlur(res, (sigma * 6 + 1,) * 2, sigma)
        res_blur /= (res_blur.max() / 255)
        return res_blur.astype(np.uint8)

    def test_likelihood(self):
        estimate = self.default_estimate
        img = self.get_test_image(estimate, sigma=4)
        cost, grad = likelihood(estimate, img)
        best_cost = img.max() * np.prod(Nb_dots)
        assert cost == best_cost
        assert np.all(grad == 0)

        dimg = diff_image(img)
        estimate = estimate.reshape((2, -1))
        estimate[0, :] -= 3
        grid = spawn_grid(estimate.ravel())
        cost, grad = likelihood(estimate.ravel(), img)
        assert cost < best_cost
        grad = grad.reshape(2, -1)
        assert np.all(grad[0] > 10)
        assert np.all(grad[1] == 0)
        pass

    def test_diff_image(self):
        img = self.get_test_image()
        print(img.shape)
        dimg = diff_image(img)
        assert np.all(dimg[50, 50, :] == 0)
        plt.imshow(dimg[:, :, 0])
        pass

    def get_test_dimg(self, *args, **kwargs):
        return diff_image(self.get_test_image(*args, **kwargs))

    def test_gradient(self):
        est = self.default_estimate
        est = (est.reshape(2, -1) - np.array([3, 0], dtype=np.uint8)[:, None]).ravel()
        grid = spawn_grid(est)
        dimg = self.get_test_dimg()
        print(f"img shape: {dimg.shape}")
        print(get_default_config(dimg))
        res = np.zeros((np.prod(Nb_dots), 8), dtype=np.float32)
        for i, (x, y) in enumerate(product(*[range(i) for i in Nb_dots])):
            dot = dict(zip("xy", grid[i]))
            # print(f"expected_index: {np.ravel_multi_index(tuple(np.transpose([grid[i].tolist()+[0]])),dimg.shape)[0]}")
            gradient(x, y, dot, dimg, res[i])
        pass

    def test_iterate_estimate(self):
        est = self.default_estimate
        img = self.get_test_image(est)
        est = (est.reshape(2, -1) - np.array([3, 0], dtype=np.uint8)[:, None]).ravel()
        dimg = diff_image(img)
        cost, grad = likelihood(est, img)
        iterate_estimate(est, img, dimg.ravel(), 30, 1 / (10 * 255), 0.5)
        est = np.round(est).astype(np.uint16)

    @property
    def images(self):
        for i in glob(join(dirname(abspath(__file__)), "test*/*")):
            image = load_image(i, page_extract=False)
            imbw = img_to_bw(image)
            ks = guess_kernel_size(imbw)
            imc = match_filter_image(imbw, ks, int(np.sqrt(ks)))
            yield normalize(imc)

    def test_iterate_estimate2(self):
        for i, img in enumerate(self.images):
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(f"/tmp/{i}.jpg", img)
            dimg = diff_image(img)

            estimate = np.reshape([100, 100, 700, 100, 700, 500, 100, 500], (-1, 2))
            estimate = estimate[None] + np.random.uniform(-100, 100, (100,) + estimate.shape)
            # estimate = np.random.uniform(0, 1, (4, 2)) * np.array(img.shape)[None, :]
            estimate = np.moveaxis(estimate, 1, -1).reshape(-1, 8).astype(np.float32)

            iterate_estimate(estimate, img, dimg.ravel(), 2000, 1 / (10 * 255), 0.5)

            plt.imshow(img.T, origin="lower")
            plot_lines = np.moveaxis(estimate.reshape(estimate.shape[0], 2, -1), [1,0], [0,-1])
            plt.plot(*plot_lines)

            pass
