from glob import glob
from unittest import TestCase

from t2b.c_funs import likelihood, gen_index, gen_all_indexes, _likelihood
from t2b.constants import corrections
from t2b.main import *
from t2b.tools import rot_matrix


class Test(TestCase):
    @property
    def images(self):
        return glob("*.jpg")

    def test_load_image(self):
        for image in self.images:
            im, (image, points) = load_image(image, return_info=True)

            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(image)
            axes[0].scatter(*points.T)
            axes[1].imshow(im)
            pass

    def test_find_images_coordinates(self):
        for image in self.images:
            im = load_image(image)
            find_image_coordinates(im)

    def test_find_grid_coordinates(self):
        for image in self.images:
            im = load_image(image)
            coord = find_image_coordinates(im)
            grid = find_grid_coordinates(coord)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(im)
            ax.scatter(*coord.T)
            pass

    def test_gen_index(self):
        shape = (600, 800)
        margin = 25
        scale = (np.array(shape) - margin * 2) / Nb_dots[::-1]
        scale = scale.astype(np.uint32)
        x, y = [np.arange(i) * s for i, s in zip(Nb_dots[::-1], scale)]
        xy = np.array(list(product(x, y)))
        angle = np.deg2rad(-0.5)
        R = rot_matrix(angle)
        xy = R.dot(xy.T).T
        xy += np.array([margin] * 2)[None]
        xy = xy.astype(np.uint)

        res = []
        for i in range(x.size):
            for ii in range(y.size):
                _res = gen_index(i, ii, margin, margin, scale[0], scale[1], angle)
                res.append([_res["x"], _res["y"]])
        res = np.array(res)
        assert np.abs(res - xy).max() == 0

    def test_likelihood(self):
        shape = (800, 600)
        img = np.zeros(shape)
        margin = 25
        scale = (np.array(shape) - margin * 2) / Nb_dots
        scale = scale.astype(np.uint32)
        x, y = [np.arange(i) * s for i, s in zip(Nb_dots, scale)]
        xy = np.array(list(product(x, y)))
        angle = np.deg2rad(-0.5)
        R = rot_matrix(angle)
        xy = R.dot(xy.T).T
        xy += np.array([margin] * 2)[None]
        xy = xy.astype(np.uint)
        img[xy[:, 0], xy[:, 1]] = 1
        img = normalize(convolve2d(img, make_gaussian_kernel(10), mode="same")).astype(np.float64)
        # indexes = gen_all_indexes(margin,margin,scale[0],scale[1],angle)
        # plt.imshow(img.T,origin="lower")
        # plt.scatter(*indexes.T,facecolor="none",edgecolors="r",s=200)
        cost = _likelihood(margin, margin, scale[0], scale[1], angle, img)
        assert cost == 1
        angle = np.deg2rad(np.arange(-1, 1, 0.1)).astype(np.float64)
        start = np.arange(margin - 10, margin + 10).astype(np.uint32)
        size = np.arange(18, 25).astype(np.float64)
        shape = (start.size, start.size, size.size, size.size, angle.size)
        res = np.zeros(shape, dtype=np.float64) - 1
        likelihood(start,start, size,size, angle, img, res.ravel())
        assert res.max() == 1
        pass

    def test_find_grid_coordinates2(self):
        for image in self.images:
            im = load_image(image)
            grid = find_grid_coordinates2(im)

            coord = find_image_coordinates(im)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(im)
            ax.scatter(*coord.T)
            pass

    def test_correction(self):
        for image in self.images:
            image = load_image(image)
            coord = find_image_coordinates(image)
            grid = find_grid_coordinates(coord)
            result = evaluate(grid, coord, corrections[0] == 6)

            plot_result(image, grid, coord, result, debug=False)
            pass
