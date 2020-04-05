from glob import glob
from unittest import TestCase

from t2b.c_funs import likelihood
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

    def test_likelihood(self):
        shape = (600, 800)
        img = np.zeros(shape)
        margin = 25
        scale = (np.array(shape) - margin * 2) / Nb_dots[::-1]
        x, y = [np.arange(i) * s for i, s in zip(Nb_dots[::-1], scale)]
        xy = np.array(list(product(x, y)))
        R = rot_matrix(np.deg2rad(-0.5))
        xy = R.dot(xy.T).T
        xy += np.array([margin] * 2)[None]
        xy = xy.astype(np.uint)
        img[xy[:, 0], xy[:, 1]] = 1
        img = normalize(convolve2d(img, make_gaussian_kernel(10), mode="same")).astype(np.float64)
        plt.imshow(img)
        angle = np.arange(-1, 1, 0.1).astype(np.float64)
        start = np.arange(margin - 10, margin + 10).astype(np.uint32)
        size = np.arange(18, 25).astype(np.uint32)
        shape = (start.size, start.size, size.size, size.size, angle.size)
        res = np.zeros(shape, dtype=np.float64) - 1
        likelihood(start, size, angle, img, res.ravel())
        plt.imshow(res[:, :, -1, -1, 0], extent=[start.min(), start.max(), start.min(), start.max()])
        pass

    def test_find_grid_coordinates2(self):
        for image in self.images:
            im = load_image(image)
            coord = find_image_coordinates(im)
            grid = find_grid_coordinates2(im)

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
