from glob import glob
from unittest import TestCase

from t2b.constants import corrections
from t2b.main import *


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

    def test_unravel_index(self):
        index = np.arange(6 * 8)
        shape = (6, 8)
        res = np.zeros((6 * 8, 2), dtype=np.uint64)
        for i in index:
            unravel_index(i, shape, res[i])
        assert np.all(np.array([np.unravel_index(index, shape)])[0].T == res)

    def test_likelihood(self):
        from t2b.c_funs import likelihood
        for image in self.images:
            image = load_image(image).mean(-1)
            res = likelihood(0, 0, 10, 10, 0, image)
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
