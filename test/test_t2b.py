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
