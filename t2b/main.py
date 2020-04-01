from itertools import product

import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from imutils.perspective import four_point_transform
from scipy.signal import convolve2d

from t2b.constants import Nb_dots


def make_gaussian_kernel(kernel_size, sigma=5):
    x = np.linspace(-1, 1, kernel_size)
    x = np.array(np.broadcast_arrays(x[:, None], x[None, :]))
    kernel = np.exp(-0.5 * sigma * np.linalg.norm(x - np.array([0, 0])[:, None, None], axis=0))
    kernel /= kernel.max()
    return kernel


def find_image_coordinates(image):
    shape = image.shape
    dots = image.std(-1)
    kernel_size = int(np.min(shape[:-1]) / 100)
    if kernel_size > 1:
        kernel = make_gaussian_kernel(kernel_size)
        dots = convolve2d(dots, kernel, mode="same")

    # On choisit un seuil de trigger à 98%
    trigger = np.sort(dots.ravel())[int(dots.size * 0.98)]
    dots = dots > trigger

    cnt = cv2.findContours(dots.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    coord = np.round(np.array([np.median(i[:, 0, :], 0) for i in cnt])).astype(np.uint)
    coordr = coord / np.array(shape[:2])[None, ::-1]

    # On filtre les points qui sont trop près des bords
    margin = 0.01
    selector = np.logical_and(coordr > margin, coordr < (1 - margin)).all(1)

    return coord[selector]


def rotate_image(image, angle, center=None):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def find_grid_coordinates(coord):
    box = np.array([coord.min(0), coord.max(0)])
    grid = np.array(list(product(*[np.linspace(0, 1, n) * (stop - start) + start
                                   for n, (start, stop) in zip(Nb_dots, box.T)])))

    return grid


def evaluate(grid, coord, correction):
    # Find coordinates of each dot
    d = np.linalg.norm(coord[:, None] - grid[None, :], axis=-1)
    dot_index = np.argmin(d, axis=1)

    is_correct = correction.ravel()[dot_index]

    nb_found = np.sum(is_correct)
    result = dict(nombre_trouves=nb_found, nombre_erreurs=np.sum(~is_correct),
                  nb_omissions=np.sum(correction) - nb_found, is_correct=is_correct, correction=correction,
                  index=dot_index)

    return result


def plot_result(image, grid, coord, result, ax=None, debug=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(30//2, 21//2))
    correct, correction, index = result["is_correct"], result["correction"], result["index"]
    ax.imshow(image)
    kwargs = dict(facecolor="none", s=200)

    omission_sel = np.ones(correction.size, dtype=np.bool)
    omission_sel[index] = False
    ommissions = grid.reshape(-1, 2)[np.logical_and(correction.ravel(), omission_sel)]

    ax.text(10, image.shape[0], f'Nombre de trouvés: {result["nombre_trouves"]}\n'
                                f'Nombre d\'omissions: {result["nb_omissions"]}\n'
                                f'Nombre d\'erreurs: {result["nombre_erreurs"]}\n')
    ax.scatter(*coord[~correct].T, edgecolors="r", label="Erreurs", **kwargs)
    ax.scatter(*coord[correct].T, edgecolors="g", label="Trouvés", **kwargs)
    ax.scatter(*ommissions.T, edgecolors="orange", label="Omissions", **kwargs)

    if debug:
        ax.scatter(*grid.T, alpha=.3, c=correction.ravel() + 0)

    ax.legend()

    return ax


def load_image(filename, return_info=False):
    res = Image.open(filename)
    im = ImageEnhance.Contrast(res).enhance(1.6)
    im = np.array(im)
    im = im[::2][:, ::2]  # To comment!

    # 4 points transform
    imbw = im.mean(-1)
    imbw = convolve2d(imbw, make_gaussian_kernel(im.shape[0] // 100), mode="same")
    normalize(imbw)

    # find contours in the thresholded image
    thresholded = (imbw > 0.5).astype(np.uint8)
    cnts = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = cnts[0][:, 0]

    # Find 4 points
    cost = np.linalg.norm((cnts / np.array(imbw.shape[::-1])[None, :])[None, :, :] -
                          np.reshape([0, 0, 1, 0, 1, 1, 0, 1], (4, 1, 2)), axis=-1)
    points = cnts[np.argmin(cost, axis=1)]

    page = four_point_transform(im, points)

    if return_info:
        return page, (im, points)
    return page
