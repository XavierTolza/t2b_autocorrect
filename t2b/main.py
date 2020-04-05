from itertools import product

import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imutils.perspective import four_point_transform
from scipy.signal import convolve2d

from t2b.c_funs import likelihood, gen_all_indexes
from t2b.constants import Nb_dots
from t2b.tools import charger_motifs, rot_matrix


# from t2b.c_funs import likelihood


def make_gaussian_kernel(kernel_size, sigma=5):
    x = np.linspace(-1, 1, kernel_size)
    x = np.array(np.broadcast_arrays(x[:, None], x[None, :]))
    kernel = np.exp(-0.5 * sigma * np.linalg.norm(x - np.array([0, 0])[:, None, None], axis=0))
    kernel /= kernel.max()
    return kernel


def find_image_coordinates(image, trigger=0.5, debug=False):
    shape = image.shape
    dots = image.std(-1)
    kernel_size = int(np.min(shape[:-1]) / 100)
    if kernel_size > 1:
        dots = cv2.blur(dots, (kernel_size,) * 2)
    dots = normalize(dots)

    # trigger = np.sort(dots.ravel())[int(dots.size * 0.98)]
    dots_blur = normalize(convolve2d(dots, make_gaussian_kernel(int(image.shape[0] * 30 / 1342)), mode="same"))
    dots_trigger = dots_blur > trigger

    cnt = cv2.findContours(dots_trigger.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    coord = np.round(np.array([np.median(i[:, 0, :], 0) for i in cnt])).astype(np.uint)
    coordr = coord / np.array(shape[:2])[None, ::-1]

    # On filtre les points qui sont trop près des bords
    margin = 0.01
    selector = np.logical_and(coordr > margin, coordr < (1 - margin)).all(1)
    coord = coord[:, ::-1]

    if debug:
        return coord[selector], dots
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


def gradient_norm(im):
    return np.linalg.norm(np.gradient(im), axis=0)


def match_filter_image(image, kernel_size=15):
    imbw = -image.max(-1).astype(np.float32)
    imbw = normalize(gradient_norm(imbw))

    kernel = charger_motifs(["all"])[0]
    kernel = cv2.resize(kernel, (kernel_size, kernel_size))

    imm1 = convolve2d(imbw, kernel, mode="same")
    imc = normalize(cv2.blur(imm1, (kernel_size,) * 2))
    return imc


def gen_all_indexes(offset, scale, angle):
    x, y = [np.arange(i) * s for i, s in zip(Nb_dots, scale)]
    xy = np.array(list(product(x, y)))
    R = rot_matrix(angle)
    xy = R.dot(xy.T).T
    xy += np.array(offset)[None]
    xy = xy.astype(np.uint)
    return xy


def find_grid_coordinates2(image):
    imc = match_filter_image(image).astype(np.float64)
    cv2.imwrite("/tmp/out.jpg", (imc * 255).astype(np.uint8))

    angle = np.deg2rad(np.arange(-1, 1, 0.1)).astype(np.float64)

    shape = np.array(imc.shape[:2])
    rect_size = np.array([
        np.linspace(700 / 846, 800 / 846, 10),
        np.linspace(450 / 600, 500 / 600, 10),
    ]) * shape[:, None]
    offset = (np.array([
        np.linspace(30 / 846, 90 / 846, 15),
        np.linspace(40 / 600, 100 / 600, 15),
    ]) * shape[:, None]).astype(np.uint32)

    size = (rect_size / Nb_dots[:, None]).astype(np.float64)
    shape = (offset.shape[1], offset.shape[1], size.shape[1], size.shape[1], angle.size)
    cost = np.zeros(shape)

    likelihood(*offset, *size, angle, imc, cost.ravel())

    # Find max
    argmax = np.unravel_index(cost.ravel().argmax(), cost.shape)
    values = np.array([i[j] for i, j in zip([offset[0], offset[1], size[0], size[1], angle], argmax)])

    res = gen_all_indexes(values[:2], values[2:4], values[4])
    return res


def filter_marked(grid, image, trigger=0.2):
    imc = normalize(cv2.blur(gradient_norm(image.std(-1)), (10,) * 2))
    sel = imc[grid[:, 0], grid[:, 1]]
    return grid[sel > trigger]


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
        fig, ax = plt.subplots(1, 1, figsize=(30 // 2, 21 // 2))
    correct, correction, index = result["is_correct"], result["correction"], result["index"]
    ax.imshow(np.moveaxis(image, 1, 0), origin="lower")
    kwargs = dict(facecolor="none", s=200)

    omission_sel = np.ones(correction.size, dtype=np.bool)
    omission_sel[index] = False
    ommissions = grid.reshape(-1, 2)[np.logical_and(correction.ravel(), omission_sel)]

    ax.text(10, 10, f'Nombre de trouvés: {result["nombre_trouves"]}\n'
                    f'Nombre d\'omissions: {result["nb_omissions"]}\n'
                    f'Nombre d\'erreurs: {result["nombre_erreurs"]}\n')
    ax.scatter(*coord[~correct].T, edgecolors="r", label="Erreurs", **kwargs)
    ax.scatter(*coord[correct].T, edgecolors="g", label="Trouvés", **kwargs)
    ax.scatter(*ommissions.T, edgecolors="orange", label="Omissions", **kwargs)
    plt.tight_layout(0, 0, 0)

    if debug:
        ax.scatter(*grid.T, alpha=.3, c=correction.ravel() + 0)

    ax.legend()

    return ax


def load_image(filename, return_info=False):
    im = Image.open(filename)
    # im = ImageEnhance.Contrast(im).enhance(1.6)
    im = np.array(im)
    if im.shape[0] < im.shape[1]:
        im = np.moveaxis(im, 1, 0)

    # 4 points transform
    imbw = im.mean(-1)
    imbw = cv2.blur(imbw, (im.shape[1] // 100,) * 2)
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

    page = cv2.resize(page, (846, 600)[::-1])

    if return_info:
        return page, (im, points)
    return page
