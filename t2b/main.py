from itertools import product

import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from imutils.perspective import four_point_transform
from numba import njit, prange, uint32, guvectorize, float64
from scipy.signal import convolve2d

from t2b.constants import Nb_dots
from t2b.tools import charger_motifs
import numba as nb


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
        kernel = make_gaussian_kernel(kernel_size)
        dots = convolve2d(dots, kernel, mode="same")
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

    kernel_size = int(imbw.shape[0] * 93 / 3000)
    kernel = charger_motifs(["all"])[0]
    kernel = cv2.resize(kernel, (kernel_size, kernel_size))

    imm1 = convolve2d(imbw, kernel, mode="same")
    imc = normalize(convolve2d(imm1, make_gaussian_kernel(kernel_size), mode="same"))
    return imc


# @njit()
@guvectorize([(uint32, uint32, uint32, uint32, float64, float64[:, :], float64[:])], '(),(),(),(),(),(n,m)->()',
             target="parallel")
def likelihood(startx, starty, sizex, sizey, rot, image, cost):
    x, y = [np.arange(i) * s for i, s in zip(Nb_dots, [sizex, sizey])]
    nx, ny = Nb_dots

    # Create coordinates
    _coord = np.zeros((nx, ny, 2))
    _coord[:, :, 0] = x.reshape((nx, 1))
    _coord[:, :, 1] = y.reshape((1, ny))
    # coord = np.array([x.repeat(ny).reshape(nx, ny), y.repeat(nx).reshape(ny, nx).T]).reshape(2, -1).T
    # coord = np.array(np.meshgrid(x, y)).reshape((2, -1)).T
    coord = _coord.reshape(nx * ny, 2)

    # Create R
    R = np.zeros((2, 2))
    R[0, 0] = R[1, 1] = np.cos(rot)
    R[0, 1] = np.sin(rot)
    R[1, 0] = -np.sin(rot)
    # R = np.array([np.cos(-rot), np.sin(-rot)])[np.array([0, 1, 1, 0])]
    # R = (R * np.array([1, -1, 1, 1])).reshape(2, 2)

    coord_rot = R.dot(coord.T).T
    coord_rot[:, 0] += startx
    coord_rot[:, 1] += starty
    index = coord_rot.astype(np.uint)

    # Ravel index
    shape = image.shape
    indexr = (index[:, 1] * shape[1] + index[:, 0]).astype(np.uint)

    imager = image.ravel()
    selection = imager[indexr]
    cost[0] = np.linalg.norm(selection)
    # plt.imshow(image)
    # plt.scatter(*index.T)
    # return cost


@njit()
def unravel_index(i, shape, out):
    count = nb.uint64(i)
    N = len(shape)
    for j in range(N):
        out[(N - 1) - j] = count % shape[(N - 1) - j]
        count -= out[(N - 1) - j]
        count /= shape[(N - 1) - j]


# @njit(parallel={'comprehension': False,  # parallel comprehension
#                 'prange': True,  # parallel for-loop
#                 'numpy': False,  # parallel numpy calls
#                 'reduction': False,  # parallel reduce calls
#                 'setitem': False,  # parallel setitem
#                 'stencil': False,  # parallel stencils
#                 'fusion': False,  # enable fusion or not
#                 })
# def likelihood(start, size, angle, image):
#     nb = np
#     total = (start.shape[0] ** 2) * (size.shape[0] ** 2) * angle.shape[0]
#     shape = start.size, start.size, size.size, size.size, angle.size
#
#     res = list(np.zeros(total))
#     for i in prange(total):
#         index = np.zeros(5, dtype=nb.int64)
#         unravel_index(i, shape, index)
#         args = np.zeros(5)
#         args[0], args[1] = start[index[0]], start[index[1]]
#         args[2], args[3] = size[index[2]], size[index[3]]
#         args[4] = angle[index[4]]
#         res[i] = _likelihood(args[:2], args[2:4], args[4], image)
#         # j = 0
#         #     count -= index[-j - 1]
#         #     count /= shape[-j - 1]
#         # pass
#     return res  # .reshape(shape)


def find_grid_coordinates2(image):
    imc = match_filter_image(image).astype(np.float64)

    angle = np.deg2rad(np.arange(-2, 2, 0.1)).astype(np.float64)
    start = (np.arange(-20, 21) + np.mean([53, 67])).astype(np.uint32)
    size = np.arange(35, 45).astype(np.uint32)

    cost = likelihood(start[:, None, None, None, None],
                      start[None, :, None, None, None],
                      size[None, None, :, None, None],
                      size[None, None, None, :, None],
                      angle[None, None, None, None, :],
                      imc)
    cost = _likelihood(np.array([53, 67]), [18, 18], np.deg2rad(-1), imc)
    pass


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
    im = Image.open(filename)
    # im = ImageEnhance.Contrast(im).enhance(1.6)
    im = np.array(im)
    if im.shape[0] > im.shape[1]:
        im = np.moveaxis(im, 1, 0)
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

    page = cv2.resize(page, (600, 846)[::-1])

    if return_info:
        return page, (im, points)
    return page
