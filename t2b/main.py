from itertools import product

import cv2

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np

from t2b.likelihood import likelihood, gen_all_indexes
from t2b.line_likelihood import line_likelihood, line_find_coordinates
from t2b.constants import Nb_dots
from t2b.tools import charger_motifs, rot_matrix


def make_gaussian_kernel(kernel_size, sigma=5):
    x = np.linspace(-1, 1, kernel_size)
    x = np.array(np.broadcast_arrays(x[:, None], x[None, :]))
    kernel = np.exp(-0.5 * sigma * np.linalg.norm(x - np.array([0, 0])[:, None, None], axis=0))
    kernel /= kernel.max()
    return kernel


def normalize(x):
    x -= x.min()
    x /= x.max()
    return x


def gradient_norm(im):
    return np.linalg.norm(np.gradient(im), axis=0)


def convolve2d(im, kernel, mode="same"):
    return cv2.filter2D(im, -1, kernel)


def img_to_bw(img):
    imbw = -img.max(-1).astype(np.float32)
    return imbw


def guess_kernel_size(imbw):
    S = np.fft.fft2(imbw)
    S = np.log(np.abs(S))
    start = np.array([15, 5])
    _S = S[start[0]:75, start[1]:55]
    argmax = np.array(np.unravel_index(np.argmax(_S.ravel()), _S.shape))
    argmax += start
    res = np.mean(np.array(imbw.shape) / argmax)
    return res


def draw_contours(img, cnts, color=(0, 255, 0)):
    return cv2.drawContours(img, cnts, -1, color, 3)


def contour_to_rect(cnt):
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx


def match_filter_image(imbw, kernel_size=None, blur_size=None):
    if kernel_size is None:
        kernel_size_float = guess_kernel_size(imbw)
    else:
        kernel_size_float = kernel_size
    kernel_size = int(kernel_size_float)

    if blur_size is None:
        blur_size = kernel_size_float
    blur_size = int(blur_size)

    imbw = gradient_norm(imbw)

    kernel = charger_motifs(["all"])[0]
    kernel = cv2.resize(kernel, (kernel_size, kernel_size)) / 255
    kernel = gradient_norm(kernel)
    kernel = kernel / kernel.sum()

    imm1 = convolve2d(imbw, kernel, mode="same")
    imc = cv2.blur(imm1, (blur_size,) * 2)
    return imc


def guess_trigger(imc):
    margin = 0.15
    shapex, shapey = shape = np.array(imc.shape[:2])
    marginx, marginy = (shape * margin).astype(np.int)
    sample = imc[marginx:shapex - marginx, marginy:shapey - marginy]
    return sample.mean() - 2 * sample.std()


def contour_from_lines(lines):
    a1, r1 = lines
    a2, r2 = [np.roll(i, 1) for i in lines]
    (s1, c1), (s2, c2) = [[m(i) for m in [np.sin, np.cos]] for i in [a1, a2]]

    den = (c2 * s1 - c1 * s2)
    return np.transpose([
        (r2 * s1 - r1 * s2) / den,
        (c2 * r1 - c1 * r2) / den,
    ])


def search_lines(imc, pos, angles):
    angles = np.deg2rad(np.arange(-2, 2, 0.1))

    line_positions = np.array([65, 56, 808, 540])
    line_offset = np.arange(-40, 40, 3)[None]
    radius = line_positions[:, None] + line_offset
    angles_delta = np.deg2rad([0, 90, 0, 90])

    data = np.array(np.broadcast_arrays(angles[:, None, None] + angles_delta[None, :, None], radius[None]))

    cost = line_likelihood(*np.reshape(data, (2, -1)), imc).reshape(data.shape[1:])
    return cost


def find_grid_coordinates(image):
    imc = match_filter_image(image, blur_size=20)
    imc = normalize(gradient_norm(imc))

    angles = np.deg2rad(np.arange(-2, 2, 0.1))
    line_positions = np.array([65, 56, 808, 540])
    line_offset = np.arange(-40, 40, 3)[None]
    radius = line_positions[:, None] + line_offset
    angles_delta = np.deg2rad([0, 90, 0, 90])

    data = np.array(np.broadcast_arrays(angles[:, None, None] + angles_delta[None, :, None], radius[None]))

    cost = line_likelihood(*np.reshape(data, (2, -1)), imc).reshape(data.shape[1:])
    argmax = np.argmax(np.moveaxis(cost, 1, 0).reshape(4, -1), axis=1)
    argmax = np.unravel_index(argmax, cost[:, 0].shape)
    found_angle = angles[argmax[0]] + angles_delta
    found_radius = line_offset[0][argmax[1]] + line_positions

    points = contour_from_lines(np.roll([found_angle, found_radius], -1, axis=1))

    x, y = [np.linspace(0.5 / i, 1 - (0.5 / i), i) for i in Nb_dots]
    grid = x[:, None, None] * points[[0, -1]][None] + (1 - x)[:, None, None] * points[[1, 2]][None]
    grid = y[None, :, None] * grid[:, 0, None] + (1 - y)[None, :, None] * grid[:, 1, None]
    return grid


def marked_selector(grid, image, trigger=0.2):
    imc = normalize(cv2.blur(gradient_norm(image.std(-1)), (10,) * 2))
    sel = imc[grid[:, 0], grid[:, 1]]
    return sel > trigger


def filter_marked(grid, image, trigger=0.2):
    return grid[marked_selector(grid, image, trigger)]


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


def four_point_transform(im, points, dst_shape=None):
    if dst_shape is None:
        dst_shape = im.shape[:2][::-1]
    maxWidth, maxHeight = dst_shape
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(points.astype("float32"), dst)
    warped = cv2.warpPerspective(im, M, (maxWidth, maxHeight))
    return warped


def find_contours(thresholded, min_area=0, max_area=1):
    thresholded = thresholded.astype(np.uint8) * 255
    cnts = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    area = np.array([cv2.contourArea(i) for i in cnts]) / np.prod(thresholded.shape)
    sel = np.logical_and(area > min_area, area < max_area)
    if (~sel).any():
        cnts = [i for i, s in zip(cnts, sel) if s]
        area = area[sel]
    return cnts, area


def load_image(filename, return_info=False, page_extract=True, resize=True, page_threshold=0.4):
    if type(filename) == str:
        im = cv2.imread(filename)
        # im = ImageEnhance.Contrast(im).enhance(1.6)
    else:
        im = filename
    im = np.array(im)
    if im.shape[0] < im.shape[1]:
        im = np.moveaxis(im, 1, 0)[:, ::-1]

    if page_extract:
        # 4 points transform
        imbw = im.mean(-1)
        imc = cv2.blur(imbw, (im.shape[1] // 100,) * 2)

        # find contours in the thresholded image
        imc = normalize(imc)
        thresholded = (imc > page_threshold).astype(np.uint8)
        cnts = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        area = np.array([cv2.contourArea(i) for i in cnts])
        cnts = cnts[np.argmax(area)][:, 0]

        # Find 4 points
        cost = np.linalg.norm((cnts / np.array(imbw.shape[::-1])[None, :])[None, :, :] -
                              np.reshape([0, 0, 1, 0, 1, 1, 0, 1], (4, 1, 2)), axis=-1)
        points = cnts[np.argmin(cost, axis=1)]

        page = four_point_transform(im, points)
    else:
        page = im

    if resize:
        page = cv2.resize(page, (846, 600)[::-1])

    if return_info:
        return page, (im, points)
    return page
