from os.path import abspath, dirname, join

import numpy as np

Nb_dots = np.array([40, 25])  # nombre de points

with open(join(dirname(abspath(__file__)), "correction.txt"), "r") as fp:
    corrections = fp.read().replace("\n", "")
corrections = np.array(list(corrections)).astype(np.uint8).reshape((-1,) + tuple(Nb_dots))
corrections = corrections[:, :, ::-1]
corrections[1] = corrections[1, :, ::-1]
corrections = np.array([-1, 0, 1, 2, 3, -1, 4, 5, 6, 7])[corrections]

is_correct = np.array([
    corrections[0] == 6,
    np.logical_or(corrections[1] == 2, corrections[1] == 3)
])
