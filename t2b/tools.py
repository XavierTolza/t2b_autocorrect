import tarfile

import cv2
import numpy as np
from t2b.constants import Nb_dots as N

from PIL import Image


def generate_test_image(correction):
    symbols = []
    with tarfile.open("motifs.tar.xz") as tar:
        for i in "12346789":
            im = Image.open(tar.extractfile(tar.getmember(f"{i}.png")))
            symbols.append(np.array(im))
    _symbols = np.array(symbols)[:, :, :, -1]
    newshape = int(_symbols.shape[1] * 1.05012823501982)
    symbols = np.zeros((8, _symbols.shape[1], newshape))
    symbols[:, :,
    (newshape - _symbols.shape[1]) // 2:(newshape - _symbols.shape[1]) // 2 + _symbols.shape[2]] = _symbols

    result = np.moveaxis(symbols[correction[:, ::-1]], 1, 2)
    result = np.reshape(result, (np.prod(result.shape[:2]), np.prod(result.shape[2:])))
    result = 255 - result

    cv2.imwrite("/tmp/out.png", result)


if __name__ == '__main__':
    from t2b.constants import corrections

    generate_test_image(corrections[0])
