import tarfile
from io import BytesIO
from os.path import abspath, dirname, join

import cv2
import numpy as np


def rot_matrix(theta):
    return np.reshape([np.cos(-theta), np.sin(-theta), -np.sin(-theta), np.cos(-theta)], (2, 2))


class TarError(Exception):
    pass


def image_from_bytes(data, flag=-1):
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, flags=flag)


def charger_motifs(noms):
    symbols = []
    filename = join(dirname(abspath(__file__)), "motifs.tar.xz")
    try:
        with tarfile.open(filename) as tar:
            for i in noms:
                im = tar.extractfile(tar.getmember(f"{i}.png")).read()
                im = image_from_bytes(im)
                symbols.append(np.array(im))
    except Exception as e:
        raise TarError(f"Echec d'ouverture du fichier contenant les motifs :{filename}\n{str(e)}")
    symbols = np.array(symbols)[:, :, :, -1]
    return symbols


def generate_test_image(correction):
    _symbols = charger_motifs("12346789")
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

    generate_test_image(corrections[1])
