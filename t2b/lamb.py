import lzma
from base64 import b64encode, b64decode
from io import BytesIO

import numpy as np
from PIL import Image

from t2b.constants import current_version, is_correct
from t2b.main import *


def make_packet(img_path, test_id, id=None, token=None):
    return dict(img=b64encode(open(img_path, "rb").read()), token=token, id=id, version=current_version,
                test_id=test_id, compressed=False)


def lambda_handler(event, context=None):
    res = dict(success=False, msg="")
    try:
        id, version, token, test_id, img, compressed = [event[i]
                                                        for i in "id,version,token,test_id,img,compressed".split(",")]
        img = b64decode(img)
        if compressed:
            img = lzma.decompress(img)
        img = Image.open(BytesIO(img))
        img = np.array(img)

        im = load_image(img)
        grid = find_grid_coordinates2(im)
        coord_selector = marked_selector(grid, im)

        result = evaluate(grid, grid[coord_selector], is_correct[test_id - 1])
        res.update(dict(
            nombre_trouves=result["nombre_trouves"],
            nombre_erreurs=result["nombre_erreurs"],
            nombre_omissions=result["nb_omissions"],
            grille=grid.tolist(),
            is_marked=coord_selector.tolist(),
            msg="OK"
        ))
    except Exception as e:
        res["msg"] = str(e)
        res["error_type"] = e.__class__.__name__
        return res
    res["success"] = True
    return res
