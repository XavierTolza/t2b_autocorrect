import lzma
from argparse import ArgumentParser
from base64 import b64encode, b64decode

from t2b.constants import current_version, is_correct
from t2b.main import *
from t2b.tools import image_from_bytes


def make_packet(img_path, test_id, id=None, token=None):
    return dict(img=b64encode(open(img_path, "rb").read()).decode("utf-8"), token=token, id=id, version=current_version,
                test_id=test_id, compressed=False)


def lambda_handler(event, context=None):
    res = dict(success=False, msg="")
    try:
        id, version, token, test_id, img, compressed = [event[i]
                                                        for i in "id,version,token,test_id,img,compressed".split(",")]
        img = b64decode(img)
        if compressed:
            img = lzma.decompress(img)
        img = image_from_bytes(img)
        img = np.array(img)

        im = load_image(img)
        grid = find_grid_coordinates2(im)
        coord_selector = marked_selector(grid, im)
        coord_selector = np.arange(coord_selector.size)[coord_selector]

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
        # raise
        res["msg"] = str(e)
        res["error_type"] = e.__class__.__name__
        return res
    res["success"] = True
    return res


if __name__ == '__main__':
    # Simple runtime test
    parser = ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("test_id",type=int)

    packet = make_packet(**parser.parse_args().__dict__)
    print(lambda_handler(packet))
