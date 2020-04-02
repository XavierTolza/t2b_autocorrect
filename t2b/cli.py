#!python

import click
import cv2
import numpy as np

from t2b.constants import corrections
from t2b.main import load_image, find_image_coordinates, find_grid_coordinates, evaluate, plot_result


@click.command()
@click.argument("input_filename",
                type=click.Path(True, True, False, readable=True, resolve_path=True))
@click.option("-o", "--output_filename", help="Nom du fichier image de sortie", default=None)
@click.option('--debug/--no-debug', default=False)
def cli(input_filename, output_filename, debug=False):
    if output_filename is None:
        name, ext = input_filename.split(".")[0], ".".join(input_filename.split(".")[1:])
        output_filename = f"{name}_corrected.{ext}"

    image = load_image(input_filename)
    coord, dots = find_image_coordinates(image, debug=True)
    grid = find_grid_coordinates(coord)
    result = evaluate(grid, coord, corrections[0] == 6)

    ax = plot_result(image, grid, coord, result, debug=False)
    ax.figure.savefig(output_filename)

    if debug:
        cv2.imwrite(output_filename.replace("corrected", "dots_mask"), image * dots[:, :, None])
        cv2.imwrite(output_filename.replace("corrected", "dots"), (dots * 255).astype(np.uint8))


if __name__ == '__main__':
    cli()
