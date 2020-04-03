#!python

import click
import cv2
import numpy as np

from t2b.constants import corrections
from t2b.main import load_image, find_image_coordinates, find_grid_coordinates, evaluate, plot_result
import matplotlib.pyplot as plt


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
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        ax.imshow((image * ((dots[:, :, None] * 0.5) + 0.5)).astype(np.uint8))
        ax.scatter(*coord.T, facecolor="none", edgecolors="r", s=250)
        ax.scatter(*grid.T, marker="x", color="w",alpha=.3)
        plt.tight_layout()
        fig.savefig(output_filename.replace("corrected", "dots"))

        cv2.imwrite(output_filename.replace("corrected", "dots_mask"), image * dots[:, :, None])


if __name__ == '__main__':
    cli()
