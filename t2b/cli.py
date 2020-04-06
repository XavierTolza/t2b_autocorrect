#!python

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

from t2b.constants import corrections, is_correct
from t2b.main import load_image, find_image_coordinates, evaluate, plot_result, \
    find_grid_coordinates2, filter_marked


@click.command()
@click.argument("input_filename",
                type=click.Path(True, True, False, readable=True, resolve_path=True))
@click.option("-o", "--output_filename", help="Nom du fichier image de sortie", default=None)
@click.option("-t", "--test", help="Test à corriger", default=1, type=int)
@click.option('--debug/--no-debug', default=False)
def cli(input_filename, output_filename, test, debug=False):
    print("Chargement de l'image")
    image = load_image(input_filename)
    grid = find_grid_coordinates2(image)
    coord = filter_marked(grid, image)
    result = evaluate(grid, coord, is_correct[test-1])
    print("Les résultats trouvés sont les suivants:")
    print(result)

    ax = plot_result(image, grid, coord, result, debug=False)
    if output_filename is None:
        plt.show()
    else:
        ax.figure.savefig(output_filename)

    if debug and output_filename is not None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))
        ax.imshow(image)
        ax.scatter(*coord.T, facecolor="none", edgecolors="r", s=250)
        ax.scatter(*grid.T, marker="x", color="w", alpha=.3)
        plt.tight_layout()
        fig.savefig(output_filename.replace("corrected", "dots"))

        # cv2.imwrite(output_filename.replace("corrected", "dots_mask"), image * dots[:, :, None])

if __name__ == '__main__':
    cli()
