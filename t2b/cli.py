#!python

import click

from t2b.constants import correction
from t2b.main import load_image, find_image_coordinates, find_grid_coordinates, evaluate, plot_result


@click.command()
@click.argument("input_filename",
                type=click.Path(True, True, False, readable=True, resolve_path=True))
@click.option("-o", "--output_filename", help="Nom du fichier image de sortie", default=None)
def cli(input_filename, output_filename):
    if output_filename is None:
        name, ext = input_filename.split(".")[0], ".".join(input_filename.split(".")[1:])
        output_filename = f"{name}_corrected.{ext}"

    image = load_image(input_filename)
    coord = find_image_coordinates(image)
    grid = find_grid_coordinates(coord)
    result = evaluate(grid, coord, correction[0])

    ax = plot_result(image, grid, coord, result, debug=False)
    ax.figure.savefig(output_filename)


if __name__ == '__main__':
    cli()
