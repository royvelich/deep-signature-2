# standard library
import argparse

# surface-diff-inv
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator, QuadraticMonagePatchGenerator, QuadraticMonagePatchGenerator2
from core.geometry import ScalarField

# pyvista
import pyvista as pv

# surface-diff-inv
from core.geometry import Mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches-count', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--limit', type=float)
    parser.add_argument('--coeff-limit', type=float)
    parser.add_argument('--grid-size', type=int)
    parser.add_argument('--points-count', type=int)
    parser.add_argument('--min-sigma', type=float)
    parser.add_argument('--max-sigma', type=float)
    parser.add_argument('--max-abs-z', type=float)
    args = parser.parse_args()

    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=200, sigma=2)
    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=400, sigma=0.5)
    # patch_generator = SimplexNoisePatchGenerator(limit=3, grid_size=300, scale=0.5)
    # patch_generator = RBFPatchGenerator(limit=2, grid_size=200, points_count=300)
    # patch_generator = QuadraticMonagePatchGenerator(limit=1, grid_size=20)
    # patch_generator = RBFPatchGenerator(limit=10, grid_size=300, points_count=500)
    # patch_generator = RBFPatchGenerator(limit=2, grid_size=400, points_count=400)
    # patch_generator = RBFPatchGenerator(limit=args.limit, grid_size=args.grid_size, points_count=args.points_count)

    patch_generator = QuadraticMonagePatchGenerator(limit=args.limit, grid_size=args.grid_size, coeff_limit=args.coeff_limit)
    patch = patch_generator.generate()
    Mesh.plot_meshes(meshes=[patch], show_principal_directions=False, scalar_field=ScalarField.K1)
