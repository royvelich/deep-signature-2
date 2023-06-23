# surface-diff-inv
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator, QuadraticMonagePatchGenerator

# pyvista
import pyvista as pv

# surface-diff-inv
from core.geometry import Mesh


if __name__ == "__main__":
    # patch_generator = GaussianPatchGenerator(limit=20, grid_size=800, sigma=1)
    # patch_generator = InverseFourierPatchGenerator(limit=10, grid_size=10, scale=4)
    patch_generator = RBFPatchGenerator(limit=1, grid_size=500, points_count=600)
    # patch_generator = QuadraticMonagePatchGenerator(limit=1, grid_size=20)
    # patch_generator = SimplexNoisePatchGenerator(limit=5, grid_size=200, scale=1)
    patch = patch_generator.generate()
    # patch = patch.downsample(ratio=0.2)
    Mesh.plot_meshes(meshes=[patch])
