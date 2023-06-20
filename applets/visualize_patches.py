# surface-diff-inv
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator

# pyvista
import pyvista as pv


if __name__ == "__main__":
    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=20, sigma=2)
    # patch_generator = InverseFourierPatchGenerator(limit=10, grid_size=10, scale=4)
    patch_generator = RBFPatchGenerator(limit=2, grid_size=200, points_count=300)
    # patch_generator = SimplexNoisePatchGenerator(limit=3, grid_size=300, scale=0.5)
    patch = patch_generator.generate()
    # patch = patch.downsample(ratio=0.2)
    patch.plot()
