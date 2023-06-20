# surface-diff-inv
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator
from data.evaluation import CorrelationEvaluator


if __name__ == "__main__":
    # patch_generator = SimplexNoisePatchGenerator(limit=3, grid_size=300, scale=0.5)
    patch_generator = RBFPatchGenerator(limit=2, grid_size=200, points_count=300)
    correlation_evaluator = CorrelationEvaluator(patches_count=20, points_ratio=1, num_workers=18, patch_generator=patch_generator)
    correlation_evaluator.evaluate()
