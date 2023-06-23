# standard library
import argparse

# deep-signature-2
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator, QuadraticMonagePatchGenerator
from data.evaluation import CorrelationEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches-count', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--grid-size', type=int)
    parser.add_argument('--points-count', type=int)
    args = parser.parse_args()

    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=200, sigma=2)
    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=400, sigma=0.5)
    # patch_generator = SimplexNoisePatchGenerator(limit=3, grid_size=300, scale=0.5)
    # patch_generator = RBFPatchGenerator(limit=2, grid_size=200, points_count=300)
    # patch_generator = QuadraticMonagePatchGenerator(limit=1, grid_size=20)
    # patch_generator = RBFPatchGenerator(limit=10, grid_size=300, points_count=500)
    # patch_generator = RBFPatchGenerator(limit=2, grid_size=400, points_count=400)
    patch_generator = RBFPatchGenerator(limit=1, grid_size=args.grid_size, points_count=args.points_count)
    correlation_evaluator = CorrelationEvaluator(patches_count=args.patches_count, points_ratio=1, num_workers=args.num_workers, patch_generator=patch_generator)
    correlation_evaluator.evaluate()
