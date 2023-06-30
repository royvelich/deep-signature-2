# standard library
import argparse
from datetime import datetime
from pathlib import Path

# deep-signature-2
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator, QuadraticMonagePatchGenerator, QuadraticMonagePatchGenerator2
from data.evaluation import CorrelationEvaluator


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

    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d-%H-%M")
    dir_path = Path(f'./output/{date_string}')
    dir_path.mkdir(parents=True, exist_ok=True)
    core_utils.save_command_arguments(path=dir_path / Path(f'./args.yml'), args=args)

    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=200, sigma=2)
    # patch_generator = GaussianPatchGenerator(limit=1, grid_size=400, sigma=0.5)
    # patch_generator = SimplexNoisePatchGenerator(limit=3, grid_size=300, scale=0.5)
    # patch_generator = RBFPatchGenerator(limit=2, grid_size=200, points_count=300)
    # patch_generator = QuadraticMonagePatchGenerator(limit=1, grid_size=20)
    # patch_generator = RBFPatchGenerator(limit=10, grid_size=300, points_count=500)
    # patch_generator = RBFPatchGenerator(limit=2, grid_size=400, points_count=400)
    # patch_generator = RBFPatchGenerator(limit=args.limit, grid_size=args.grid_size, points_count=args.points_count)
    # patch_generator = GaussianPatchGenerator(limit=args.limit, grid_size=args.grid_size, min_sigma=args.min_sigma, max_sigma=args.max_sigma, max_abs_z=args.max_abs_z)

    patch_generator = QuadraticMonagePatchGenerator(limit=args.limit, grid_size=args.grid_size, coeff_limit=args.coeff_limit)
    correlation_evaluator = CorrelationEvaluator(patches_count=args.patches_count, num_workers=args.num_workers, patch_generator=patch_generator, dir_path=dir_path)
    correlation_evaluator.evaluate()
