# standard library
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# numpy
import numpy as np

# pandas
import pandas as pd

# joblib
from joblib import delayed

# surface-diff-inv
from core import utils as core_utils
from data.generation import PatchGenerator


class CorrelationEvaluator:
    def __init__(self, patches_count: int, num_workers: int, patch_generator: PatchGenerator):
        self._patches_count = patches_count
        self._num_workers = num_workers
        self._patch_generator = patch_generator

    def _calculate_codazzi_arguments(self) -> np.ndarray:
        patch = self._patch_generator.generate()
        codazzi_arguments = patch.calculate_codazzi_arguments()
        return codazzi_arguments

    def evaluate(self) -> np.ndarray:
        codazzi_arguments_list = core_utils.ProgressParallel(n_jobs=self._num_workers, total=self._patches_count)(delayed(self._calculate_codazzi_arguments)() for _ in range(self._patches_count))
        codazzi_arguments = np.stack(codazzi_arguments_list).T
        corrcoef1 = np.corrcoef(codazzi_arguments)
        df = pd.DataFrame(corrcoef1)
        pd.set_option('display.precision', 5)
        print(df)
