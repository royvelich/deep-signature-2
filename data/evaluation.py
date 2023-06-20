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
    def __init__(self, patches_count: int, points_ratio: float, num_workers: int, patch_generator: PatchGenerator):
        self._patches_count = patches_count
        self._points_ratio = points_ratio
        self._num_workers = num_workers
        self._patch_generator = patch_generator

    def _calculate_codazzi_arguments(self) -> np.ndarray:
        patch = self._patch_generator.generate()
        codazzi_arguments = patch.calculate_codazzi_arguments()
        return codazzi_arguments

    def evaluate(self) -> np.ndarray:
        codazzi_arguments_list = core_utils.ProgressParallel(n_jobs=self._num_workers, total=self._patches_count)(delayed(self._calculate_codazzi_arguments)() for _ in range(self._patches_count))
        codazzi_arguments = np.hstack(codazzi_arguments_list)


        columns_count = codazzi_arguments.shape[1]
        num_columns_to_remove = int(columns_count * self._points_ratio)

        # Generate a list of all column indices
        all_indices = np.arange(columns_count)

        # Randomly choose some indices to remove
        remove_indices = np.random.choice(all_indices, num_columns_to_remove, replace=False)

        # Get the indices of the columns to keep
        keep_indices = np.setdiff1d(all_indices, remove_indices)

        # Select only the columns to keep
        codazzi_arguments_filtered = codazzi_arguments[:, keep_indices]

        # codazzi_arguments = np.array(codazzi_arguments_list).reshape(8, -1)
        corrcoef1 = np.corrcoef(codazzi_arguments)
        # corrcoef2 = np.corrcoef(codazzi_arguments_filtered)
        # Converting to a DataFrame
        df = pd.DataFrame(corrcoef1)

        # Setting pandas print options
        pd.set_option('display.precision', 5)

        print(df)