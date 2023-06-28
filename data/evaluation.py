# standard library
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# numpy
import numpy as np

# pandas
import pandas as pd

# joblib
from joblib import delayed

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
        # codazzi_arguments = np.stack(codazzi_arguments_list).T
        codazzi_arguments = np.hstack(codazzi_arguments_list)
        corrcoef1 = np.corrcoef(codazzi_arguments)
        df = pd.DataFrame(corrcoef1)
        pd.set_option('display.precision', 5)
        print(df)
        df.to_excel("./output2.xlsx")


        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # get data
        x = codazzi_arguments[0, :]
        y = codazzi_arguments[1, :]

        bins = 200

        # Plot histogram for first row
        axs[0].hist(x, bins=bins, color='blue', alpha=0.7)
        axs[0].set_title('Histogram of first row')

        # Plot histogram for second row
        axs[1].hist(y, bins=bins, color='green', alpha=0.7)
        axs[1].set_title('Histogram of second row')

        # Quadrant color mapping
        colors = ['red', 'green', 'blue', 'yellow']

        quadrant = np.zeros(x.shape, dtype=int)
        quadrant[(x < 0) & (y >= 0)] = 1
        quadrant[(x < 0) & (y < 0)] = 2
        quadrant[(x >= 0) & (y < 0)] = 3

        # Plot scatter plot to show correlation
        axs[2].scatter(x, y, c=quadrant, alpha=0.7, cmap=ListedColormap(colors))

        axs[2].set_title('Scatter plot of first row vs. second row')

        # Move x and y axis to the center
        axs[2].spines['left'].set_position('center')
        axs[2].spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        axs[2].spines['right'].set_color('none')
        axs[2].spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        axs[2].xaxis.set_ticks_position('bottom')
        axs[2].yaxis.set_ticks_position('left')

        # Set x and y limits to include all data points and center the origin
        axis_limit = max(np.max(np.abs(x)), np.max(np.abs(y)))
        axs[2].set_xlim([-axis_limit, axis_limit])
        axs[2].set_ylim([-axis_limit, axis_limit])

        # Remove default labels
        axs[2].set_xlabel('')
        axs[2].set_ylabel('')

        # Add custom labels at desired positions
        axs[2].text(axis_limit + 0.1 * axis_limit, 0, 'First row', ha='right', va='center', rotation='vertical')
        axs[2].text(0, -axis_limit - 0.15 * axis_limit, 'Second row', ha='center', va='top')

        plt.tight_layout()
        plt.show()