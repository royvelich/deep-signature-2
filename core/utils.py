# standard library
import sys
from pathlib import Path
from tqdm.auto import tqdm
import yaml
import argparse

# numpy
import numpy as np

# joblib
from joblib import Parallel


def save_command_arguments(path: Path, args: argparse.Namespace):
    args_dict = vars(args)

    # Save the dictionary to a file
    with open(str(path), 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)


def standard_faces_to_pyvista_faces(standard_f: np.ndarray) -> np.ndarray:
    """
    Converts from standard face representation to PyVista representation.

    Args:
    faces: A 2D NumPy array. Each row represents a face and contains the indices of its vertices.

    Returns:
    A 1D NumPy array representing the faces in PyVista format.
    """
    num_faces = standard_f.shape[0]
    sizes = np.full((num_faces, 1), standard_f.shape[1])
    f_pyvista = np.hstack((sizes, standard_f)).flatten()
    return f_pyvista


def pyvista_faces_to_standard_faces(pyvista_f: np.ndarray) -> np.ndarray:
    """
    Converts from PyVista face representation to standard representation.
    Args:
    faces: A 1D NumPy array representing the faces in PyVista format.
    Returns:
    A list of lists. Each list represents a face and contains the indices of its vertices.
    """
    face_counts = pyvista_f[::pyvista_f[0]+1]
    face_indices = np.split(pyvista_f, np.cumsum(face_counts + 1)[:-1])
    return np.array([indices[1:].tolist() for indices in face_indices])


# https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
