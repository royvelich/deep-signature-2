# standard library
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum

# igl
import igl

# scipy
from scipy.spatial import cKDTree

# pyvista
import pyvista as pv

# numpy
import numpy as np

# sklearn
from sklearn.neighbors import KDTree

# torch-geometric
from torch_geometric.nn import fps

# torch
import torch

# surface-diff-inv
from core.utils import pyvista_faces_to_standard_faces, standard_faces_to_pyvista_faces


class PrincipalDirection(Enum):
    D1 = 1
    D2 = 2


class BoundaryType(Enum):
    UPPER = 1
    LOWER = 2
    INTERIOR = 3
    UNKNOWN = 4


class Mesh:
    def __init__(self, v: np.ndarray, f: np.ndarray):
        self._v = v
        self._tree = cKDTree(v)
        self._f = f
        self._pyvista_f = standard_faces_to_pyvista_faces(standard_f=f)
        self._d1, self._d2, self._k1, self._k2 = igl.principal_curvature(v=v, f=f)

    @property
    def k1(self) -> np.ndarray:
        return self._k1

    @property
    def k2(self) -> np.ndarray:
        return self._k2

    @property
    def d1(self) -> np.ndarray:
        return self._d1

    @property
    def d2(self) -> np.ndarray:
        return self._d2

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def f(self) -> np.ndarray:
        return self._f

    @property
    def pyvista_f(self) -> np.ndarray:
        return self._pyvista_f

    def calculate_codazzi_arguments(self) -> np.ndarray:
        dk1_1 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k1)
        dk1_2 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k1)
        dk1_22 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=dk1_2)
        dk2_1 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k2)
        dk2_2 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k2)
        dk2_11 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=dk2_1)
        return np.stack((self._k1, self._k2, dk1_1, dk1_2, dk1_22, dk2_1, dk2_2, dk2_11), axis=0)

    def _directional_derivative_at_point(self, point: np.ndarray, direction: np.ndarray, scalar_field: np.ndarray, h: float = 1e-5, max_attempts: int = 10) -> np.ndarray:
        _, idx = self._tree.query(point)
        attempt = 0
        while True:
            point_plus_h = point + h * direction
            point_minus_h = point - h * direction
            _, idx_plus = self._tree.query(point_plus_h)
            _, idx_minus = self._tree.query(point_minus_h)

            if (idx != idx_plus and idx != idx_minus) or attempt == max_attempts:
                break

            h *= 10
            attempt += 1

        return (scalar_field[idx_plus] - scalar_field[idx_minus]) / (2 * h)

    def _directional_derivative(self, direction_field: np.ndarray, scalar_field: np.ndarray, h: float = 1e-5) -> np.ndarray:
        return np.array([self._directional_derivative_at_point(point=self._v[i], direction=direction_field[i], scalar_field=scalar_field, h=h) for i in range(self._v.shape[0])])

    def _principal_direction_derivative(self, principal_direction: PrincipalDirection, scalar_field: np.ndarray, h: float = 1e-5) -> np.ndarray:
        if principal_direction == PrincipalDirection.D1:
            return self._directional_derivative(direction_field=self._d1, scalar_field=scalar_field, h=h)
        else:
            return self._directional_derivative(direction_field=self._d2, scalar_field=scalar_field, h=h)

    def plot(self):
        Mesh.plot_meshes([self])

    def downsample(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = fps(x=v, ratio=ratio)
        v = v[indices].cpu().detach().numpy()
        return Mesh.from_vertices(v=v)

    @staticmethod
    def from_file(file_path: Path) -> Mesh:
        v, f = igl.read_triangle_mesh(filename=str(file_path), dtypef=np.float64)
        return Mesh(v=v, f=f)

    @staticmethod
    def from_vertices(v: np.ndarray) -> Mesh:
        poly_data = pv.PolyData(v)
        surf = poly_data.delaunay_2d()
        v = surf.points
        pyvista_f = surf.faces
        standard_f = pyvista_faces_to_standard_faces(pyvista_f=pyvista_f)
        return Mesh(v=v, f=standard_f)

    @staticmethod
    def plot_meshes(meshes: List[Mesh], show_edges: bool = False, show_principal_directions: bool = True, window_size: Tuple[int, int] = (1000, 1000), shape: Optional[Tuple[int, int]] = None):
        mesh_count = len(meshes)
        plotter = pv.Plotter(shape=shape if shape is not None else (1, mesh_count), window_size=window_size)
        for index, mesh in enumerate(meshes):
            pyvista_mesh = pv.PolyData(mesh.v, mesh.pyvista_f)

            if shape is None:
                plotter.subplot(0, index)
            else:
                plotter.subplot(index // shape[0], index % shape[0])

            plotter.add_mesh(mesh=pyvista_mesh, show_edges=show_edges)

            if show_principal_directions is True:
                pyvista_mesh['d1'] = mesh.d1
                pyvista_mesh['d2'] = mesh.d2
                glyphs1 = pyvista_mesh.glyph(orient='d1', scale=False, factor=0.1, tolerance=0.01)
                glyphs2 = pyvista_mesh.glyph(orient='d2', scale=False, factor=0.1, tolerance=0.01)
                plotter.add_mesh(glyphs1, color='red', opacity=1)
                plotter.add_mesh(glyphs2, color='blue', opacity=1)

        plotter.show()

    def center(self) -> Mesh:
        t = np.mean(self._v, 0, keepdims=True)
        v = self._v - t
        return Mesh(v=v, f=self._f)
