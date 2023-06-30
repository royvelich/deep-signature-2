# standard library
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum

# trimesh
import trimesh

# pymesh
import pymesh

# igl
import igl
import numpy

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


class ScalarField(Enum):
    K1 = 1
    K2 = 2
    DK1_1 = 3
    DK1_2 = 4
    DK2_1 = 5
    DK2_2 = 6
    DK1_22 = 7
    DK2_11 = 8


class BoundaryType(Enum):
    UPPER = 1
    LOWER = 2
    INTERIOR = 3
    UNKNOWN = 4


class Mesh:
    def __init__(self, v: np.ndarray, f: np.ndarray):
        self._v = v
        self._f = f
        self._pyvista_f = standard_faces_to_pyvista_faces(standard_f=f)
        self._d1, self._d2, self._k1, self._k2 = igl.principal_curvature(v=v, f=f)

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def f(self) -> np.ndarray:
        return self._f

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
    def pyvista_f(self) -> np.ndarray:
        return self._pyvista_f

    def plot(
            self,
            plotter: pv.Plotter,
            scalars: Optional[np.ndarray] = None,
            show_edges: bool = True,
            show_principal_directions: bool = False):
        pyvista_mesh = pv.PolyData(self._v, self._pyvista_f)
        plotter.add_mesh(mesh=pyvista_mesh, scalars=scalars, show_edges=show_edges)

        if show_principal_directions is True:
            pyvista_mesh['d1'] = self._d1
            pyvista_mesh['d2'] = self._d2
            glyphs1 = pyvista_mesh.glyph(orient='d1', scale=False, factor=0.01, tolerance=0.05)
            glyphs2 = pyvista_mesh.glyph(orient='d2', scale=False, factor=0.01, tolerance=0.05)
            plotter.add_mesh(glyphs1, color='red', opacity=1)
            plotter.add_mesh(glyphs2, color='blue', opacity=1)

    @staticmethod
    def from_file(file_path: Path) -> Mesh:
        v, f = igl.read_triangle_mesh(filename=str(file_path), dtypef=np.float64)
        return Mesh(v=v, f=f)

    @staticmethod
    def plot_meshes(
            meshes: List[Mesh],
            window_size: Tuple[int, int] = (1000, 1000),
            shape: Optional[Tuple[int, int]] = None,
            **kwargs):
        mesh_count = len(meshes)
        plotter = pv.Plotter(shape=shape if shape is not None else (1, mesh_count), window_size=window_size)
        for index, mesh in enumerate(meshes):
            if shape is None:
                plotter.subplot(0, index)
            else:
                plotter.subplot(index // shape[0], index % shape[0])

            mesh.plot(plotter=plotter, **kwargs)
        plotter.add_axes()
        plotter.show_grid()
        plotter.view_xz()
        plotter.show()

    def center(self) -> Mesh:
        t = np.mean(self._v, 0, keepdims=True)
        v = self._v - t
        return Mesh(v=v, f=self._f)


class Patch(Mesh):
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray):
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._z_grid = z_grid

        x = x_grid.ravel()
        y = y_grid.ravel()
        z = z_grid.ravel()

        v = np.stack([x, y, np.zeros_like(a=z)], axis=1)
        poly_data = pv.PolyData(v)
        surf = poly_data.delaunay_2d()
        v = surf.points
        v[:, 2] = z
        pyvista_f = surf.faces
        standard_f = pyvista_faces_to_standard_faces(pyvista_f=pyvista_f)
        super().__init__(v=v, f=standard_f)
        self._d1_grid = self._d1.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])
        self._d2_grid = self._d2.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])
        self._k1_grid = self._k1.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])[:, :, 0]
        self._k2_grid = self._k2.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])[:, :, 0]
        self._v_grid = self._v.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])
        self._tree = cKDTree(self._v)
        self._dk1_1_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k1_grid)
        self._dk1_2_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k1_grid)
        self._dk1_22_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._dk1_2_grid)
        self._dk2_1_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k2_grid)
        self._dk2_2_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k2_grid)
        self._dk2_11_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._dk2_1_grid)

    def plot(self,
             plotter: pv.Plotter,
             show_edges: bool = False,
             show_principal_directions: bool = False,
             show_grid_points: bool = True,
             grid_step: Optional[int] = None,
             scalar_field: Optional[ScalarField] = None):

        if scalar_field is ScalarField.K1:
            scalars = self._k1_grid
        elif scalar_field is ScalarField.K2:
            scalars = self._k2_grid
        elif scalar_field is ScalarField.DK1_1:
            scalars = self._dk1_1_grid
        elif scalar_field is ScalarField.DK1_2:
            scalars = self._dk1_2_grid
        elif scalar_field is ScalarField.DK2_1:
            scalars = self._dk2_1_grid
        elif scalar_field is ScalarField.DK2_2:
            scalars = self._dk2_2_grid
        elif scalar_field is ScalarField.DK1_22:
            scalars = self._dk1_22_grid
        elif scalar_field is ScalarField.DK2_11:
            scalars = self._dk2_11_grid
        else:
            scalars = None

        super().plot(plotter=plotter, show_edges=show_edges, scalars=scalars, show_principal_directions=show_principal_directions)
        if show_grid_points is True:
            grid_v = self._v.reshape([self._x_grid.shape[0], self._x_grid.shape[1], 3])
            if grid_step is not None:
                sampled_grid_v = grid_v[grid_step:self._x_grid.shape[0] - grid_step + 1:grid_step, grid_step:self._x_grid.shape[1] - grid_step + 1:grid_step]
            else:
                sampled_grid_v = grid_v[self._x_grid.shape[0] // 2, self._x_grid.shape[1] // 2]

            cloud = pv.PolyData(sampled_grid_v.reshape((-1, 3)))
            plotter.add_mesh(cloud, point_size=10.0, color="red", render_points_as_spheres=True)

    def calculate_codazzi_arguments(self) -> np.ndarray:
        # k1 = self._k1_grid                                                                                                          # 0
        # k2 = self._k2_grid                                                                                                          # 1
        # dk1_1 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k1_grid)    # 2
        # dk1_2 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k1_grid)    # 3
        # dk1_22 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=dk1_2)           # 4
        # dk2_1 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k2_grid)    # 5
        # dk2_2 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k2_grid)    # 6
        # dk2_11 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=dk2_1)           # 7
        #
        # step = 2
        # rows, cols = k1.shape
        # row_indices = np.arange(0, rows, step)
        # col_indices = np.arange(0, cols, step)

        row_indices = np.array([self._x_grid.shape[0] // 2])
        col_indices = np.array([self._x_grid.shape[1] // 2])

        return np.array([
            self._k1_grid[row_indices, col_indices],
            self._k2_grid[row_indices, col_indices],
            self._dk1_1_grid[row_indices, col_indices],
            self._dk1_2_grid[row_indices, col_indices],
            self._dk1_22_grid[row_indices, col_indices],
            self._dk2_1_grid[row_indices, col_indices],
            self._dk2_2_grid[row_indices, col_indices],
            self._dk2_11_grid[row_indices, col_indices]
        ])

    def downsample(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = fps(x=v, ratio=ratio)
        v = v[indices].cpu().detach().numpy()
        return Mesh.from_vertices(v=v)

    def _directional_derivative_at_point(self, point: np.ndarray, direction: np.ndarray, scalar_field: np.ndarray, h: float = 1e-8, max_attempts: int = 8) -> np.ndarray:
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

        idx_plus_grid = np.unravel_index(idx_plus, self._v_grid.shape[:-1])
        idx_minus_grid = np.unravel_index(idx_minus, self._v_grid.shape[:-1])
        return (scalar_field[idx_plus_grid] - scalar_field[idx_minus_grid]) / (2 * h)

    def _directional_derivative(self, direction_field: np.ndarray, scalar_field: np.ndarray, h: float = 1e-8) -> np.ndarray:
        vfunc = np.vectorize(self._directional_derivative_at_point, excluded={2}, signature='(n),(n)->()')
        return vfunc(self._v_grid, direction_field, scalar_field)

    def _principal_direction_derivative(self, principal_direction: PrincipalDirection, scalar_field: np.ndarray, h: float = 1e-8) -> np.ndarray:
        if principal_direction == PrincipalDirection.D1:
            return self._directional_derivative(direction_field=self._d1_grid, scalar_field=scalar_field, h=h)
        else:
            return self._directional_derivative(direction_field=self._d2_grid, scalar_field=scalar_field, h=h)
