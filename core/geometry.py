# standard library
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
from enum import Enum

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
            show_edges: bool = False,
            show_principal_directions: bool = False):
        pyvista_mesh = pv.PolyData(self._v, self._pyvista_f)
        plotter.add_mesh(mesh=pyvista_mesh, show_edges=True)

        if show_principal_directions is True:
            pyvista_mesh['d1'] = self._d1
            pyvista_mesh['d2'] = self._d2
            glyphs1 = pyvista_mesh.glyph(orient='d1', scale=False, factor=0.1, tolerance=0.01)
            glyphs2 = pyvista_mesh.glyph(orient='d2', scale=False, factor=0.1, tolerance=0.01)
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
        self._k1_grid = self._k1.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])
        self._k2_grid = self._k2.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])
        self._v_grid = self._v.reshape([self._x_grid.shape[0], self._x_grid.shape[1], -1])
        self._tree = cKDTree(self._v)

    def plot(self,
             plotter: pv.Plotter,
             show_edges: bool = False,
             show_principal_directions: bool = False,
             show_center_point: bool = True):
        super().plot(plotter=plotter, show_edges=show_edges, show_principal_directions=show_principal_directions)
        if show_center_point is True:
            grid_v = self._v.reshape([self._x_grid.shape[0], self._x_grid.shape[1], 3])

            # specify step size
            step = 1

            # sample every 10th element
            # sampled_arr = grid_v[step:self._x_grid.shape[0] - step + 1:step, step:self._x_grid.shape[1] - step + 1:step]

            sampled_arr = grid_v[self._x_grid.shape[0] // 2, self._x_grid.shape[1] // 2 ]

            # Create a PolyData object from the points
            cloud = pv.PolyData(sampled_arr.reshape((-1, 3)))

            # Create a Plotter object and add our points to it, then plot
            plotter.add_mesh(cloud, point_size=10.0, color="red", render_points_as_spheres=True)

    def calculate_codazzi_arguments(self) -> np.ndarray:
        # indices = numpy.array([[self._x_grid.shape[0] // 2, self._x_grid.shape[1] // 2]])
        indices = None
        k1 = self._k1_grid[:, :, 0]
        k2 = self._k2_grid[:, :, 0]
        dk1_1 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k1_grid, indices=indices)
        dk1_2 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k1_grid, indices=indices)
        dk1_22 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=dk1_2, indices=indices)
        dk2_1 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k2_grid, indices=indices)
        dk2_2 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k2_grid, indices=indices)
        dk2_11 = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=dk2_1, indices=indices)
        # return np.stack((self._k1, self._k2, dk1_1, dk1_2, dk1_22, dk2_1, dk2_2, dk2_11), axis=0)

        # dk1_1 = np.expand_dims(dk1_1, axis=2)
        # dk1_2 = np.expand_dims(dk1_2, axis=2)
        # dk1_22 = np.expand_dims(dk1_22, axis=2)
        # dk2_1 = np.expand_dims(dk2_1, axis=2)
        # dk2_2 = np.expand_dims(dk2_2, axis=2)
        # dk2_11 = np.expand_dims(dk2_11, axis=2)

        # # specify step size
        # step = 40
        #
        # k1_reshape = self._k1.reshape((self._x_grid.shape[0], self._x_grid.shape[1]))
        # k2_reshape = self._k2.reshape((self._x_grid.shape[0], self._x_grid.shape[1]))
        #
        # # sample every 10th element
        # # k1_sampled = k1_reshape[step:self._x_grid.shape[0] - step + 1:step, step:self._x_grid.shape[1] - step + 1:step]
        # # k2_sampled = k2_reshape[step:self._x_grid.shape[0] - step + 1:step, step:self._x_grid.shape[1] - step + 1:step]
        #
        # k1_sampled = k1_reshape[self._x_grid.shape[0] // 2, self._x_grid.shape[1] // 2]
        # k2_sampled = k2_reshape[self._x_grid.shape[0] // 2, self._x_grid.shape[1] // 2]

        index1 = self._x_grid.shape[0] // 2
        index2 = self._x_grid.shape[1] // 2

        return np.array([
            k1[index1, index2],
            k2[index1, index2],
            dk1_1[index1, index2],
            dk1_2[index1, index2],
            dk1_22[index1, index2],
            dk2_1[index1, index2],
            dk2_2[index1, index2],
            dk2_11[index1, index2]
        ])

        # return np.stack((k1_sampled.ravel(), k2_sampled.ravel()), axis=0)

    def downsample(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = fps(x=v, ratio=ratio)
        v = v[indices].cpu().detach().numpy()
        return Mesh.from_vertices(v=v)

    def _directional_derivative_at_point(self, point: np.ndarray, direction: np.ndarray, scalar_field: np.ndarray, h: float = 1e-5, max_attempts: int = 8) -> np.ndarray:
        h = 1e-8
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

    def _directional_derivative(self, direction_field: np.ndarray, scalar_field: np.ndarray, indices: Optional[numpy.ndarray] = None, h: float = 1e-5) -> np.ndarray:
        # return np.array([self._directional_derivative_at_point(point=self._v[i], direction=direction_field[i], scalar_field=scalar_field, h=h) for i in range(self._v.shape[0])])

        if indices is None:
            indices_pool = np.ndindex(self._v_grid.shape[:2])
        else:
            indices_pool = [tuple(row) for row in indices]

        vfunc = np.vectorize(self._directional_derivative_at_point, excluded={2}, signature='(n),(n)->()')
        return vfunc(self._v_grid, direction_field, scalar_field)

        # return np.array([self._directional_derivative_at_point(point=self._v_grid[index], direction=direction_field[index], scalar_field=scalar_field, h=h) for index in indices_pool])

    def _principal_direction_derivative(self, principal_direction: PrincipalDirection, scalar_field: np.ndarray, indices: Optional[numpy.ndarray] = None,  h: float = 1e-5) -> np.ndarray:
        if principal_direction == PrincipalDirection.D1:
            return self._directional_derivative(direction_field=self._d1_grid, scalar_field=scalar_field, indices=indices, h=h)
        else:
            return self._directional_derivative(direction_field=self._d2_grid, scalar_field=scalar_field, indices=indices, h=h)
