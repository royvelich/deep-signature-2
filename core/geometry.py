# standard library
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from enum import Enum

# trimesh
import trimesh

# pymesh
import pymesh

# igl
import igl

# scipy
from scipy.spatial import cKDTree
from scipy.stats import norm

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

# open3d
import open3d as o3d

# scipy
from scipy.spatial import Delaunay


class PrincipalCurvature(Enum):
    K1 = 1
    K2 = 2


class PrincipalDirection(Enum):
    D1 = 1
    D2 = 2


class GridDirection(Enum):
    X = 1
    Y = 2


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
        pass

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
            show_edges: bool = False,
            show_principal_directions: bool = False):
        # pyvista_mesh = pv.PolyData(self._v, self._pyvista_f)
        # plotter.add_mesh(mesh=pyvista_mesh, scalars=scalars, show_edges=show_edges)

        pyvista_mesh = pv.PolyData(self._v)
        plotter.add_mesh(mesh=pyvista_mesh, color='red', point_size=5)

        # if show_principal_directions is True:
        #     pyvista_mesh['d1'] = self._d1
        #     pyvista_mesh['d2'] = self._d2
        #     factor = 0.03 * (np.abs(np.max(self._v) - np.min(self._v)))
        #     tolerance = 0.001
        #
        #     glyphs1 = pyvista_mesh.glyph(orient='d1', scale=False, factor=factor, tolerance=tolerance)
        #     glyphs2 = pyvista_mesh.glyph(orient='d2', scale=False, factor=factor, tolerance=tolerance)
        #     plotter.add_mesh(glyphs1, color='red', opacity=1)
        #     plotter.add_mesh(glyphs2, color='blue', opacity=1)

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
        # self._tree = cKDTree(self._v)
        # self._dk1_1_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k1_grid)
        # self._dk1_2_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k1_grid)
        # self._dk1_22_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._dk1_2_grid)
        # self._dk2_1_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._k2_grid)
        # self._dk2_2_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D2, scalar_field=self._k2_grid)
        # self._dk2_11_grid = self._principal_direction_derivative(principal_direction=PrincipalDirection.D1, scalar_field=self._dk2_1_grid)

    def plot(self,
             plotter: pv.Plotter,
             show_edges: bool = True,
             show_principal_directions: bool = True,
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
                mid_x = self._x_grid.shape[0] // 2
                mid_y = self._x_grid.shape[1] // 2
                # sampled_grid_v = grid_v[range(mid_x - 2, mid_x + 2 + 1), range(mid_y - 2, mid_y + 2 + 1)]
                sampled_grid_v1 = grid_v[range(mid_x - 2, mid_x + 2 + 1), [mid_y] * 5]
                sampled_grid_v2 = grid_v[[mid_x] * 5, range(mid_y - 2, mid_y + 2 + 1)]
                sampled_grid_v = np.stack((sampled_grid_v1, sampled_grid_v2))

            cloud = pv.PolyData(sampled_grid_v.reshape((-1, 3)))
            plotter.add_mesh(cloud, point_size=15.0, color="black", render_points_as_spheres=True)

    def _calculate_derivative(
            self,
            principal_curvature: PrincipalCurvature,
            principal_direction: PrincipalDirection,
            p_dir_to_g_dir: Dict[PrincipalDirection, GridDirection],
            mid_x: int,
            mid_y: int,
            order: int,
            h: float,
            accuracy: int) -> float:
        k = np.zeros(shape=5)
        scalar_field = self._k1_grid if principal_curvature == PrincipalCurvature.K1 else self._k2_grid

        if p_dir_to_g_dir[principal_direction] == GridDirection.X:
            k = scalar_field[range(mid_x - 4, mid_x + 4 + 1), mid_y]
        elif p_dir_to_g_dir[principal_direction] == GridDirection.Y:
            k = scalar_field[mid_x, range(mid_y - 4, mid_y + 4 + 1)]

        table = {}
        table[(1, 2)] = np.array([0, 0, 0, -1/2, 0, 1/2, 0, 0, 0])
        table[(1, 4)] = np.array([0, 0, 1/12, -2/3, 0, 2/3, -1/12, 0, 0])
        table[(1, 6)] = np.array([0, -1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60, 0])
        table[(1, 8)] = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])
        table[(2, 2)] = np.array([0, 0, 0, 1, -2, 1, 0, 0, 0])
        table[(2, 4)] = np.array([0, 0, -1/12, 4/3, -5/2, 4/3, -1/12, 0, 0])
        table[(2, 6)] = np.array([0, 1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90, 0])
        table[(2, 8)] = np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])

        coeffs = table[(order, accuracy)]
        return coeffs.dot(k) / np.power(h, order)

    def calculate_codazzi_arguments(self, accuracy: int) -> np.ndarray:
        # row_indices = np.array([self._x_grid.shape[0] // 2])
        # col_indices = np.array([self._x_grid.shape[1] // 2])
        #
        # return np.array([
        #     self._k1_grid[row_indices, col_indices],
        #     self._k2_grid[row_indices, col_indices],
        #     self._dk1_1_grid[row_indices, col_indices],
        #     self._dk1_2_grid[row_indices, col_indices],
        #     self._dk1_22_grid[row_indices, col_indices],
        #     self._dk2_1_grid[row_indices, col_indices],
        #     self._dk2_2_grid[row_indices, col_indices],
        #     self._dk2_11_grid[row_indices, col_indices]
        # ])

        mid_x = self._x_grid.shape[0] // 2
        mid_y = self._x_grid.shape[1] // 2
        # sampled_grid_v = grid_v[range(mid_x - 2, mid_x + 2 + 1), range(mid_y - 2, mid_y + 2 + 1)]
        x_dir = self._v_grid[mid_x + 1, mid_y] - self._v_grid[mid_x, mid_y]
        y_dir = self._v_grid[mid_x, mid_y + 1] - self._v_grid[mid_x, mid_y]
        p1_dir = self._d1_grid[mid_x, mid_y]
        p2_dir = self._d2_grid[mid_x, mid_y]

        p_dir_to_g_dir = {}
        if np.abs(x_dir.dot(p1_dir)) < 1e-6:
            p_dir_to_g_dir[PrincipalDirection.D1] = GridDirection.Y
            p_dir_to_g_dir[PrincipalDirection.D2] = GridDirection.X
        else:
            p_dir_to_g_dir[PrincipalDirection.D1] = GridDirection.X
            p_dir_to_g_dir[PrincipalDirection.D2] = GridDirection.Y

        h = np.abs(self._y_grid[mid_x, mid_y] - self._y_grid[mid_x + 1, mid_y])

        k1 = self._k1_grid[mid_x, mid_y]
        k2 = self._k2_grid[mid_x, mid_y]
        dk1_1 = self._calculate_derivative(principal_curvature=PrincipalCurvature.K1, principal_direction=PrincipalDirection.D1, p_dir_to_g_dir=p_dir_to_g_dir, mid_x=mid_x, mid_y=mid_y, order=1, h=h, accuracy=accuracy)
        dk1_2 = self._calculate_derivative(principal_curvature=PrincipalCurvature.K1, principal_direction=PrincipalDirection.D2, p_dir_to_g_dir=p_dir_to_g_dir, mid_x=mid_x, mid_y=mid_y, order=1, h=h, accuracy=accuracy)
        dk1_22 = self._calculate_derivative(principal_curvature=PrincipalCurvature.K1, principal_direction=PrincipalDirection.D2, p_dir_to_g_dir=p_dir_to_g_dir, mid_x=mid_x, mid_y=mid_y, order=2, h=h, accuracy=accuracy)
        dk2_1 = self._calculate_derivative(principal_curvature=PrincipalCurvature.K2, principal_direction=PrincipalDirection.D1, p_dir_to_g_dir=p_dir_to_g_dir, mid_x=mid_x, mid_y=mid_y, order=1, h=h, accuracy=accuracy)
        dk2_2 = self._calculate_derivative(principal_curvature=PrincipalCurvature.K2, principal_direction=PrincipalDirection.D2, p_dir_to_g_dir=p_dir_to_g_dir, mid_x=mid_x, mid_y=mid_y, order=1, h=h, accuracy=accuracy)
        dk2_11 = self._calculate_derivative(principal_curvature=PrincipalCurvature.K2, principal_direction=PrincipalDirection.D1, p_dir_to_g_dir=p_dir_to_g_dir, mid_x=mid_x, mid_y=mid_y, order=2, h=h, accuracy=accuracy)

        return np.stack(([k1], [k2], [dk1_1], [dk1_2], [dk1_22], [dk2_1], [dk2_2], [dk2_11]))

    def _generate_random_gaussian_param(self, N):
        mean = np.random.randint(0, N)
        std_dev = np.random.randint(10, N // 4)
        return mean, std_dev

    def _generate_random_gaussian_params(self, N):
        gaussians_params = []
        gaussian_num = np.random.randint(5, 20)
        for i in range(gaussian_num):
            mean, std_dev = self._generate_random_gaussian_param(N)
            gaussians_params.append((mean, std_dev))
        return gaussians_params

    def downsample2(self, ratio: float):
        # Step 1: Define N indices
        indices = np.arange(self._v.shape[0])

        # Step 2: Compose a PDF by summing multiple Gaussians
        probabilities = np.zeros(self._v.shape[0])
        gaussians_params = self._generate_random_gaussian_params(N=self._v.shape[0])
        for mean, std_dev in gaussians_params:
            probabilities += norm.pdf(indices, loc=mean, scale=std_dev)

        # Normalize the composed PDF
        probabilities /= probabilities.sum()

        # Step 3: Sample K items without replacement based on the non-uniform PDF
        sampled_items = np.random.choice(indices, size=int(ratio * self._v.shape[0]), replace=False, p=probabilities)

        v = self._v[sampled_items]
        f = Delaunay(v[:, :2]).simplices
        return Mesh(v=v, f=f)

    def downsample(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = fps(x=v, ratio=ratio)
        v = v[indices].cpu().detach().numpy()
        f = Delaunay(v[:, :2]).simplices
        return Mesh(v=v, f=f)

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
