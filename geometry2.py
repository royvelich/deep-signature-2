# standard library
from __future__ import annotations

import random
from math import sqrt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from enum import Enum
from scipy.spatial import Delaunay

# trimesh
# import trimesh

# pymesh
# import pymesh

# igl
import igl
import numpy as np

# scipy
from scipy.spatial import cKDTree

# numpy
import numpy as np

# sklearn
from sklearn.neighbors import KDTree

# torch-geometric
from torch_geometric.nn import fps

# torch
import torch

# from data.dataset_vis import generate_mesh_vis
from data.non_uniform_sampling import non_uniform_sampling, non_uniform_2d_sampling


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
        # self._d1, self._d2, self._k1, self._k2 = igl.principal_curvature(v=v, f=f)

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def f(self) -> np.ndarray:
        return self._f

    @v.setter
    def v(self, value):
        self._v = value

    # @property
    # def k1(self) -> np.ndarray:
    #     return self._k1
    #
    # @property
    # def k2(self) -> np.ndarray:
    #     return self._k2
    #
    # @property
    # def d1(self) -> np.ndarray:
    #     return self._d1
    #
    # @property
    # def d2(self) -> np.ndarray:
    #     return self._d2

    def plot(self, scalars: Optional[np.ndarray] = None, show_principal_directions: bool = False):
    # Add your visualization code here using an alternative library like matplotlib or Plotly.
        pass

    @staticmethod
    def from_file(file_path: Path) -> Mesh:
        v, f = igl.read_triangle_mesh(filename=str(file_path), dtypef=np.float64)
        return Mesh(v=v, f=f)

    @staticmethod
    def plot_meshes(meshes: List[Mesh], **kwargs):
        # Add your visualization code here using an alternative library like matplotlib or Plotly.
        pass

    def center(self) -> Mesh:
        t = np.mean(self._v, 0, keepdims=True)
        v = self._v - t
        return Mesh(v=v, f=self._f)


def normalize_points(vertices):
    """
    Normalize a set of 3D vertices by translation and rotation according to the covariance matrix principal directions.

    Args:
        vertices (numpy.ndarray): 3D array of shape (num_points, 3) representing the input vertices.

    Returns:
        numpy.ndarray: Normalized vertices of shape (num_points, 3).
    """
    # Step 1: Compute the centroid
    centroid = np.mean(vertices, axis=0)

    # Step 2: Compute the covariance matrix
    centered_points = vertices - centroid
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Step 3: Find the principal directions (eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors)
    # normalized_eigenvectors = eigenvectors / np.sqrt(eigenvalues)
    # Rotation matrix to align with principal axes
    rotation_matrix = normalized_eigenvectors

    # Apply rotation to the centered point cloud
    normalized_points = np.dot(centered_points, rotation_matrix)

    #
    #
    # # Step 4: Normalize the points
    # normalized_points = np.dot(centered_points, normalized_eigenvectors)


    return normalized_points


class Patch(Mesh):
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,downsample=True):
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._z_grid = z_grid

        x = x_grid.ravel()
        y = y_grid.ravel()
        z = z_grid.ravel()

        # self._v = np.stack([x, y, z], axis=1)  # Use only x and y coordinates
        # normalization needs to be before downsample
        # self._v = normalize_points(self._v)
        # x, y, z = self._v[:,0], self._v[:,1], self._v[:,2]
        self._v = np.stack([x, y], axis=1)  # Use only x and y coordinates
        # ratio = 0.5
        ratio = np.random.uniform(0.03, 0.05)
        # ratio = 0.05
        if downsample:
           indices = self.downsample_non_uniform(ratio=ratio)
        else:
            indices = np.arange(len(self._v))
        self._v = np.stack([x[indices], y[indices], z[indices]], axis=1)  # Use only x and y coordinates
        v = np.stack([self._v[:,0], self._v[:,1]], axis=1)
        # Perform triangulation using Delaunay method
        # self._f = igl.delaunay_triangulation(v)
        self._f = Delaunay(v).simplices


        # self._d1, self._d2, self._k1, self._k2 = igl.principal_curvature(v=v, f=self._f)
        # generate_mesh_vis(v=self._v, f=self._f)


        # x = torch.from_numpy(self._v[:,0])
        # y = torch.from_numpy(self._v[:,1])
        # z = torch.from_numpy(self._v[:,2])

        # calculate second moments
        # xx = x ** 2
        # yy = y ** 2
        # zz = z ** 2
        # xy = x * y
        # yz = y * z
        # zx = z * x
        #
        # # will be our input to the model
        # self.v_second_moments = torch.stack([x, y, z, xx, yy, zz, xy, yz, zx], axis=1)

    def plot(self, show_principal_directions: bool = True, show_grid_points: bool = False, grid_step: Optional[int] = None, scalar_field: Optional[ScalarField] = None):
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

        super().plot(scalars=scalars, show_principal_directions=show_principal_directions)
        if show_grid_points is True:
            grid_v = self._v.reshape([self._x_grid.shape[0], self._x_grid.shape[1], 3])
            if grid_step is not None:
                sampled_grid_v = grid_v[grid_step:self._x_grid.shape[0] - grid_step + 1:grid_step,
                                  grid_step:self._x_grid.shape[1] - grid_step + 1:grid_step]
            else:
                mid_x = self._x_grid.shape[0] // 2
                mid_y = self._x_grid.shape[1] // 2
                sampled_grid_v1 = grid_v[range(mid_x - 2, mid_x + 2 + 1), [mid_y] * 5]
                sampled_grid_v2 = grid_v[[mid_x] * 5, range(mid_y - 2, mid_y + 2 + 1)]
                sampled_grid_v = np.stack((sampled_grid_v1, sampled_grid_v2))

            # Add your visualization code for grid points using an alternative library like matplotlib or Plotly.

    def set_faces(self, faces):
        self._f = faces

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

    def downsample(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = fps(x=v, ratio=ratio)
        return indices

    def downsample_non_uniform(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = non_uniform_2d_sampling(int(sqrt(len(v))),  ratio=ratio)
        return indices

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

class Patch2(Mesh):
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,downsample=True):
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._z_grid = z_grid

        x = x_grid.ravel()
        y = y_grid.ravel()
        z = z_grid.ravel()

        # self._v = np.stack([x, y, z], axis=1)  # Use only x and y coordinates
        # normalization needs to be before downsample
        # self._v = normalize_points(self._v)
        # x, y, z = self._v[:,0], self._v[:,1], self._v[:,2]

        self._v = np.stack([x, y, z], axis=1)  # Use only x and y coordinates




class Torus(Mesh):
    def __init__(self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,downsample=True):
        self._x_grid = x_grid
        self._y_grid = y_grid
        self._z_grid = z_grid

        x = x_grid.ravel()
        y = y_grid.ravel()
        z = z_grid.ravel()

        self._v = np.stack([x, y], axis=1)  # Use only x and y coordinates
        if downsample:
           indices = self.downsample(ratio=random.uniform(0.8,0.82))
        else:
            indices = np.arange(len(self._v))
        self._v = np.stack([x[indices], y[indices], z[indices]], axis=1)  # Use only x and y coordinates
        self._v = normalize_points(self._v)
        v = np.stack([self._v[:,0], self._v[:,1]], axis=1)
        # Perform triangulation using Delaunay method
        self._f = igl.delaunay_triangulation(v)
        # fs = []
        # division = 16
        # v_division = int(len(v)/division)
        #
        # for i in range(division):
        #
        #     f  = Delaunay(v[i*v_division:(i+1)*v_division]).simplices
        #     fs.append(f)
        #
        # self._f = np.concatenate(fs)

        # self._d1, self._d2, self._k1, self._k2 = igl.principal_curvature(v=v, f=self._f)



        x = torch.from_numpy(self._v[:,0])
        y = torch.from_numpy(self._v[:,1])
        z = torch.from_numpy(self._v[:,2])

        # calculate second moments
        xx = x ** 2
        yy = y ** 2
        zz = z ** 2
        xy = x * y
        yz = y * z
        zx = z * x

        # will be our input to the model
        self.v_second_moments = torch.stack([x, y, z, xx, yy, zz, xy, yz, zx], axis=1)

    def downsample(self, ratio: float) -> Mesh:
        v = torch.tensor(data=self._v)
        indices = fps(x=v, ratio=ratio)
        return indices
