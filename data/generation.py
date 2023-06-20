# standard library
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# numpy
import numpy as np
from numpy import vectorize

# scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf

# pyvista
import pyvista as pv

# surface-diff-inv
from core.geometry import Mesh

# noise
from noise import snoise3


class PatchGenerator(ABC):
    def __init__(self, limit: float, grid_size: int):
        self._limit = limit
        self._grid_size = grid_size

    @abstractmethod
    def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def generate(self) -> Mesh:
        X, Y, Z = self._generate()
        v = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return Mesh.from_vertices(v=v)


class GaussianPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, sigma: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._sigma = sigma

    def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create 2D arrays for X, Y
        x_linspace = np.linspace(-self._limit, self._limit, self._grid_size)
        y_linspace = np.linspace(-self._limit, self._limit, self._grid_size)
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

        # Generate random Z values
        z_grid = np.random.normal(0, 1, x_grid.shape)

        # Apply Gaussian smoothing
        z_grid = gaussian_filter(z_grid, sigma=self._sigma)

        return x_grid, y_grid, z_grid


class InverseFourierPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, scale: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._scale = scale

    def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = 20
        # Create 2D arrays for kx, ky
        kx, ky = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size))

        # Generate random complex Fourier coefficients
        S = (np.random.normal(size=(size, size)) + 1j * np.random.normal(size=(size, size)))

        # Scale the Fourier coefficients for smoothness
        S /= np.sqrt(kx ** 8 + ky ** 8 + 1e-10)

        # Perform inverse 2D Fourier transform to obtain Z values
        z_grid = np.fft.ifft2(S).real

        # Create 2D arrays for X, Y
        x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

        return x_grid, y_grid, z_grid


class RBFPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, points_count: int):
        super().__init__(limit=limit, grid_size=grid_size)
        self._points_count = points_count

    def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Generate random points
        x = np.random.uniform(low=-self._limit, high=self._limit, size=self._points_count)
        y = np.random.uniform(low=-self._limit, high=self._limit, size=self._points_count)
        z = np.random.uniform(low=-self._limit, high=self._limit, size=self._points_count)

        # Fit radial basis function to the points
        rbf = Rbf(x, y, z, function='multiquadric', smooth=2)

        # Create 2D grid
        linspace_x = np.linspace(-self._limit, self._limit, self._grid_size)
        linspace_y = np.linspace(-self._limit, self._limit, self._grid_size)
        x_grid, y_grid = np.meshgrid(linspace_x, linspace_y)

        # Evaluate RBF on the grid to get Z values
        z_grid = rbf(x_grid, y_grid)

        return x_grid, y_grid, z_grid


class SimplexNoisePatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, scale: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._scale = scale

    @staticmethod
    @vectorize
    def _snoise3_vectorized(x, y, z, octaves, persistence, lacunarity):
        return snoise3(x, y, z, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

    def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Generate 2D grid
        x = np.linspace(-self._limit, self._limit, self._grid_size)
        y = np.linspace(-self._limit, self._limit, self._grid_size)
        x_grid, y_grid = np.meshgrid(x, y)

        # Generate random offset
        offset = np.random.uniform(0, 10000)

        # Compute noise values for the grid points
        z_grid = SimplexNoisePatchGenerator._snoise3_vectorized(
            (x_grid + offset) / self._scale,
            (y_grid + offset) / self._scale,
            0,
            octaves=1,
            persistence=0.5,
            lacunarity=2.0)

        return x_grid, y_grid, z_grid
