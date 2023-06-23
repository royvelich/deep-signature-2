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
from core.geometry import Mesh, Patch

# noise
from noise import snoise3


class PatchGenerator(ABC):
    def __init__(self, limit: float, grid_size: int):
        self._limit = limit
        self._grid_size = grid_size

    # @abstractmethod
    # def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     pass

    @abstractmethod
    def generate(self) -> Patch:
        pass
        # X, Y, Z = self._generate()
        # v = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        # return Mesh.from_vertices(v=v)


class GaussianPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, sigma: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._sigma = sigma

    def generate(self) -> Patch:
        # Create 2D arrays for X, Y
        x_linspace = np.linspace(-self._limit, self._limit, self._grid_size)
        y_linspace = np.linspace(-self._limit, self._limit, self._grid_size)
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

        # Generate random Z values
        z_grid = np.random.normal(0, 1, x_grid.shape)

        # Apply Gaussian smoothing
        z_grid = gaussian_filter(z_grid, sigma=self._sigma)

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)


class InverseFourierPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, scale: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._scale = scale

    def generate(self) -> Patch:
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

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)


class RBFPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, points_count: int):
        super().__init__(limit=limit, grid_size=grid_size)
        self._points_count = points_count

    def generate(self) -> Patch:
        # Generate random points
        x = np.random.uniform(low=-self._limit, high=self._limit, size=self._points_count)
        y = np.random.uniform(low=-self._limit, high=self._limit, size=self._points_count)
        z = np.random.uniform(low=-self._limit, high=self._limit, size=self._points_count)

        # Fit radial basis function to the points
        smooth = np.random.uniform(low=0.01, high=10)
        rbf = Rbf(x, y, z, function='linear', smooth=smooth)

        # Create 2D grid
        linspace_x = np.linspace(-self._limit, self._limit, self._grid_size)
        linspace_y = np.linspace(-self._limit, self._limit, self._grid_size)
        x_grid, y_grid = np.meshgrid(linspace_x, linspace_y)

        # Evaluate RBF on the grid to get Z values
        z_grid = rbf(x_grid, y_grid)

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)


class SimplexNoisePatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, scale: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._scale = scale

    @staticmethod
    @vectorize
    def _snoise3_vectorized(x, y, z, octaves, persistence, lacunarity):
        return snoise3(x, y, z, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

    def generate(self) -> Patch:
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

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)


# https://math.stackexchange.com/questions/4722103/pearson-correlation-of-the-principal-curvatures
class QuadraticMonagePatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int):
        super().__init__(limit=limit, grid_size=grid_size)

    def generate(self) -> Patch:
        u = np.linspace(-self._limit, self._limit, self._grid_size)
        v = np.linspace(-self._limit, self._limit, self._grid_size)
        u_grid, v_grid = np.meshgrid(u, v)

        coeff_limit = 1
        a = np.random.uniform(-coeff_limit, coeff_limit)
        b = np.random.uniform(-coeff_limit, coeff_limit)
        c = np.random.uniform(-coeff_limit, coeff_limit)
        h = a * u_grid * u_grid + 2 * b * u_grid * v_grid + c * v_grid * v_grid

        return Patch(x_grid=u_grid, y_grid=v_grid, z_grid=h)

