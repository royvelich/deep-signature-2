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

    @abstractmethod
    def generate(self) -> Patch:
        pass


class GaussianPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, min_sigma: float, max_sigma: float, max_abs_z: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._max_abs_z = max_abs_z

    def generate(self) -> Patch:
        # Create 2D arrays for X, Y
        x_linspace = np.linspace(-self._limit, self._limit, self._grid_size)
        y_linspace = np.linspace(-self._limit, self._limit, self._grid_size)
        x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

        # Generate random Z values
        z_grid = np.random.uniform(-self._max_abs_z, self._max_abs_z, x_grid.shape)

        # sigma = np.random.uniform(low=self._min_sigma, high=self._max_sigma)

        # Define a list of valid function arguments
        sigma_args = [3, 3.5, 4, 5]

        # Select a random function argument
        sigma = np.random.choice(sigma_args)

        # Apply Gaussian smoothing
        z_grid = gaussian_filter(z_grid, sigma=sigma)

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
        points_count = np.random.randint(20, 800)
        # Generate random points
        x = np.random.uniform(low=-self._limit, high=self._limit, size=points_count)
        y = np.random.uniform(low=-self._limit, high=self._limit, size=points_count)
        z = np.random.uniform(low=-self._limit, high=self._limit, size=points_count)

        # Define a list of valid function arguments
        function_args = ['multiquadric']

        # Select a random function argument
        random_function_arg = np.random.choice(function_args)

        # Fit radial basis function to the points
        smooth = np.random.uniform(low=0.1, high=10)
        rbf = Rbf(x, y, z, function=random_function_arg, smooth=smooth)

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
    _rng = np.random.default_rng()
    def __init__(self, limit: float, grid_size: int, coeff_limit: float):
        super().__init__(limit=limit, grid_size=grid_size)
        self._coeff_limit = coeff_limit

    def generate(self) -> Patch:
        u = np.linspace(-self._limit, self._limit, self._grid_size)
        v = np.linspace(-self._limit, self._limit, self._grid_size)
        u_grid, v_grid = np.meshgrid(u, v)

        a = QuadraticMonagePatchGenerator._rng.uniform(-self._coeff_limit, self._coeff_limit)
        b = QuadraticMonagePatchGenerator._rng.uniform(-self._coeff_limit, self._coeff_limit)
        c = QuadraticMonagePatchGenerator._rng.uniform(-self._coeff_limit, self._coeff_limit)
        d = QuadraticMonagePatchGenerator._rng.uniform(-self._coeff_limit, self._coeff_limit)
        e = QuadraticMonagePatchGenerator._rng.uniform(-self._coeff_limit, self._coeff_limit)

        a = 1
        b = -0.2


        h = a * (u_grid * u_grid * u_grid) + b * (u_grid * v_grid) + a * (v_grid * v_grid * v_grid)
        # h = (a / 2) * (u_grid * u_grid) + (c / 2) * (v_grid * v_grid) + d * u_grid + e * v_grid
        return Patch(x_grid=u_grid, y_grid=v_grid, z_grid=h)


# https://math.stackexchange.com/questions/4722103/pearson-correlation-of-the-principal-curvatures
class QuadraticMonagePatchGenerator2(PatchGenerator):
    def __init__(self, limit: float, grid_size: int):
        super().__init__(limit=limit, grid_size=grid_size)

    def generate(self) -> Patch:
        x = np.linspace(-self._limit, self._limit, self._grid_size)
        y = np.linspace(-self._limit, self._limit, self._grid_size)

        k1 = np.random.uniform(low=-1, high=1)
        k2 = np.random.uniform(low=-1, high=1)

        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = k1 * x_grid ** 2 / 2 + k2 * y_grid ** 2 / 2

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)
