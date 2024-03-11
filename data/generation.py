# standard library
from abc import ABC, abstractmethod

# numpy
from typing import Tuple

import numpy as np
from numpy import vectorize, ndarray

# scipy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf

# pyvista

# surface-diff-inv
from geometry2 import Patch, Torus, Patch2

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
        # h = (a / 2) * (u_grid * u_grid) + b * (u_grid * v_grid) + (c / 2) * (v_grid * v_grid)
        h = (a / 2) * (u_grid * u_grid) + (c / 2) * (v_grid * v_grid) + d * u_grid + e * v_grid
        return Patch(x_grid=u_grid, y_grid=v_grid, z_grid=h)


# https://math.stackexchange.com/questions/4722103/pearson-correlation-of-the-principal-curvatures
class QuadraticMonagePatchGenerator2(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, downsample: bool = True):
        super().__init__(limit=limit, grid_size=grid_size)
        self.downsample = downsample

    def generate(self, k1=0, k2=0,grid_size_delta=0) -> tuple[
        Patch, ndarray | int | float | complex, ndarray | int | float | complex, int]:
        if grid_size_delta != 0:
            curr_grid_size = self._grid_size+grid_size_delta
        else:
            curr_grid_size = self._grid_size

        x = np.linspace(-self._limit, self._limit, curr_grid_size)
        y = np.linspace(-self._limit, self._limit, curr_grid_size)

        # calculated to be the index of the vertex with (x,y) closest to (0,0) and positive
        point0_0_index = int((curr_grid_size** 2 + curr_grid_size)/2)
        curvature_limit = 3
        if k1==0:
            k1 = np.random.uniform(low=-curvature_limit, high=curvature_limit)
            k2 = np.random.uniform(low=-curvature_limit, high=curvature_limit)

        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = k1 * x_grid ** 2 / 2 + k2 * y_grid ** 2 / 2

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, downsample=self.downsample), k1, k2,point0_0_index

class QuadraticMonagePatchPointCloudGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, downsample: bool = False, ratio: float = 0.05):
        super().__init__(limit=limit, grid_size=grid_size)
        self.downsample = downsample
        self.ratio = ratio

    def generate(self, k1=0, k2=0,grid_size_delta=0, patch_type='spherical') -> tuple[
        Patch2, ndarray | int | float | complex, ndarray | int | float | complex, int]:
        if grid_size_delta != 0:
            curr_grid_size = self._grid_size+grid_size_delta
        else:
            curr_grid_size = self._grid_size

        x = np.linspace(-self._limit, self._limit, curr_grid_size)
        y = np.linspace(-self._limit, self._limit, curr_grid_size)

        # calculated to be the index of the vertex with (x,y) closest to (0,0) and positive
        point0_0_index = int((curr_grid_size** 2 + curr_grid_size)/2)
        curvature_limit = 1.5
        if k1==0:
            eps = 0.01
            if patch_type=='spherical':
                if np.random.uniform() < 0.5:
                    k1 = np.random.uniform(low=eps, high=curvature_limit)
                    k2 = np.random.uniform(low=eps, high=curvature_limit)
                else:
                    k1 = np.random.uniform(low=-curvature_limit, high=-eps)
                    k2 = np.random.uniform(low=-curvature_limit, high=-eps)
            elif patch_type=='parabolic':
                if np.random.uniform() < 0.25:
                    k1 = np.random.uniform(low=eps, high=curvature_limit)
                    k2 = 0
                elif np.random.uniform() < 0.5 and np.random.uniform() > 0.25:
                    k1 = np.random.uniform(low=-curvature_limit, high=-eps)
                    k2 = 0
                elif np.random.uniform() < 0.75 and np.random.uniform() > 0.5:
                    k1 = 0
                    k2 = np.random.uniform(low=eps, high=curvature_limit)
                else:
                    k1 = 0
                    k2 = np.random.uniform(low=-curvature_limit, high=-eps)
            elif patch_type=='hyperbolic':
                if np.random.uniform() < 0.5:
                    k1 = np.random.uniform(low=eps, high=curvature_limit)
                    k2 = np.random.uniform(low=-curvature_limit, high=-eps)
                else:
                    k1 = np.random.uniform(low=-curvature_limit, high=-eps)
                    k2 = np.random.uniform(low=eps, high=curvature_limit)
            elif patch_type == 'planar':
                k1 = 0
                k2 = 0
            elif patch_type == 'NaN':
                pass
            else:
                raise ValueError('patch_type not recognized')


        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = k1 * x_grid ** 2 + k2 * y_grid ** 2

        return Patch2(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, downsample=self.downsample,ratio=self.ratio), k1, k2,point0_0_index




class QuadraticMonageParabolicPlanarPatchGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, downsample: bool = True):
        super().__init__(limit=limit, grid_size=grid_size)
        self.downsample = downsample

    def generate(self, k1=0, k2=0) -> tuple[Patch, ndarray | int | float | complex, ndarray | int | float | complex, int]:
        curr_grid_size = self._grid_size

        x = np.linspace(-self._limit, self._limit, curr_grid_size)
        y = np.linspace(-self._limit, self._limit, curr_grid_size)

        # calculated to be the index of the vertex with (x,y) closest to (0,0) and positive
        point0_0_index = int((curr_grid_size** 2 + curr_grid_size)/2)
        curvature_limit = 3
        k1_or_k2_zero = np.random.uniform(0,1)

        if k1_or_k2_zero < 0.35:
            k1 = np.random.uniform(low=-curvature_limit, high=curvature_limit)
            k2 = 0
        elif k1_or_k2_zero < 0.7:
            k1 = 0
            k2 = np.random.uniform(low=-curvature_limit, high=curvature_limit)
        else:
            k1 = 0
            k2 = 0

        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = k1 * x_grid ** 2 / 2 + k2 * y_grid ** 2 / 2

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, downsample=self.downsample), k1, k2,point0_0_index

class TorusGenerator(PatchGenerator):
    def __init__(self, limit: float, grid_size: int, downsample: bool = True):
        super().__init__(limit=limit, grid_size=grid_size)
        self.downsample = downsample

    def generate(self, R=1, r=0.5,grid_size_delta=0,k1=1,k2=1) -> Patch:
        if grid_size_delta != 0:
            curr_grid_size = self._grid_size+grid_size_delta
        else:
            curr_grid_size = self._grid_size

        u = np.linspace(0, 2 * np.pi, curr_grid_size)
        v = np.linspace(0, 2 * np.pi, curr_grid_size)

        u_grid, v_grid = np.meshgrid(u, v)
        x_grid = (R + r * np.cos(v_grid)) * np.cos(u_grid)
        y_grid = (R + r * np.cos(v_grid)) * np.sin(u_grid)
        z_grid = r * np.sin(v_grid)

        return Torus(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, downsample=self.downsample), R, r, 0


class PeakSaddleGenerator(PatchGenerator):

    def __init__(self, limit: float, grid_size: int, downsample: bool = True):
        super().__init__(limit=limit, grid_size=grid_size)
        self.downsample = downsample

    def generate(self, grid_size_delta=0, shape="peak") -> Patch:
        if grid_size_delta != 0:
            curr_grid_size = self._grid_size+grid_size_delta
        else:
            curr_grid_size = self._grid_size

        x = np.linspace(-self._limit, self._limit, curr_grid_size)
        y = np.linspace(-self._limit, self._limit, curr_grid_size)
        if 0.0 not in x:
            x = np.insert(x, np.searchsorted(x, 0), 0)

        if 0.0 not in y:
            y = np.insert(y, np.searchsorted(y, 0), 0)

        x_grid, y_grid = np.meshgrid(x, y)

        if shape == "peak1":
            z_grid = -x_grid ** 2 - y_grid ** 2
        elif shape == "peak2":
            z_grid = 1.5*x_grid ** 2 + 0.2*y_grid ** 2
        elif shape == "peak3":
            z_grid = 0.5*x_grid ** 2 + 0.5*y_grid ** 2
        elif shape == "peak4":
            z_grid = 1.5*x_grid ** 2 + 1.5*y_grid ** 2
        elif shape == "saddle1":
            z_grid = 0.2*x_grid ** 2 - 0.2*y_grid ** 2
        elif shape == "saddle2":
            z_grid = 1.5*x_grid ** 2 - 0.5*y_grid ** 2
        elif shape == "saddle3":
            z_grid = x_grid ** 2 - y_grid ** 2
        elif shape == "saddle4":
            z_grid = 1.5 * x_grid ** 2 - 1.5* y_grid ** 2
        elif shape == "parabolic1":
            z_grid = 0.5*x_grid**2
        elif shape == "mixed1":
            z_grid = x_grid**2 + x_grid*y_grid - y_grid**2

        elif shape == "random_order_2":
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(-1, 1)
            c = np.random.uniform(-1, 1)
            z_grid = a*x_grid**2 + b*y_grid**2 + c*x_grid*y_grid

        return Patch(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, downsample=self.downsample)