import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import matplotlib
import torch
from scipy.stats import norm, multivariate_normal


# matplotlib.use('TkAgg')  # Use Tkinter as the backend; you can try other backends as well

def density_function_sampling(mesh_points, density_function, ratio):
    # Create Delaunay triangulation
    triangulation = Delaunay(mesh_points)

    # Calculate density values for each mesh point
    density_values = density_function(mesh_points)

    # Normalize density values to [0, 1]
    normalized_density = (density_values - np.min(density_values)) / (np.max(density_values) - np.min(density_values))

    # Sample points based on normalized density
    sampled_indices = np.arange(len(mesh_points))
    sampled_indices = np.random.choice(sampled_indices, size=int(len(sampled_indices)*ratio), replace=False, p=normalized_density / np.sum(normalized_density))

    # Get the sampled points
    sampled_points = mesh_points[sampled_indices]

    return sampled_indices

def non_uniform_sampling(N, ratio):
    gaussians_params = generate_random_gaussian_params(N)
    K = int(N * ratio)
    # Step 1: Define N indices
    indices = np.arange(N)

    # Step 2: Compose a PDF by summing multiple Gaussians
    probabilities = np.zeros(N)
    for mean, std_dev in gaussians_params:
        probabilities += norm.pdf(indices, loc=mean, scale=std_dev)

    # Normalize the composed PDF
    probabilities /= probabilities.sum()

    # Step 3: Sample K items without replacement based on the non-uniform PDF
    sampled_items = np.random.choice(indices, size=K, replace=False, p=probabilities)

    return sampled_items

def generate_random_2d_gaussian_params(N):
    mean = np.random.randint(0, N, size=2)
    covariance = np.random.randint(1, N/2, size=2)
    return mean, covariance

def generate_random_2d_gaussian_params_set(N):
    gaussian_params_set = []
    num_gaussians = np.random.randint(250, 300)
    for i in range(num_gaussians):
        mean, covariance = generate_random_2d_gaussian_params(N)
        gaussian_params_set.append((mean, covariance))
    return gaussian_params_set

def non_uniform_2d_sampling(grid_size, ratio):
    gaussian_params_set = generate_random_2d_gaussian_params_set(grid_size)
    K = int(grid_size * grid_size * ratio)

    # Step 1: Define N indices
    indices = np.arange(grid_size)
    x, y = np.meshgrid(indices, indices)

    # Step 2: Compose a 2D PDF by summing multiple 2D Gaussians
    probabilities = np.zeros((grid_size, grid_size))
    for mean, covariance in gaussian_params_set:
        mvn = multivariate_normal(mean=mean, cov=np.diag(covariance))
        pos = np.column_stack((x.ravel(), y.ravel()))
        probabilities += mvn.pdf(pos).reshape(grid_size, grid_size)

    # Normalize the composed PDF
    probabilities /= probabilities.sum()

    # Step 3: Sample K items without replacement based on the non-uniform 2D PDF
    flat_indices = np.arange(grid_size * grid_size)
    sampled_flat_indices = np.random.choice(flat_indices, size=K, replace=False, p=probabilities.flatten())
    # sampled_items = np.column_stack(np.unravel_index(sampled_flat_indices, (grid_size, grid_size)))


    return sampled_flat_indices

def generate_random_gaussian_param(N):
    mean = np.random.randint(0, N)
    std_dev = np.random.randint(1, N/7)
    return mean, std_dev

def generate_random_gaussian_params(N):
    gaussians_params= []
    gaussian_num = np.random.randint(5, 10)
    for i in range(gaussian_num):
        mean, std_dev = generate_random_gaussian_param(N)
        gaussians_params.append((mean, std_dev))
    return gaussians_params



# Example density function using PyTorch
# def density_function(points):
#     # Convert numpy array to PyTorch tensor
#     points_tensor = torch.tensor(points, dtype=torch.float32)
#
#     # Example: higher density towards the center of the mesh
#     center = torch.mean(points_tensor, dim=0)
#     distances = torch.norm(points_tensor - center, dim=1)
#     density_values = torch.exp(-distances)  # Adjust this function as needed
#
#     # Convert the result back to a numpy array
#     return density_values.numpy()

def vis_example():
    # Example usage
    mesh_points = np.random.rand(100, 2)  # Replace this with your actual mesh points
    # sampled_points = non_uniform_sampling(mesh_points, density_function, ratio=0.1)
    sampled_points = non_uniform_sampling(100, 10, generate_random_gaussian_params())
    sampled_points = mesh_points[sampled_points]
    # Plotting for visualization
    plt.scatter(mesh_points[:, 0], mesh_points[:, 1], c='blue', alpha=0.5, label='Mesh Points')
    plt.scatter(sampled_points[:, 0], sampled_points[:, 1], c='red', label='Sampled Points')
    plt.legend()
    plt.show()

# vis_example()
# N = 100
# ratio = 0.2
# sampled_items_2d = non_uniform_2d_sampling(N, ratio)
# print(sampled_items_2d)