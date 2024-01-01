import os
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.transforms import FaceToEdge

import wandb
from pyvista import PolyData
from sklearn.neighbors import KDTree, NearestNeighbors
import re
import trimesh.visual as visual

# import pyvista as pv
# from geometry2 import normalize_points


def rearange_mesh_faces(faces):
    faces = faces[np.arange(len(faces)) % 4 != 0]
    faces = faces.reshape(-1, 3)
    return faces

def calc_dki_j(points, desired_principal_dir, k_i, delta):
    """
    calculate derivative of k_i in the direction of desired_principal_dir
    :param points:
    :param desired_principal_dir:
    :param k_i:
    :param delta:
    :return:
    """
    output = np.zeros((len(points), 1))
    A = KDTree(points)
    for i in range(len(points)):
        p = points[i]
        p_delta = p + delta * desired_principal_dir[i]
        p_delta = p_delta[np.newaxis, :]
        p_minus_delta = p - delta * desired_principal_dir[i]
        p_minus_delta = p_minus_delta[np.newaxis, :]
        # calculate closest point to p_delta using kd-tree
        closest_point_p_delta = A.query(p_delta,k=2)
        dist_p_delta = closest_point_p_delta[0][0][0]
        p_delta = closest_point_p_delta[1][0][0]
        # calculate closest point to p_minus_delta
        closest_point_p_minus_delta = A.query(p_minus_delta, k=2)
        dist_p_minus_delta = closest_point_p_minus_delta[0][0][0]
        p_minus_delta = closest_point_p_minus_delta[1][0][0]
        # check if the closest point is the same as the point we are calculating the curvature for
        if closest_point_p_delta[1][0][0] == i:
            # dist_p_delta = closest_point_p_delta[0][0][1]
            p_delta = closest_point_p_delta[1][0][1]
        if closest_point_p_minus_delta[1][0][0] == i:
            # dist_p_minus_delta = closest_point_p_minus_delta[0][0][1]
            p_minus_delta = closest_point_p_minus_delta[1][0][1]



        output[i] = (k_i[p_delta] - k_i[p_minus_delta]) / (np.linalg.norm(points[p_delta] - points[p_minus_delta]))

    return output

def plot_mesh_and_color_by_k_or_dki_j(surf : PolyData, k, title):
    surf.plot(scalars=k, show_edges=False, text=title, cmap='jet', cpos='xy', screenshot='k.png')



# def plot_mesh_with_vector_field(surf : PolyData,visualization_surf : PolyData, k,  principal_directions1, principal_directions2,title):
#     plotter = pv.Plotter()
#     visualization_surf['principal_directions1'] = principal_directions1
#     visualization_surf['principal_directions2'] = principal_directions2
#     glyphs1 = visualization_surf.glyph(orient='principal_directions1', scale=False, factor=0.1)
#     glyphs2 = visualization_surf.glyph(orient='principal_directions2', scale=False, factor=0.1)
#     plotter.add_mesh(surf, scalars=k, show_edges=False)
#
#     plotter.add_mesh(glyphs1, color='blue', opacity=1)
#     plotter.add_mesh(glyphs2, color='red', opacity=1)
#     plotter.add_title(title=title)
#     plotter.show(screenshot='k.png')



# def generate_surface(grid_points=200):
#     grid_points_count = grid_points
#     grid_linspace = np.linspace(-1, 1, grid_points_count)
#     grid_x, grid_y = np.meshgrid(grid_linspace, grid_linspace)
#     points = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(shape=grid_points_count * grid_points_count)],
#                          axis=1)
#
#     points_tensor = torch.tensor(points)
#
#     # make the patch more irregular and more real world like
#     # indices = fps(x=points_tensor, ratio=0.1)
#     # sampled_points = points_tensor[indices]
#     # sampled_points = sampled_points.cpu().detach().numpy()
#     # sampled_points[:, 2] = 0.1*numpy.sin(2*numpy.pi*sampled_points[:, 0]*0.5) + numpy.cos(4*numpy.pi*sampled_points[:, 1]*0.1)
#     points_tensor = points_tensor.cpu().detach().numpy()
#     # scalar rand between 0 to 100
#     a = np.random.rand()/2
#     b = np.random.rand()/2
#     points_tensor[:, 2] = a* np.sin(2 * np.pi * points_tensor[:, 0] * 0.5) + b*np.cos(
#         4 * np.pi * points_tensor[:, 1] * 0.1)
#     # points_tensor[:, 2] = 0.1 * np.sin(2 * np.pi * points_tensor[:, 0] * 0.5) + np.cos(
#     #     4 * np.pi * points_tensor[:, 1] * 0.1)
#     # points_tensor[:, 2] = points_tensor[:, 0]**2 + points_tensor[:, 1]**2
#
#     cloud = pv.PolyData(points_tensor)
#     # cloud.plot(point_size=5)
#
#     surf = cloud.delaunay_2d()
#     # surf.plot(show_edges=True)
#
#     v = surf.points
#     f = surf.faces
#     f = rearange_mesh_faces(f)
#     return surf, v,f

def calculate_derivatives(v, f,delta=0.01):
    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    dk1_1 = calc_dki_j(v, v1, k1, delta)

    dk1_2 = calc_dki_j(v, v2, k1, delta)

    dk2_1 = calc_dki_j(v, v1, k2, delta)

    dk2_2 = calc_dki_j(v, v2, k2, delta)

    dk1_22 = calc_dki_j(v, v2, dk1_2, delta)

    dk2_11 = calc_dki_j(v, v1, dk2_1, delta)
    return k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11


def calculate_pearson_corr_matrix(k1, k2, dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11):



    # plot_mesh_and_color_by_k_or_dki_j(surf, k1, "k1")
    # plot_mesh_and_color_by_k_or_dki_j(surf, k2, "k2")


#     calculate pearson correlation matrix
    pearson_corr_matrix = np.corrcoef(np.hstack([k1[:,np.newaxis],k2[:,np.newaxis],dk1_1, dk1_2, dk2_1, dk2_2, dk1_22, dk2_11]),rowvar=False)

    return pearson_corr_matrix


def init_wandb(lr=0.001,max_epochs=100, weight_decay=0.001, dataset_path = ""):
    # Set a custom temporary directory for WandB
    wandb_dir = "./wandb_tmp_dir"
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_TEMP"] = wandb_dir

    wandb.login(key="fbd4729419e01772f8a97c41e71e422c6311e896")
    wandb.init(project="train_on_patches",
               # entity="geometric-dl",
               config={
                   "learning_rate": lr,
                   "architecture": "Point Transformer Net Mean pool",
                   "dataset": dataset_path,
                   "epochs": max_epochs,
                   "weight_decay": weight_decay
               })

    return WandbLogger()

# Define your custom transform function
# def custom_affine_transform(patch):
#     # Apply rotation and translation
#     transform = Compose([
#         RandomAffine(degrees=90, translate=(0.1, 0.1))
#     ])
#     patch.v = transform(torch.from_numpy(np.expand_dims(patch.v, axis=0)))
#     # calculate second moments of v
#     patch.v_second_moments = torch.stack([patch.v[:,0],patch.v[:,1],patch.v[:,2],patch.v[:,0]**2,patch.v[:,1]**2,patch.v[:,2]**2,patch.v[:,0]*patch.v[:,1],patch.v[:,0]*patch.v[:,2],patch.v[:,1]*patch.v[:,2]],dim=1)
#     return patch

def custom_euclidean_transform(patch):
    # Generate random rotation angle between 0 and 90 degrees
    theta = np.radians(np.random.uniform(0, 90))

    # Generate random translations
    translation = np.random.uniform(-0.5, 0.5, size=3)

    # Define rotation matrix for rotation around x,y,z axis
    rand_num = np.random.randint(0, 3)
    if rand_num == 0:
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta)],
                                    [0, np.sin(theta), np.cos(theta)]])
    elif rand_num == 1:
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                    [0, 1, 0],
                                    [-np.sin(theta), 0, np.cos(theta)]])
    else:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])


    # Apply rotation
    patch.v = np.dot(rotation_matrix, patch.v.T).T

    # Apply translation
    patch.v += translation

    # # Expand dimensions
    # patch.v = np.expand_dims(patch.v, axis=0)

    x = torch.from_numpy(patch.v[:, 0])
    y = torch.from_numpy(patch.v[:, 1])
    z = torch.from_numpy(patch.v[:, 2])
    # Calculate second moments of v
    patch.v_second_moments = torch.stack(
        [x,y,z,x ** 2, y ** 2, z ** 2, x * y, x * z, y * z], dim=1)

    return patch


# Function to convert MATLAB-style face indices to Python-style
def convert_indices(line):
    return re.sub(r'\d+', lambda x: str(int(x.group()) - 1), line)

def modify_obj_to_pywavefront_format(obj_path):
    # Read the original OBJ file
    with open('C:/Users\galyo\PycharmProjects\point_descriptor\mesh.obj', 'r') as f:
        lines = f.readlines()

    # Modify face indices and save to a new file
    modified_lines = [convert_indices(line) if line.startswith('f') else line for line in lines]
    with open('modified_mesh.obj', 'w') as f:
        f.writelines(modified_lines)

# Function to compute edges from faces
def compute_edges_from_faces(faces):
    edges = []
    for face in faces:
        edges.extend([(face[i], face[(i + 1) % 3]) for i in range(3)])
    edges = [tuple(sorted(edge)) for edge in edges]  # Remove duplicates
    edges = list(set(edges))  # Ensure uniqueness of edges
    # Convert the list of edges to a 2D tensor with two rows (source and target nodes)
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index


def compute_edges_from_faces2(faces):
    face_to_edge = FaceToEdge()

    return face_to_edge(faces)


def get_faces_containing_vertices(f, param):
    faces = []
    for face in f:
        if face[0] in param and face[1] in param and face[2] in param:
            # change indices from global mesh to local patch(from 0...N-1 to 0...k-1)
            face_curr = np.array(
                [np.where(param == face[0])[0][0], np.where(param == face[1])[0][0], np.where(param == face[2])[0][0]])
            faces.append(face_curr)
    return faces

def calculate_bounding_box(vertices):
    # Initialize min and max values for each dimension
    min_values = np.inf * np.ones(3)
    max_values = -np.inf * np.ones(3)

    # Iterate through each vertex
    for vertex in vertices:
        # Update min and max values for each dimension
        min_values = np.minimum(min_values, vertex)
        max_values = np.maximum(max_values, vertex)

    # Calculate the bounding box size
    bounding_box_size = max_values - min_values

    # Return the bounding box as a tuple of (min_point, max_point) and size
    return tuple(min_values), tuple(max_values), tuple(bounding_box_size)

def calculate_radius(vertices):
    # Calculate the bounding box and size
    min_point, max_point, bounding_box_size = calculate_bounding_box(vertices)

    # Use the maximum dimension of the bounding box as the radius
    radius = max(bounding_box_size) / 5.0

    return radius

def compute_patches_from_mesh(v, f, k=10, is_radius=False):
    if is_radius:
        radius = calculate_radius(v)
        nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(v)
        indices = nbrs.radius_neighbors(v, return_distance=False)

    else:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(v)
        # Find k-nearest neighbors
        distances, indices = nbrs.kneighbors(v)
    normalized_input = []
    for i in range(len(v)):
        # normalized_input.append(v[indices[i]])
        normalized_input.append(normalize_points_just_translation(vertices=v[indices[i]], center_point=v[i]))


    # Compute faces for each patch in normalized_input, use f only faces containing vertices from each patch
    faces = []
    for patch_index in range(len(normalized_input)):
        # get all faces that fully contained with vertices from normalized_input[patch_index]
        faces.append(get_faces_containing_vertices(f, indices[patch_index]))

    return normalized_input, faces



@staticmethod
def save_glb(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray = None, path: Path = ''):
    # Create a mesh with vertex colors
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
        # visual=TextureVisuals(uv=uv, image=image),
        visual=visual.ColorVisuals(vertex_colors=colors),
        process=False
    )

    # Export to glTF (non-binary)
    glb_file = mesh.export(file_type="glb")

    # Save to a file
    with open(str(path), "wb") as f:
        f.write(glb_file)

@staticmethod
def unregularize_a_regularize_mesh(v : np.ndarray, f : np.ndarray):
#     like decimate but in unregularize way
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh = mesh.remove_duplicate_faces()
    mesh = mesh.remove_duplicate_vertices()
    mesh = mesh.remove_infinite_values()


# region Mesh processing

# fixing pathologies on delaunay triangulation using ratio of inradius to circumradius
def calculate_area(vertices):
    # Function to calculate the area of a triangle given its vertices
    a, b, c = vertices
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def calculate_inradius(vertices):
    # Function to calculate the inradius of a triangle given its vertices
    a, b, c = vertices
    s = np.sum([np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]) / 2
    return calculate_area(vertices) / s

def calculate_circumradius(vertices):
    # Function to calculate the circumradius of a triangle given its vertices
    a, b, c = vertices
    A = calculate_area(vertices)
    return (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)) / (4 * A)

def calculate_inradius_to_circumradius_ratio(vertices):
    # Function to calculate the inradius-to-circumradius ratio of a triangle given its vertices
    return calculate_inradius(vertices) / calculate_circumradius(vertices)

def fix_pathologies(v, f):
    f_fixed = []
    min = 10000
    for face in f:
        ratio = calculate_inradius_to_circumradius_ratio(v[face])
        if ratio < min:
            min = ratio
        if ratio > 0.3:
            f_fixed.append(face)
        if calculate_area(v[face]) < 0.02:
            f_fixed.append(face)

    print(min)
    return np.array(f_fixed)





# endregion

def is_vertex_in_boundary(f, vertex_index, faces_threshold=4):
    t = 0
    for face in f:
        if vertex_index in face:
            t = t+1
        if t > faces_threshold:
            return False
    return True


def normalize_points_translation_and_rotation(vertices, center_point):
    """
    Normalize a set of 3D vertices by translation and rotation according to the covariance matrix principal directions.

    Args:
        vertices (numpy.ndarray): 3D array of shape (num_points, 3) representing the input vertices.

    Returns:
        numpy.ndarray: Normalized vertices of shape (num_points, 3).
    """
    # Step 1: Compute the centroid
    # centroid = np.mean(vertices, axis=0)

    # Step 2: Compute the covariance matrix
    centered_points = vertices - center_point
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


def normalize_points_translation_and_rotation_torch(vertices, center_point):
    """
    Normalize a set of 3D vertices by translation and rotation according to the covariance matrix principal directions.

    Args:
        vertices (torch.Tensor): 3D tensor of shape (num_points, 3) representing the input vertices.
        center_point (torch.Tensor): 1D tensor of shape (3,) representing the center point.

    Returns:
        torch.Tensor: Normalized vertices of shape (num_points, 3).
    """
    # Step 1: Compute the centroid
    # centroid = torch.mean(vertices, dim=0)

    # Step 2: Compute the covariance matrix
    centered_points = vertices - center_point
    covariance_matrix = torch.matmul(centered_points.t(), centered_points) / centered_points.size(0)

    # Step 3: Find the principal directions (eigenvectors)
    eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
    # Sort eigenvalues in descending order
    _, sorted_indices = torch.sort(eigenvalues[:, 0], descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    normalized_eigenvectors = eigenvectors / torch.norm(eigenvectors, dim=0)
    # normalized_eigenvectors = eigenvectors / torch.sqrt(eigenvalues)
    # Rotation matrix to align with principal axes
    rotation_matrix = normalized_eigenvectors

    # Apply rotation to the centered point cloud
    normalized_points = torch.matmul(centered_points, rotation_matrix)

    #
    #
    # # Step 4: Normalize the points
    # normalized_points = torch.matmul(centered_points, normalized_eigenvectors)

    return normalized_points

def normalize_points_just_translation(vertices, center_point):
    """
    Normalize a set of 3D vertices by translation and rotation according to the covariance matrix principal directions.

    Args:
        vertices (numpy.ndarray): 3D array of shape (num_points, 3) representing the input vertices.

    Returns:
        numpy.ndarray: Normalized vertices of shape (num_points, 3).
    """
    # Step 1: Compute the centroid
    # centroid = np.mean(vertices, axis=0)

    # Step 2: Compute the covariance matrix
    centered_points = vertices - center_point


    #
    #
    # # Step 4: Normalize the points
    # normalized_points = np.dot(centered_points, normalized_eigenvectors)


    return centered_points