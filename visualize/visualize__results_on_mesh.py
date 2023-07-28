import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import igl
import wandb

def calculate_knn(vertices, radius):
    """
    Find all points within the specified radius for each point in the 3D shape.

    Args:
        vertices (numpy.ndarray): 3D shape represented as (num_of_points, 3).
        radius (float): Radius to search for neighbors.

    Returns:
        list: A list of numpy arrays containing the neighbors for each point.
    """
    tree = cKDTree(vertices)
    indices = tree.query_ball_point(vertices, r=radius)

    # Convert the indices to numpy arrays
    # knn_shape = [vertices[idxs] for idxs in indices]

    return indices

def barycentric_average(vertices, faces, values):
    """
    Compute the vertex colors using barycentric averaging.

    Args:
        vertices (numpy.ndarray): The original 3D shape represented as (num_of_points, 3).
        faces (numpy.ndarray): The faces of the 3D shape represented as (num_of_faces, 3),
                               where each entry contains the indices of the vertices for each face.
        values (numpy.ndarray): The values to be used for coloring the shape, in the length of the number of points.

    Returns:
        numpy.ndarray: The vertex colors for each vertex in the shape.
    """
    values = values.cpu().numpy()  # Move values to the CPU

    vertex_colors = np.zeros_like(vertices)

    for face in faces:
        v0, v1, v2 = vertices[face]
        val_v0, val_v1, val_v2 = values[face]

        # Calculate barycentric coordinates for all vertices of the face at once
        total_area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v0 - v1))
        barycentric_coords = np.column_stack([
            0.5 * np.linalg.norm(np.cross(v2 - v1, vertices - v1), axis=1) / total_area,
            0.5 * np.linalg.norm(np.cross(v0 - v2, vertices - v2), axis=1) / total_area,
            0.5 * np.linalg.norm(np.cross(v1 - v0, vertices - v0), axis=1) / total_area,
        ])

        # Calculate the vertex colors using barycentric averaging for the current face
        vertex_color = np.sum(barycentric_coords[:, np.newaxis] * [val_v0, val_v1, val_v2], axis=0)

        # Accumulate the vertex colors for each vertex in the face
        vertex_colors[face] += vertex_color

    # Normalize the vertex colors
    vertex_counts = np.bincount(faces.ravel(), minlength=len(vertices))
    vertex_colors /= vertex_counts[:, np.newaxis]

    return vertex_colors


def plot_on_shape(vertices, faces, values):
    """
    Plot the shape with colors based on the values array.

    Args:
        vertices (numpy.ndarray): The original 3D shape represented as (num_of_points, 3).
        faces (numpy.ndarray): The faces of the 3D shape represented as (num_of_faces, 3), where each entry contains
                               the indices of the vertices for each face.
        values (numpy.ndarray): The values to be used for coloring the shape, in the length of the number of points.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot original shape with faces
    ax.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))

    # Plot the shape with colors based on the values array
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)  # Normalize the values to [0, 1]

    for i, face in enumerate(faces):
        # Calculate the average value for the vertices of the current face
        avg_value = torch.mean(normalized_values[face])
        face_color = plt.cm.coolwarm(avg_value.detach().numpy())  # Use a colormap to get the color based on the value

        # Plot the face with the computed color
        ax.add_collection3d(Poly3DCollection([vertices[face]], alpha=1.0, facecolors=face_color))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def plot_shape_color_faces(vertices, faces, values1, values2, k1, k2):
    """
    Plot the shape with colors based on the values1 and values2 arrays.
    Additionally, calculate k1 and k2 using the igl library.

    Args:
        vertices (numpy.ndarray): The original 3D shape represented as (num_of_points, 3).
        faces (numpy.ndarray): The faces of the 3D shape represented as (num_of_faces, 3),
                               where each entry contains the indices of the vertices for each face.
        values1 (numpy.ndarray): The first set of values to be used for coloring the shape,
                                 in the length of the number of points.
        values2 (numpy.ndarray): The second set of values to be used for coloring the shape,
                                 in the length of the number of points.

    Returns:
        None
    """
    values1 = barycentric_average(vertices,faces, values1)
    values2 = barycentric_average(vertices,faces, values2)
    k1 = barycentric_average(vertices,faces, k1)
    k2 = barycentric_average(vertices,faces, k2)
    fig = plt.figure(figsize=(12, 12))
    # Plot original shape with faces
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))
    ax1.set_title("Values1")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.set_title("Values2")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_title("k1")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.set_title("k2")

    # Plot original shape with faces
    ax1.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))
    ax2.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))
    ax3.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))
    ax4.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))

    # Plot the shape with colors based on values1
    min_value1, max_value1 = values1.min(), values1.max()
    normalized_values1 = (values1 - min_value1) / (max_value1 - min_value1)  # Normalize values1 to [0, 1]
    # Plot the shape with colors based on values2
    min_value2, max_value2 = values2.min(), values2.max()
    normalized_values2 = (values2 - min_value2) / (max_value2 - min_value2)  # Normalize values2 to [0, 1]
    # Plot the shape with colors based on k1 and k2
    min_k1, max_k1 = k1.min(), k1.max()
    normalized_k1 = (k1 - min_k1) / (max_k1 - min_k1)
    # Plot the shape with colors based on k1 and k2
    min_k2, max_k2 = k2.min(), k2.max()
    normalized_k2 = (k2 - min_k2) / (max_k2 - min_k2)
    for i, face in enumerate(faces):
        # Calculate the average value for the vertices of the current face
        avg_value1 = np.mean(normalized_values1[face])
        face_color1 = plt.cm.coolwarm(avg_value1)  # Use a colormap to get the color based on values1
        avg_value2 = np.mean(normalized_values2[face])
        face_color2 = plt.cm.coolwarm(avg_value2)  # Use a colormap to get the color based on values2

        # Plot the face with the computed color based on values1
        ax1.add_collection3d(Poly3DCollection([vertices[face]], alpha=1.0, facecolors=face_color1))

        # Plot the face with the computed color based on values2
        ax2.add_collection3d(Poly3DCollection([vertices[face]], alpha=1.0, facecolors=face_color2))

    # Plot the shape with colors based on k1 and k2
    for i, face in enumerate(faces):
        # Calculate the average value for the vertices of the current face
        avg_k1 = np.mean(normalized_k1[face])
        avg_k2 = np.mean(normalized_k2[face])
        face_color_k1 = plt.cm.coolwarm(avg_k1)  # Use a colormap to get the color based on k1
        face_color_k2 = plt.cm.coolwarm(avg_k2)  # Use a colormap to get the color based on k2

        # Plot the face with the computed colors based on k1 and k2
        ax3.add_collection3d(Poly3DCollection([vertices[face]], alpha=1.0, facecolors=face_color_k1))
        ax4.add_collection3d(Poly3DCollection([vertices[face]], alpha=1.0, facecolors=face_color_k2))

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")

    plt.savefig("shape_plot.png")
    plt.show()

    wandb.log({"plot": wandb.Image("shape_plot.png")})


def plot_shape_color_vertices(vertices, faces, values1, values2, k1, k2):
    """
        Plot the shape with colors based on the values1 and values2 arrays.
        Additionally, calculate k1 and k2 using the igl library.

        Args:
            vertices (numpy.ndarray): The original 3D shape represented as (num_of_points, 3).
            faces (numpy.ndarray): The faces of the 3D shape represented as (num_of_faces, 3),
                                   where each entry contains the indices of the vertices for each face.
            values1 (numpy.ndarray): The first set of values to be used for coloring the vertices,
                                     in the length of the number of points.
            values2 (numpy.ndarray): The second set of values to be used for coloring the vertices,
                                     in the length of the number of points.
            k1 (numpy.ndarray): The k1 values to be used for coloring the vertices,
                                in the length of the number of points.
            k2 (numpy.ndarray): The k2 values to be used for coloring the vertices,
                                in the length of the number of points.

        Returns:
            None
        """
    fig = plt.figure(figsize=(12, 12))

    # Calculate k1 and k2 using igl
    vertices = vertices.astype(np.double)  # Convert vertices to double for igl
    faces = faces.astype(np.int32)  # Convert faces to int32 for igl

    # Normalize values1, values2, k1, and k2 to [0, 1]
    values1 = (values1 - values1.min()) / (values1.max() - values1.min())
    values2 = (values2 - values2.min()) / (values2.max() - values2.min())
    k1 = (k1 - k1.min()) / (k1.max() - k1.min())
    k2 = (k2 - k2.min()) / (k2.max() - k2.min())

    # Plot original shape with faces
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_title("Values1")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.set_title("Values2")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.set_title("k1")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.set_title("k2")

    for ax in [ax1, ax2, ax3, ax4]:
        # Plot original shape with faces
        ax.add_collection3d(Poly3DCollection([vertices[face] for face in faces], alpha=0.5, facecolors='b'))

    # Plot the shape with colors based on values1
    for i, vertex in enumerate(vertices):
        face_indices = np.where(np.all(faces == i, axis=1))[0]
        if len(face_indices) == 0:
            continue
        color1 = plt.cm.coolwarm(values1[i])  # Use a colormap to get the color based on values1
        ax1.scatter(vertex[0], vertex[1], vertex[2], color=color1)
        color2 = plt.cm.coolwarm(values2[i])  # Use a colormap to get the color based on values2
        ax2.scatter(vertex[0], vertex[1], vertex[2], color=color2)
        color3 = plt.cm.coolwarm(k1[i])  # Use a colormap to get the color based on k1
        ax3.scatter(vertex[0], vertex[1], vertex[2], color=color3)
        color4 = plt.cm.coolwarm(k2[i])  # Use a colormap to get the color based on k2
        ax4.scatter(vertex[0], vertex[1], vertex[2], color=color4)

    plt.show()









def forward_with_knn(model, shape, radius, k1, k2, device):
    """
    Perform a forward pass through the given deep learning model using KNN for each point in the 3D shape.

    Args:
        model: The deep learning model.
        shape (numpy.ndarray): 3D shape represented as (num_of_points, 3).
        k (int): Number of nearest neighbors to consider.
        radius (float): Radius to search for neighbors.

    Returns:
        numpy.ndarray: The output from the deep learning model represented as (num_of_points, 3).
    """
    indices = calculate_knn(shape.v, radius)
    input = [shape.v_second_moments[idxs] for idxs in indices]

    # input = input.reshape(-1, k * 3)  # Flatten the KNN array
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True, padding_value=0)
    input = torch.transpose(input, 1, 2).float()



    output = model(input.to(device))
    # output = output.reshape(-1, k, 3)  # Reshape back to (num_of_points, k, 3)
    val1 = output[:, 0]  # Take the first entry for each point
    val2 = output[:, 1]  # Take the second entry for each point
    plot_shape_color_faces(shape.v, shape.f, val1, val2, k1, k2)
    # plot_shape_color_vertices(shape.v, shape.f, val1, val2, k1, k2)

# Custom Callback to perform forward_with_knn every M epochs
class VisualizerCallback(Callback):
    def __init__(self, radius, sample):
        self.radius = radius
        self.sample = sample

        # calculate the k1,k2 values here because it is expensive to calculate it each time we want to plot
        # Calculate k1 and k2 using igl
        vertices = self.sample.v.astype(np.double)  # Convert vertices to double for igl
        faces = self.sample.f.astype(np.int32)  # Convert faces to int32 for igl



        k1, k2, d1, d2 =  igl.principal_curvature(vertices, faces)

        # Normalize k1 and k2 to [0, 1]
        self.k1 = (k1 - k1.min()) / (k1.max() - k1.min())
        self.k2 = (k2 - k2.min()) / (k2.max() - k2.min())

    # def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     forward_with_knn(pl_module, self.sample, self.radius)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch % 10 == 0:
            forward_with_knn(pl_module, self.sample, self.radius, self.k1, self.k2, pl_module.device)
