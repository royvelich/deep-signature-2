import pickle
import threading
import time

import networkx as nx
import numpy as np
import pyvista as pv
import torch
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# from utils import normalize_point_cloud, normalize_points_translation_and_rotation
from torch_geometric.nn import knn_graph


def thread_plot_patch(plotter):
    while True:
        plotter.show()
        time.sleep(0.1)

def is_connected(pointcloud, k=12):
    edge_indices = knn_graph(torch.tensor(pointcloud), k=k, batch=None, loop=False)
    edge_indices = edge_indices.view(-1, 2).numpy()

    # Create a graph from edge indices
    G = nx.Graph()
    G.add_edges_from(edge_indices)

    # Check if the graph is connected
    return nx.is_connected(G)

def visualize_pointclouds(*pointcloud_color_tuples, check_connected=False):
    pointcloud_color_tuples = [
        (pc.detach().cpu().numpy(), color) if isinstance(pc, torch.Tensor) else (pc, color)
        for pc, color in pointcloud_color_tuples
    ]

    plotter = pv.Plotter()


    for pc, color in pointcloud_color_tuples:
        if pc is not None:
            cloud = pv.PolyData(pc)
            plotter.add_points(cloud, color=color, render_points_as_spheres=True, point_size=5)
            if check_connected:
                print("the patch in "+str(color)+ " is connected: "+str(is_connected(pc)))
                # Get edge indices using the knn_function
                # edge_indices = knn_graph(torch.tensor(pc), k=6, batch=None, loop=False)
                # Reshape edge_indices to (N, 2) for add_cells
                # edge_indices = edge_indices.view(-1, 2).numpy()

                # Create lines from edge indices
                # lines = pv.PolyData()
                # lines.points = pc
                # lines.lines = edge_indices
                #
                # # Add edges to the plot
                # plotter.add_mesh(lines, color="black", line_width=1.0)
    plotter.add_axes_at_origin()

    # Show the plot
    #
    # thread = threading.Thread(target=thread_plot_patch, args=(plotter,))
    # thread.start()
    plotter.show()

def visualize_pointclouds2(pointcloud1, pointcloud2=None, pointcloud3=None, pointcloud4=None, labels=None, title=None, vector_fields_to_visualize=None, fps_indices=None, arrow_scale=0.001):
        # from torch to numpy
        # if is torch
        if isinstance(pointcloud1, torch.Tensor):
            pointcloud1 = pointcloud1.detach().cpu().numpy()
        if pointcloud2 is not None and isinstance(pointcloud2, torch.Tensor):
            pointcloud2 = pointcloud2.detach().cpu().numpy()
        if pointcloud3 is not None and isinstance(pointcloud3, torch.Tensor):
            pointcloud3 = pointcloud3.detach().cpu().numpy()
        if pointcloud4 is not None and isinstance(pointcloud4, torch.Tensor):
            pointcloud4 = pointcloud4.detach().cpu().numpy()


        cloud1 = pv.PolyData(pointcloud1)

        plotter = pv.Plotter()
        plotter.add_points(cloud1, color="blue", render_points_as_spheres=True, point_size=5)
        if pointcloud2 is not None:
            cloud2 = pv.PolyData(pointcloud2)
            plotter.add_points(cloud2, color="red", render_points_as_spheres=True, point_size=5)
        if pointcloud3 is not None:
            cloud3 = pv.PolyData(pointcloud3)
            plotter.add_points(cloud3, color="green", render_points_as_spheres=True, point_size=5)
        if pointcloud4 is not None:
            cloud4 = pv.PolyData(pointcloud4)
            plotter.add_points(cloud4, color="yellow", render_points_as_spheres=True, point_size=5)
        if vector_fields_to_visualize is not None:
            colors = ["red","blue", "green", "yellow"]
            for i in range(len(vector_fields_to_visualize)):
                plotter.add_arrows(cent=pointcloud1[fps_indices], direction=vector_fields_to_visualize[i], color=colors[i], mag=arrow_scale)


        if title:
            plotter.set_title(title)

        plotter.add_axes_at_origin()

        # Show the plot

        plotter.show()



def apply_pca_and_align(points, eigenvectors=None):
    """ Apply PCA on a set of points and align the principal components with the axes. """
    points_mean = points.mean(axis=0)
    centered_points = points - points_mean
    covariance_matrix = np.cov(centered_points, rowvar=False)
    if eigenvectors is None:
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # sort eigenvalues in increasing order
        # idx = eigenvalues.argsort()[::-1]
        # eigenvectors = eigenvectors[:, idx]
    # normalize eigenvectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    # for i in range(len(eigenvectors)):
    #     if eigenvectors[i,i] < 0:
    #         eigenvectors[:,i] = -eigenvectors[:,i]
    # Align the principal components with the X, Y, Z axes
    aligned_points = np.dot(centered_points, eigenvectors)

    return aligned_points, eigenvectors

def plot_models(models, titles, colors=None):
    """ Plot models in a 3D space. """
    fig = plt.figure(figsize=(18, 6))

    for i, model in enumerate(models):
        ax = fig.add_subplot(1, len(models), i + 1, projection='3d')
        if colors is not None:
            ax.scatter(model[:, 0], model[:, 1], model[:, 2], c=colors[i])
        else:
            ax.scatter(model[:, 0], model[:, 1], model[:, 2])
        ax.set_title(titles[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()


def load_point_cloud(file_path):
    """Load a point cloud from a file."""
    return o3d.io.read_point_cloud(file_path)

def align_point_clouds(source, target):
    """Align two point clouds using the ICP algorithm."""
    threshold = 0.0002  # set this depending on your data
    trans_init = np.asarray([[10, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def random_translation_and_rotation(points):
    """Apply a random translation and rotation to a set of points."""
    # Translation
    translation = np.random.uniform(low=-0.1, high=0.1, size=3)
    points += translation
    # Rotation
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=3)
    rotation_matrix = np.array([[np.cos(rotation[2]) * np.cos(rotation[1]),
                                 np.cos(rotation[2]) * np.sin(rotation[1]) * np.sin(rotation[0]) - np.sin(rotation[2]) * np.cos(rotation[0]),
                                 np.cos(rotation[2]) * np.sin(rotation[1]) * np.cos(rotation[0]) + np.sin(rotation[2]) * np.sin(rotation[0])],
                                [np.sin(rotation[2]) * np.cos(rotation[1]),
                                 np.sin(rotation[2]) * np.sin(rotation[1]) * np.sin(rotation[0]) + np.cos(rotation[2]) * np.cos(rotation[0]),
                                 np.sin(rotation[2]) * np.sin(rotation[1]) * np.cos(rotation[0]) - np.cos(rotation[2]) * np.sin(rotation[0])],
                                [-np.sin(rotation[1]),
                                 np.cos(rotation[1]) * np.sin(rotation[0]),
                                 np.cos(rotation[1]) * np.cos(rotation[0])]])
    points = np.dot(points, rotation_matrix)
    return points




# file_path = "generated_triplet_data/triplets_data_size_50_N_10_all_monge_patch_normalized_pos_and_rot.pkl"
# file_path = "data/spherical_monge_patches_100_N_10.pkl"
# #
# # # Load the triplets from the file
# with open(file_path, 'rb') as f:
#     f.seek(0)  # Move the file pointer to the beginning of the file
#     data = pickle.load(f)
# #
# #
#
# rand_indice = np.random.randint(0, len(data))
# model1 = data[rand_indice].v
# model2 = data[rand_indice].v
#
# model1_center_indice = np.argmin(model1[:, 0] ** 2 + model1[:, 1] ** 2)
# model2_center_indice = np.argmin(model2[:, 0] ** 2 + model2[:, 1] ** 2)
#
# model1 = random_translation_and_rotation(model1)
# model2 = random_translation_and_rotation(model2)
# # # source = o3d.geometry.PointCloud()
# # # target = o3d.geometry.PointCloud()
# # #
# #
# # # source.points = o3d.utility.Vector3dVector(model1)
# # #
# # # target.points = o3d.utility.Vector3dVector(model2)
# # #
# # # # Align the point clouds
# # # transformation = align_point_clouds(source, target)
# # #
# # # # Apply the transformation
# # # source.transform(transformation)
# # #
# # # # Visualize the result
# # # o3d.visualization.draw_geometries([source, target])
# #
# #
# # #
# #
# visualize_pointclouds(model1, model2)
# #
# # Align each model independently
#
# model1_aligned = normalize_points_translation_and_rotation(model1, center_point=model1[model1_center_indice])
# model2_aligned = normalize_points_translation_and_rotation(model2, center_point=model2[model2_center_indice])
#
# visualize_pointclouds(model1_aligned, model2_aligned)
# # # Plot the models
# # plot_models([model1, model2, model1_aligned, model2_aligned],
# #             ['Model 1 Original', 'Model 2 Original', 'Model 1 Aligned', 'Model 2 Aligned'])
# #
# # visualize_pointclouds(model1, model2)
# # visualize_pointclouds(model1_aligned, model2_aligned)
#
# # file_path = "triplets_data_size_50_N_10_all_monge_patch_normalized_pos_and_rot.pkl"
#
# #
# # for i in range(len(data)):
# #     anc_patch = data[i][0]
# #     pos_patch = data[i][1]
# #     visualize_pointclouds(anc_patch.v, pos_patch.v, title="Triplet " + str(i) + " " + str(j))
# #     mid_point = torch.argmin(torch.sqrt((batch[i].x[:, 0]) ** 2 + (batch[i].x[:, 1]) ** 2), dim=0)
# #     anc_mid_point = torch.argmin(torch.sqrt((anc_patch.v[:, 0]) ** 2 + (anc_patch.v[:, 1]) ** 2), dim=0)
# #     pos_mid_point = torch.argmin(torch.sqrt((pos_patch.v[:, 0]) ** 2 + (pos_patch.v[:, 1]) ** 2), dim=0)
# #
# #     pointcloud_after_normalization = normalize_point_cloud