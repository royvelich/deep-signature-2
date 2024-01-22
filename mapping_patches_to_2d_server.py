import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from tqdm import tqdm

from data.non_uniform_sampling import non_uniform_2d_sampling
from manual_calc_principal_curvatures import compute_principal_curvatures_using_knn_and_polyfit
from models.point_transformer_conv.model import PointTransformerConvNet
from utils import  normalize_points_translation_and_rotation
import igl

# matplotlib.use('Qt5Agg')  # Use Tkinter as the backend; you can try other backends as well

def find_outliers(output_points):
    output_points = np.array(output_points)
    mean = np.mean(output_points, axis=0)
    std = np.std(output_points, axis=0)
    outliers = []
    for i in range(len(output_points)):
        if np.any(np.abs(output_points[i] - mean) > 1.5*std):
            outliers.append(i)
    return outliers

def map_patch_using_model(model, v):

    center_point_indice = np.argmin(v[:,0]**2+v[:,1]**2+v[:,2]**2)
    v = normalize_points_translation_and_rotation(vertices=v, center_point=v[center_point_indice])
    v = torch.tensor(v, dtype=torch.float32).to(model.device)
    output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=knn_graph(torch.tensor(v), k=12, batch=None, loop=False), global_pooling=True))
    return output

def map_patch_using_surface_fitting(v):
    # center_point_indice = np.argmin(v[:,0]**2+v[:,1]**2+v[:,2]**2)
    # v = normalize_points_translation_and_rotation(vertices=v, center_point=v[center_point_indice])
    output = compute_principal_curvatures_using_knn_and_polyfit(v)
    return output



def map_patches_to_2d():

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        dataset_eliptical_path = "./data/spherical_monge_patches_100_N_10000.pkl"
        dataset_hyperbolic_path = "./data/hyperbolic_monge_patches_100_N_10000.pkl"
        dataset_parabolic_path = "./data/parabolic_monge_patches_100_N_10000.pkl"
        model_path = "./checkpoints/model_point_transformer_1_layers_width_512_non_uniform_samples_random_rotations-epoch=92.ckpt"
        model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cuda:0'))

    else:
        dataset_eliptical_path = "data/spherical_monge_patches_100_N_10.pkl"
        dataset_hyperbolic_path = "data/hyperbolic_monge_patches_100_N_10.pkl"
        dataset_parabolic_path = "data/parabolic_monge_patches_100_N_10.pkl"
        model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_512_non_uniform_samples_random_rotations-epoch=92.ckpt"
        model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))

    model.eval()

    # Define a list of color strings corresponding to each array
    colors = ['blue', 'red', 'green']

    labels = ['Eliptical Points',
              'Hyperbolic Points',
              'Parabolic Points'
              ]

    output_points_eliptical = []
    output_points_hyperbolic = []
    # output_points_parabolic = []



    # Load the triplets from the file
    with open(dataset_eliptical_path, 'rb') as f:
        f.seek(0)  # Move the file pointer to the beginning of the file
        data_spherical = pickle.load(f)
    with open(dataset_hyperbolic_path, 'rb') as f:
        f.seek(0)
        data_hyperbolic = pickle.load(f)
    # with open(dataset_parabolic_path, 'rb') as f:
    #     f.seek(0)
    #     data_parabolic = pickle.load(f)

    # assuming same length of all 3 datasets
    for i in tqdm(range(len(data_spherical))):
        curr_eliptical_patch_indices = non_uniform_2d_sampling(grid_size=100, ratio=0.05)
        curr_hyperbolic_patch_indices = non_uniform_2d_sampling(grid_size=100, ratio=0.05)
        curr_parabolic_patch_indices = non_uniform_2d_sampling(grid_size=100, ratio=0.05)
        curr_eliptical_points = data_spherical[i].v[curr_eliptical_patch_indices]
        curr_hyperbolic_points = data_hyperbolic[i].v[curr_hyperbolic_patch_indices]
        # curr_parabolic_points = data_parabolic[i].v[curr_parabolic_patch_indices]

        # output_points_eliptical.append(map_patch_using_model(model, curr_eliptical_points).cpu().detach().numpy())
        # output_points_hyperbolic.append(map_patch_using_model(model, curr_hyperbolic_points).cpu().detach().numpy())
        # output_points_parabolic.append(map_patch_using_model(model, curr_parabolic_points).cpu().detach().numpy())

        output_points_eliptical.append(map_patch_using_surface_fitting(curr_eliptical_points))
        output_points_hyperbolic.append(map_patch_using_surface_fitting(curr_hyperbolic_points))


    output_points_eliptical = np.array(output_points_eliptical).squeeze()
    output_points_hyperbolic = np.array(output_points_hyperbolic).squeeze()
    # output_points_parabolic = np.array(output_points_parabolic).squeeze()
    # save outputs in files
    with open('output_points_eliptical.pkl', 'wb') as f:
        pickle.dump(output_points_eliptical, f)
    with open('output_points_hyperbolic.pkl', 'wb') as f:
        pickle.dump(output_points_hyperbolic, f)


    # Plot the points
    plt.scatter(output_points_eliptical[:, 0], output_points_eliptical[:, 1], c=colors[0], label=labels[0], s=5)
    plt.scatter(output_points_hyperbolic[:, 0], output_points_hyperbolic[:, 1], c=colors[1], label=labels[1], s=5)
    # plt.scatter(output_points_parabolic[:, 0], output_points_parabolic[:, 1], c=colors[2], label=labels[2], s=1)


    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0))
    plt.savefig('map_patches_to_2d.png', dpi=300)
    # save fig for only  1 output
    # Create a figure for the eliptical plot
    fig_eliptical, ax_eliptical = plt.subplots()

    # Plot the eliptical points
    ax_eliptical.scatter(output_points_eliptical[:, 0], output_points_eliptical[:, 1], c=colors[0], label=labels[0],
                         s=5)

    # Set aspect ratio and add legend
    ax_eliptical.set_aspect('equal', adjustable='box')
    ax_eliptical.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0))

    # Save the eliptical plot
    fig_eliptical.savefig('map_patches_to_2d_eliptical.png', dpi=300)

    # Create a figure for the hyperbolic plot
    fig_hyperbolic, ax_hyperbolic = plt.subplots()

    # Plot the hyperbolic points
    ax_hyperbolic.scatter(output_points_hyperbolic[:, 0], output_points_hyperbolic[:, 1], c=colors[1], label=labels[1],
                          s=5)

    # Set aspect ratio and add legend
    ax_hyperbolic.set_aspect('equal', adjustable='box')
    ax_hyperbolic.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0))

    # Save the hyperbolic plot
    fig_hyperbolic.savefig('map_patches_to_2d_hyperbolic.png', dpi=300)

    # plt.show()


map_patches_to_2d()