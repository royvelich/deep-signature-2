import matplotlib
import numpy as np
import pywavefront
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data

from models.point_transformer_conv.model import PointTransformerConvNet
from utils import compute_edges_from_faces

matplotlib.use('TkAgg')  # Use Tkinter as the backend; you can try other backends as well

def find_outliers(output_points):
    output_points = np.array(output_points)
    mean = np.mean(output_points, axis=0)
    std = np.std(output_points, axis=0)
    outliers = []
    for i in range(len(output_points)):
        if np.any(np.abs(output_points[i] - mean) > 1.5*std):
            outliers.append(i)
    return outliers

def map_patch(model, mesh_name:str="vase-lion100K"):

    # Load the OBJ file
    scene = pywavefront.Wavefront(mesh_name+".obj", collect_faces=True)
    v = np.array(scene.vertices)
    f = np.array(scene.mesh_list[0].faces)

    output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=compute_edges_from_faces(f)), global_pooling=True)
    return output


def map_patches_to_2d():
    # model_path = "C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2\images\pointtransformer\width128_trained_reg_plus_unreg_patches_k1k2\grid_size_30_reg_patches\model_point_transformer_3_layers_width_128-epoch=29.ckpt"
    model_path = "C:/Users\galyo\Downloads\model_point_transformer_3_layers_width_128_train_non_uniform_samples-epoch=79.ckpt"
    model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
    model.eval()
    const_to_improve_aspect_ratio = 10
    meshes_dir = "./mesh_different_sampling/non_uniform/same_ratio/"
    meshes_names = ["peak300", "saddle300", "peak2300", "saddle2300",  "peak3300", "saddle3300", "peak4300", "saddle4300"]
    output_points = []
    output_points2 = []
    output_points3 = []
    output_points4 = []
    output_points5  = []
    output_points6 = []
    output_points7 = []
    output_points8 = []
    num_samples = 30
    for i in range(num_samples):
        output_points.append(map_patch(model, meshes_dir+meshes_names[0]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points2.append(map_patch(model, meshes_dir+meshes_names[1]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points3.append(map_patch(model, meshes_dir+meshes_names[2]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points4.append(map_patch(model, meshes_dir+meshes_names[3]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points5.append(map_patch(model, meshes_dir+meshes_names[4]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points6.append(map_patch(model, meshes_dir+meshes_names[5]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points7.append(map_patch(model, meshes_dir+meshes_names[6]+str(i)).detach().numpy())
    for i in range(num_samples):
        output_points8.append(map_patch(model, meshes_dir+meshes_names[7]+str(i)).detach().numpy())

    # outliers = find_outliers(output_points)
    # outliers2 = find_outliers(output_points2)
    output_points = np.array(output_points)
    output_points = output_points.reshape(-1, 2)
    output_points2 = np.array(output_points2)
    output_points2 = output_points2.reshape(-1, 2)
    output_points3 = np.array(output_points3)
    output_points3 = output_points3.reshape(-1, 2)
    output_points4 = np.array(output_points4)
    output_points4 = output_points4.reshape(-1, 2)
    output_points5 = np.array(output_points5)
    output_points5 = output_points5.reshape(-1, 2)
    output_points6 = np.array(output_points6)
    output_points6 = output_points6.reshape(-1, 2)
    output_points7 = np.array(output_points7)
    output_points7 = output_points7.reshape(-1, 2)
    output_points8 = np.array(output_points8)
    output_points8 = output_points8.reshape(-1, 2)
    # Plotting for visualization
    # aspect ratio true to get a square figure
    plt.scatter(output_points[:, 0], output_points[:, 1], c='blue', label='Eliptical Point: x ** 2 + y ** 2')
    plt.scatter(output_points2[:, 0], output_points2[:, 1], c='red', label='Hyperbolic Point: 0.6*x_grid ** 2 - 0.6*y_grid ** 2')
    plt.scatter(output_points3[:, 0], output_points3[:, 1], c='green', label='Eliptical2 Point: 7 * x_grid ** 2 + y_grid ** 2')
    plt.scatter(output_points4[:, 0], output_points4[:, 1], c='yellow', label='Hyperbolic2 Point: 0.8*x_grid ** 2 - 2*y_grid ** 2')
    plt.scatter(output_points5[:, 0], output_points5[:, 1], c='black', label='Eliptical3 Point: 0.5 * x_grid ** 2 + 0.5 * y_grid ** 2')
    plt.scatter(output_points6[:, 0], output_points6[:, 1], c='purple', label='Hyperbolic3 Point: 0.1*x_grid ** 2 - 6*y_grid ** 2')
    plt.scatter(output_points7[:, 0], output_points7[:, 1], c='orange', label='Eliptical4 Point: -2 * x_grid ** 2 - y_grid ** 2')
    plt.scatter(output_points8[:, 0], output_points8[:, 1], c='pink', label='Hyperbolic4 Point: 2 * x_grid ** 2 - 2 * y_grid ** 2')
    plt.axis('equal')
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1.0))

    plt.show()




map_patches_to_2d()