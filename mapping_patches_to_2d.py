import pickle
import time

import matplotlib
import numpy as np
import pywavefront
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox
from pywavefront import Wavefront
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import mplcursors

from data.generation import QuadraticMonagePatchPointCloudGenerator
from models.point_transformer_conv.model import PointTransformerConvNet
from utils import compute_edges_from_faces, normalize_points_translation_and_rotation

# matplotlib.use('TkAgg')  # Use Tkinter as the backend; you can try other backends as well
from visualize_pointclouds import visualize_pointclouds, is_connected
import pyvista as pv

matplotlib.use('Qt5Agg')  # Use Tkinter as the backend; you can try other backends as well

def find_outliers(output_points):
    output_points = np.array(output_points)
    mean = np.mean(output_points, axis=0)
    std = np.std(output_points, axis=0)
    outliers = []
    for i in range(len(output_points)):
        if np.any(np.abs(output_points[i] - mean) > 1.5*std):
            outliers.append(i)
    return outliers

def map_patch(model, mesh_name:str="vase-lion100K", v=None):

    # Load the OBJ file
    if v is None:
        scene = pywavefront.Wavefront(mesh_name+".obj", collect_faces=True)
        v = np.array(scene.vertices)
    # f = np.array(scene.mesh_list[0].faces)
    center_point_indice = np.argmin(v[:,0]**2+v[:,1]**2+v[:,2]**2)
    v = normalize_points_translation_and_rotation(vertices=v, center_point=v[center_point_indice])
    output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=knn_graph(torch.tensor(v), k=12, batch=None, loop=False), global_pooling=True))
    return output


def add_patch_manually(model, ax, color='Brown', a=None, b=None, ratio=None):
    patch_generator = QuadraticMonagePatchPointCloudGenerator(limit=1, grid_size=100, downsample=True,
                                                              ratio=ratio)
    patch, k1, k2, point0_0 = patch_generator.generate(k1=a, k2=b, patch_type='NaN')
    output = map_patch(model, v=patch.v).detach().numpy()
    output = output.squeeze()
    # if color is integer
    # create color from a and b
    # ax.scatter(output[0], output[1], c=color, cmap='coolwarm', s=12, zorder=2)

    ax.scatter(a, b, c='Indigo', s=12, zorder=2)
    # print("the patch in Brown is connected: " + str(is_connected(patch.v)))
    return output


def map_patches_to_2d():
    # model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_128_non_uniform_samples_normalize-epoch=08.ckpt"
    # model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_128_train_non_uniform_samples_also_with_planar_patches-epoch=149.ckpt"
    # model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_512_non_uniform_samples_random_rotations-epoch=92.ckpt"
    model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_512_non_uniform_samples_random_rotations_just_cont_loss-epoch=206.ckpt"

    model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
    model.eval()
    meshes_dir = "./mesh_different_sampling/non_uniform/grid_size_100/"
    # meshes_names = ["peak300", "saddle300", "peak2300", "saddle2300",  "peak3300", "saddle3300", "peak4300", "saddle4300"]
    meshes_names = ["peak1", "peak2", "peak3", "peak4", "saddle1", "saddle2", "saddle3",  "saddle4", "parabolic1"]
    # 2d array that keep output points and color as a tuple

    output_points_and_color = []
    all_output_points = []


    num_samples = 0
    # Define a list of color strings corresponding to each array
    colors = ['blue', 'red', 'green', 'yellow', 'black', 'purple', 'orange', 'pink', 'gray']


    labels = ['Eliptical Point: x ** 2 + y ** 2',
              'Eliptical2 Point: 1.5 * x ** 2 + 0.2 * y ** 2',
                'Eliptical3 Point: 0.5 * x ** 2 + 0.5 * y ** 2',
                'Eliptical4 Point: 1.5 * x ** 2 + 1.5 * y ** 2',
                'Hyperbolic Point: 0.2 * x ** 2 - 0.2 * y ** 2',
                'Hyperbolic2 Point: 1.5 * x ** 2 - 0.5 * y ** 2',
                'Hyperbolic3 Point: x ** 2 - y ** 2',
                'Hyperbolic4 Point: 1.5 * x ** 2 - 1.5 * y ** 2',
                'Parabolic Point: 0.5 * x ** 2'
              ]

    # Iterate over arrays and colors simultaneously
    if num_samples != 0:
        for color, mesh_name, label in zip(colors, meshes_names, labels):
            output_points = []
            for i in range(num_samples):
                output_points.append(map_patch(model, meshes_dir + mesh_name + "_" + str(i)).detach().numpy())

            output_points = np.array(output_points)
            output_points = output_points.reshape(-1, 2)
            all_output_points.append(output_points)
            # Append array and color as tuple to the list
            output_points_and_color.append((output_points, color, label))
        all_output_points = np.array(all_output_points)



    # outliers = find_outliers(output_points)
    # outliers2 = find_outliers(output_points2)


    # boolean array of false values in length of all_output_points
    is_draw_already = np.zeros(len(all_output_points)*num_samples, dtype=bool)
    pointcloud_color_tuples_to_show = []
    new_patch_index = 1

    fig, ax = plt.subplots()
    ax_bbox = ax.bbox
    def get_closest_point(x, y):
        min_dist = 1000000
        min_point = -1
        clicked_point = np.array([x, y])
        for i in range(len(all_output_points)):
            for j in range(len(all_output_points[i])):
                dist = np.linalg.norm(clicked_point - all_output_points[i][j])
                if dist < min_dist:
                    min_dist = dist
                    min_point = i * num_samples + j
        return min_point


    def on_click(event):
        # if the click is not in the plot
        if event.x < ax_bbox.x0 or event.x > ax_bbox.x1 or event.y < ax_bbox.y0 or event.y > ax_bbox.y1:
            return
        xmouse, ymouse = event.xdata, event.ydata
        # x, y = artist.get_xdata(), artist.get_ydata()
        closest_point_indice = get_closest_point(xmouse, ymouse)

        patch_num = closest_point_indice // num_samples
        i = closest_point_indice % num_samples
        closest_point = all_output_points[patch_num][i]
        color = output_points_and_color[patch_num][1]
        mesh_path = meshes_dir + meshes_names[patch_num] + "_" + str(i) + ".obj"
        mesh = Wavefront(mesh_path)

        vertices = mesh.vertices
        if is_draw_already[closest_point_indice]:
            is_draw_already[closest_point_indice] = False
            # undraw the closest_point
            ax.scatter(closest_point[0], closest_point[1], c=color, s=12, zorder=2)
            plt.gca().set_aspect('equal', adjustable='box')

            plt.draw()
            plt.pause(0.001)
            # remove vertices from plot
            pointcloud_color_tuples_to_show.remove((vertices, color))
            return


        ax.scatter(closest_point[0], closest_point[1], c='cyan', s=10, zorder=2)
        plt.draw()
        plt.pause(0.001)
        is_draw_already[closest_point_indice] = True
        pointcloud_color_tuples_to_show.append((vertices, color))



    def show_pointclouds(event):
        visualize_pointclouds(*pointcloud_color_tuples_to_show, check_connected=True)

    def add_patch(event):
        global new_patch_index
        # Function to be called when the button is clicked
        a = float(text_box.text)
        b = float(text_box2.text)
        ratio = float(text_box3.text)
        # Do something with the entered value
        print(f"Button clicked with value {a}, {b} with ratio {ratio}")
        patch_generator = QuadraticMonagePatchPointCloudGenerator(limit=1, grid_size=100, downsample=True, ratio=ratio)
        patch, k1, k2, point0_0 = patch_generator.generate(k1=a, k2=b, patch_type='NaN')
        output = map_patch(model, v=patch.v).detach().numpy()
        output = output.squeeze()
        ax.scatter(output[0], output[1], c='brown', s=12, zorder=1, label='New Patch')
        ax.scatter(a, b, c='Indigo', s=12, zorder=2, label='Patch coefficients')
        plt.draw()
        plt.pause(0.001)
        print("the patch in Brown is connected: " + str(is_connected(patch.v)))


        # new_patch_index += 1

    def clear_points(event):
        for i in range(len(is_draw_already)):
            if is_draw_already[i]:
                is_draw_already[i] = False
                patch_num = i // num_samples
                sample_num = i % num_samples
                curr_point = all_output_points[patch_num][sample_num]
                color = output_points_and_color[patch_num][1]
                ax.scatter(curr_point[0], curr_point[1], c=color, s=12, zorder=2)
                plt.draw()
                plt.pause(0.001)
        pointcloud_color_tuples_to_show.clear()




    tolerance = 10  # points
    for i in range(len(output_points_and_color)):
        output_points = output_points_and_color[i][0]
        color = output_points_and_color[i][1]
        label = output_points_and_color[i][2]
        ax.scatter(output_points[:, 0], output_points[:, 1], c=color, label=label, picker=tolerance, zorder=1)

    colors = np.linspace(0, 1, 100)
    delta = np.linspace(0, 1, 100)

    outputs_manually = []
    for i in range(100):
        # Create a scatter plot using a colormap (e.g., 'viridis')
        outputs_manually.append(add_patch_manually(model=model, ax=ax, color=colors[i], a=1.5-delta[i], b=1.5-delta[i], ratio=0.05))

    outputs_manually = np.array(outputs_manually)
    ax.scatter(outputs_manually[:, 0], outputs_manually[:, 1], c=colors, cmap='coolwarm', s=12, zorder=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0))

    fig.canvas.callbacks.connect('button_press_event', on_click)

    button_ax = plt.axes([0.88, 0.05, 0.1, 0.075])  # [left, bottom, width, height]
    button = Button(button_ax, 'Show Pointclouds')
    button.on_clicked(show_pointclouds)
    button2_ax = plt.axes([0.88, 0.3, 0.1, 0.075])  # [left, bottom, width, height]
    button2 = Button(button2_ax, 'Clear Points')
    button2.on_clicked(clear_points)
    button3_ax = plt.axes([0.88, 0.175, 0.1, 0.075])  # [left, bottom, width, height]
    button3 = Button(button3_ax, 'Add Patch')
    button3.on_clicked(add_patch)
    # Define the position and size of the input box
    text_box_ax = plt.axes([0.05, 0.07, 0.1, 0.05])
    text_box = TextBox(text_box_ax, 'Enter value for a:')
    # another textbox
    text_box2_ax = plt.axes([0.05, 0.15, 0.1, 0.05])
    text_box2 = TextBox(text_box2_ax, 'Enter value for b:')
    # ratio textbox
    text_box3_ax = plt.axes([0.05, 0.23, 0.1, 0.05])
    text_box3 = TextBox(text_box3_ax, 'Enter value for ratio:', initial='0.05')

    plt.show()

    # plt.pause(0.001)
    # input("Press Enter to close the plot...")


map_patches_to_2d()