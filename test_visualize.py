from datetime import datetime
from pathlib import Path

import igl
import matplotlib
import torch
from matplotlib import pyplot as plt
from pygfx import Mesh, MeshPhongMaterial, Geometry, Color, WorldObject
from skimage.exposure import exposure
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from wgpu.gui.auto import WgpuCanvas, run

# uncomment for just snapshot
# from wgpu.gui.offscreen import WgpuCanvas

import trimesh
import pygfx as gfx
from pygfx import OrbitController
import imageio
import numpy as np
import pywavefront
import seaborn as sns

from models.point_transformer_conv.model import PointTransformerConvNet
from utils import compute_edges_from_faces, compute_patches_from_mesh, save_glb


def _rescale_k(k: np.ndarray) -> np.ndarray:
    # rescale colors so it will range between 0 and 1
    min_val = k.min()
    max_val = k.max()
    rescaled_k = (k - min_val+1e-5) / (max_val - min_val+1e-5)
    return rescaled_k


def _get_vertex_colors_from_k(k: np.ndarray,title: str = '') -> np.ndarray:
    # plot_color_histogram(k, title=title+' histogram')

    k = _rescale_k(k=k)
    k_one_minus = 1 - k
    # plot_color_histogram(k, title=title+' histogram')

    # convex combinations between red and blue colors, based on the predicted gaussian curvature
    # blue big values for k
    # green small values for k
    c1 = np.column_stack((np.zeros_like(k), k_one_minus, np.zeros_like(k), np.ones_like(k)))
    c2 = np.column_stack((np.zeros_like(k), np.zeros_like(k), k, np.ones_like(k)))
    c = c1 + c2

    return c

def _get_vertex_colors_from_2_vals(output0: np.ndarray, output1: np.ndarray, title: str = '') -> np.ndarray:
    # Plot histograms
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.hist(output0, bins=50, color='blue', alpha=0.7)
    # plt.title('Histogram of Output0')
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    #
    # plt.show()
    #
    # Perform histogram equalization
    output0 = exposure.equalize_hist(output0)
    output1 = exposure.equalize_hist(output1)


    # # plot_color_histogram(k, title=title+' histogram')
    # output0 = _rescale_k(k=output0)
    # output1 = _rescale_k(k=output1)


    # convex combinations between red and blue colors, based on the predicted gaussian curvature
    # blue big values for k
    # green small values for k
    c1 = np.column_stack((output0, np.zeros_like(output0), np.zeros_like(output0), np.ones_like(output0)))
    c2 = np.column_stack((np.zeros_like(output1), np.zeros_like(output1), output1, np.ones_like(output1)))
    c = c1 + c2

    return c

def _get_vertex_colors_from_1_val(output: np.ndarray, title: str = '') -> np.ndarray:

    # Perform histogram equalization
    output = exposure.equalize_hist(output)


    # # plot_color_histogram(k, title=title+' histogram')
    # output0 = _rescale_k(k=output0)
    # output1 = _rescale_k(k=output1)


    # convex combinations between red and blue colors, based on the predicted gaussian curvature
    # blue big values for k
    # green small values for k
    output_minus_1 = 1 - output
    c1 = np.column_stack((output, np.zeros_like(output), np.zeros_like(output), np.ones_like(output)))
    c2 = np.column_stack((np.zeros_like(output), np.zeros_like(output), output_minus_1, np.ones_like(output)))
    c = c1 + c2

    return c



def _create_world_object_for_mesh(faces, vertices, k: np.ndarray, title='title', color: Color = '#ffffff') -> WorldObject:
    c = _get_vertex_colors_from_k(k=k, title=title)
    geometry = Geometry(
        indices=np.ascontiguousarray(faces.astype(np.int32)),
        positions=np.ascontiguousarray(vertices.astype(np.float32)),
        colors=np.ascontiguousarray(c.astype(np.float32)))

    material = MeshPhongMaterial(
        color=color,
        vertex_colors=True)

    mesh = Mesh(
        geometry=geometry,
        material=material)
    return mesh



def add_colored_mesh(scene,  faces, vertices, colors,position=(0, 0, 0), title='title'):

    mesh = _create_world_object_for_mesh(faces, vertices, colors, title)
    mesh.local.position = position
    # disable specular highlights
    mesh.material.specular = 0
    scene.add(mesh)
    text = gfx.Text(
        gfx.TextGeometry(
            markdown="**"+title+"**",
            screen_space=True,
            font_size=20,
            # anchor="bottomleft",
        ),
        gfx.TextMaterial(color="#0f4"),
    )
    text.local.position = mesh.local.position + (0, 0.7, 0)
    mesh.add(text)



def xyz_second_moments(vertices):
    vertices = torch.tensor(vertices, dtype=torch.float32)
    xx = vertices[:, 0] ** 2
    yy = vertices[:, 1] ** 2
    zz = vertices[:, 2] ** 2
    xy = vertices[:, 0] * vertices[:, 1]
    xz = vertices[:, 0] * vertices[:, 2]
    yz = vertices[:, 1] * vertices[:, 2]
    # stack using torch
    return torch.stack([vertices[:, 0], vertices[:, 1], vertices[:, 2], xx, yy, zz, xy, xz, yz], dim=1)

# vis_obj = "patch"
# dataset_reg_and_unreg = True
#
# if vis_obj == "patch":
#     # file_path = "./triplets_size_30_N_100_all_monge_patch_normalized_pos_and_rot_80_per_fps_sampling_reg_and_unreg.pkl"
#     file_path = "./triplets_size_50_N_1_all_torus_normalized_pos_and_rot_80_per_fps_sampling_reg_and_unreg.pkl"
#
#     # Load the triplets from the file
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#
#     sample = data[0][1]
#     v = sample.v
#     f = sample.f
#     second_moments = sample.v_second_moments
#     dis = 2 # distance between the rendered patches in the visualization
# else:
#     # Load the OBJ file
#     scene = pywavefront.Wavefront("modified_mesh.obj", collect_faces=True)
#
#     # Get the vertices and faces
#     vertices = np.array(scene.vertices)
#     faces = np.array(scene.mesh_list[0].faces)
#
#     # Now vertices and faces contain the data from the OBJ file
#     sample = {'v': vertices, 'f': faces, "second_moments": xyz_second_moments(vertices)}
#     v = vertices
#     f = faces
#     second_moments = sample["second_moments"]
#     dis = 20
#
# # patch_generator = QuadraticMonagePatchGenerator2(limit=1.0, grid_size=100)
# # sample, k1, k2, point_0_0_index = patch_generator.generate(k1=0.5, k2=-0.5,downsample=False)
#
#
# model_path = "C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2\images\pointtransformer\width128_trained_reg_plus_unreg_patches_k1k2\grid_size_30_reg_patches\model_point_transformer_3_layers_width_128-epoch=29.ckpt"
# model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_128_train_non_uniform_samples_also_with_planar_patches-epoch=149.ckpt"
model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_512_non_uniform_samples_random_rotations-epoch=92.ckpt"

model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
model.eval()
#
# sample_patch = sample
# sample_patch = trimesh.Trimesh(vertices=v, faces=f)
#
# # d1, d2, k1, k2 = igl.principal_curvature(v, f)
# k1,k2 = np.array(0),np.array(0)
#
# canvas = WgpuCanvas(size=(900, 400))
# renderer = gfx.renderers.WgpuRenderer(canvas)
# scene = gfx.Scene()
# camera = gfx.PerspectiveCamera(70, 16 / 9)
# # camera.show_object(scene)
#
# # output = model(Data(x=second_moments.to(torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=compute_edges_from_faces(f)), global_pooling=False)
# output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=compute_edges_from_faces(f)), global_pooling=False)
# output = output.detach().numpy()
#
# add_colored_mesh(scene, faces=sample_patch.faces, vertices=sample_patch.vertices, colors=output[:,0],position=(0,0,0),title='output0')
# add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=output[:,1],position=(dis,0,0),title='output1')
# add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors = k1, position=(0,dis,0),title='k1')
# add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors = k2, position=(dis,dis,0),title='k2')
#
#
# scene.add(gfx.AmbientLight(1, 0.2))
#
# light = gfx.DirectionalLight(1, 2)
# light.local.position = (0, 0, 1)
# scene.add(light)
# dark_gray = np.array((169, 167, 168, 255)) / 255
# light_gray = np.array((100, 100, 100, 255)) / 255
# background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
# scene.add(background)




# Rotate the object a bit

# rot = la.quat_from_euler((0.71, 0.1), order="XY")
# for obj in scene.children:
#     obj.local.rotation = la.quat_mul(rot, obj.local.rotation)

# def on_move(event):
#     controls.handle_event(event)
# canvas.connect("mouse-move", on_move)
# canvas.connect("mouse-wheel", on_move)

image_num = 0

def on_key_down(event,camera,scene,renderer):
    global state, image_num
    controls = OrbitController(camera)
    controls.zoom_speed = 0.001
    controls.distance = 0.01
    controls.register_events(renderer)
    if event.key == "s":
        state = camera.get_state()
    elif event.key == "l":
        camera.set_state(state)
    elif event.key == "r":
        camera.show_object(scene)
    elif event.key == "q":
        image = renderer.snapshot()
        # Save the image as a PNG file
        date_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        imageio.imwrite('screenshot_'+'_'+str(date_now)+'.png', image)
        image_num += 1

# renderer.add_event_handler(on_key_down, "key_down")
#
# canvas.request_draw(lambda: renderer.render(scene, camera))

def snapshot(scene, camera, renderer, canvas):
    vis_obj = 'output0'
    renderer.render(scene, camera)
    # canvas.request_draw()
    camera.show_object(scene)
    camera.local.position = camera.local.position + (0, -2.0, -5.0)
    canvas.draw()
    # canvas._get_event_wait_time()
    # wait for render to complete
    image = renderer.snapshot()
    # Save the image as a PNG file
    date_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    imageio.imwrite('C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2\images\pointtransformer\width128_trained_reg_plus_unreg_patches_k1k2\snapshot_'+vis_obj+'_'+str(date_now)+'.png', image)


def animate(scene, camera, canvas, renderer):

    renderer.render(scene, camera)
    canvas.request_draw()


def snapshot_multiple_patches(data, dataset_reg_and_unreg=False):
    dis = 1
    for i in range(len(data)):
        index = np.random.randint(0, 3)
        sample = data[i][index]
        v = sample.v
        f = sample.f
        # second_moments = sample.v_second_moments
        # sample_patch = sample
        sample_patch = trimesh.Trimesh(vertices=v, faces=f)

        # d1, d2, k1, k2 = igl.principal_curvature(v, f)
        k1, k2 = np.array(0), np.array(0)

        canvas = WgpuCanvas(size=(900, 400))
        renderer = gfx.renderers.WgpuRenderer(canvas)
        scene = gfx.Scene()
        camera = gfx.PerspectiveCamera(70, 16 / 9)
        # camera.show_object(scene)

        # output = model(Data(x=second_moments.to(torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=compute_edges_from_faces(f)), global_pooling=False)
        output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),
                            edge_index=compute_edges_from_faces(f)), global_pooling=False)
        output = output.detach().numpy()

        add_colored_mesh(scene, faces=sample_patch.faces, vertices=sample_patch.vertices, colors=output[:, 0],
                         position=(0, 0, 0), title='output0')
        add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=output[:, 1], position=(dis, 0, 0),
                         title='output1')
        add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1, position=(0, dis, 0), title='k1')
        add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k2, position=(dis, dis, 0),
                         title='k2')
        if dataset_reg_and_unreg:
            sample = data[i][index+3] # take the reg sample
            v = sample.v
            f = sample.f
            # second_moments = sample.v_second_moments
            # sample_patch = sample
            sample_patch = trimesh.Trimesh(vertices=v, faces=f)
            # d1, d2, k1, k2 = igl.principal_curvature(v, f)

            add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1, position=(0, 2*dis, 0),
                             title='k1_reg')
            add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k2, position=(dis, 2*dis, 0),
                             title='k2_reg')


        scene.add(gfx.AmbientLight(1, 0.2))

        light = gfx.DirectionalLight(1, 2)
        light.local.position = (0, 0, 1)
        scene.add(light)
        dark_gray = np.array((169, 167, 168, 255)) / 255
        light_gray = np.array((100, 100, 100, 255)) / 255
        background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
        scene.add(background)
        canvas.request_draw(lambda: renderer.render(scene, camera))
        snapshot(camera=camera, scene=scene, renderer=renderer, canvas=canvas)
        del canvas
        del renderer
        del scene
        del camera

def snapshot_or_animate_torus(animate=True):
    dis = 5 # distance between the rendered patches in the visualization

    geometry = gfx.torus_knot_geometry(1.0, 0.3, 32, 8, p=1, q=0)
    # Load the OBJ file
    # scene = pywavefront.Wavefront("modified_mesh.obj", collect_faces=True)
    #
    # # Get the vertices and faces
    # v = np.array(scene.vertices)
    # f = np.array(scene.mesh_list[0].faces)
    v = geometry.positions.data.astype(np.float32)
    # need to choose for each neighborhood the faces that contain it vertices and organize them in the right order
    f = geometry.indices.data.astype(np.int32)
    normalized_patches, faces = compute_patches_from_mesh(v, f, k=15)

    # second_moments = sample.v_second_moments
    # sample_patch = sample
    sample_patch = trimesh.Trimesh(vertices=v, faces=f)

    d1, d2, k1, k2 = igl.principal_curvature(v, f)

    canvas = WgpuCanvas(size=(900, 400))
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, 16 / 9)
    # camera.show_object(scene)

    # output = model(Data(x=second_moments.to(torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=compute_edges_from_faces(f)), global_pooling=False)

    output = []
    for i in range(len(normalized_patches)):
        output.append(model(Data(x=torch.tensor(normalized_patches[i], dtype=torch.float32), pos=torch.tensor(normalized_patches[i], dtype=torch.float32),
                            edge_index=compute_edges_from_faces(faces[i])), global_pooling=True).detach().numpy())
    output = np.concatenate(output, axis=0)

    # output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),
    #                     edge_index=compute_edges_from_faces(f)), global_pooling=False)
    # output = output.detach().numpy()

    add_colored_mesh(scene, faces=sample_patch.faces, vertices=sample_patch.vertices, colors=output[:, 0],
                     position=(0, 0, 0), title='output0')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=output[:, 1], position=(dis, 0, 0),
                     title='output1')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=output[:,0]*output[:, 1], position=(2*dis, 0, 0),
                     title='output0*output1')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1, position=(0, dis, 0), title='k1')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k2, position=(dis, dis, 0),
                     title='k2')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1*k2, position=(2*dis, dis, 0),
                     title='k1*k2')

    scene.add(gfx.AmbientLight(1, 0.2))

    light = gfx.DirectionalLight(1, 2)
    light.local.position = (0, 0, 1)
    scene.add(light)
    dark_gray = np.array((169, 167, 168, 255)) / 255
    light_gray = np.array((100, 100, 100, 255)) / 255
    background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
    scene.add(background)
    canvas.request_draw(lambda: renderer.render(scene, camera))


    def on_key_down2(event):
        global state, image_num
        controls = OrbitController(camera)
        controls.zoom_speed = 0.001
        controls.distance = 0.01
        controls.register_events(renderer)
        if event.key == "s":
            state = camera.get_state()
        elif event.key == "l":
            camera.set_state(state)
        elif event.key == "r":
            camera.show_object(scene)
        elif event.key == "q":
            image = renderer.snapshot()
            # Save the image as a PNG file
            date_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            imageio.imwrite('screenshot_' + '_' + str(date_now) + '.png', image)
            image_num += 1

    renderer.add_event_handler(on_key_down2, "key_down")
    if not animate:
        snapshot(camera=camera, scene=scene, renderer=renderer, canvas=canvas)
    else:
        canvas.request_draw(animate(scene=scene, camera=camera, renderer=renderer, canvas=canvas))
        run()

def plot_color_histogram(data, title):
    bins = np.linspace(min(data), max(data), 100)
    matplotlib.use('TkAgg')  # You can choose a suitable backend

    # Create the histogram
    sns.histplot(data, bins=bins, kde=True, color='blue', edgecolor='k', stat='count')

    # Add labels and a title
    plt.xlabel('Value Ranges')
    plt.ylabel('Count')
    plt.title(title)

    # Show the plot
    plt.show()

def calculate_correlation(data1, data2):
    output = np.corrcoef(data1, data2)
    plt.plot(data1, label='Array 1')
    plt.plot(data2, label='Array 2')

    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Two Arrays')
    plt.show()

    return output

def animate_mesh(mesh_name:str="vase-lion100K.obj"):
    dis = 100  # distance between the rendered patches in the visualization

    # geometry = gfx.torus_knot_geometry(1.0, 0.3, 32, 8, p=1, q=0)
    # v = geometry.positions.data.astype(np.float32)
    # f = geometry.indices.data.astype(np.int32)

    # Load the OBJ file
    # need to be obj file
    scene = pywavefront.Wavefront(mesh_name, collect_faces=True)
    v = np.array(scene.vertices)
    f = np.array(scene.mesh_list[0].faces)

    # edges1 = compute_edges_from_faces(f)
    # edges2 = compute_edges_from_faces2(torch_geometric.data.Data(pos=torch.tensor(v, dtype=torch.float32), face=torch.tensor(f.T, dtype=torch.int32)))
    # tripy_faces = tripy.earclip(edges2)

    # downsample the mesh
    ratio = 0.1
    # indices = fps(x=torch.tensor(data=v), ratio=ratio)
    m, v_downsampled, f_downsampled, _, _ = igl.decimate(v, f, int(f.shape[0]*ratio))
    # v_downsampled, f_downsampled = v, f
    # f_downsampled = []
    # for i in range(len(f)):
    #    if np.sum(np.isin(f[i], indices)) == 3:
    #        face = [np.where(f[i][0] == indices)[0][0], np.where(f[i][1] == indices)[0][0], np.where(f[i][2] == indices)[0][0]]
    #        f_downsampled.append(face)


    # need to choose for each neighborhood the faces that contain it vertices and organize them in the right order
    # normalized_   patches, faces = compute_patches_from_mesh(v, f, k=30)
    patches_size = 200
    GT_radius_size = 30
    is_radius = False

    normalized_patches, faces = compute_patches_from_mesh(v_downsampled, f_downsampled, k=patches_size, is_radius=is_radius)

    # second_moments = sample.v_second_moments
    # sample_patch = sample 
    sample_patch = trimesh.Trimesh(vertices=v, faces=f)
    sample_patch_downsampled = trimesh.Trimesh(vertices=v_downsampled, faces=f_downsampled)

    d1, d2, k1, k2 = igl.principal_curvature(v, f, radius=GT_radius_size)

    canvas = WgpuCanvas(size=(900, 400))
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, 16 / 9)
    # camera.show_object(scene)

    # output = model(Data(x=second_moments.to(torch.float32), pos=torch.tensor(v, dtype=torch.float32),edge_index=compute_edges_from_faces(f)), global_pooling=False)

    output = []
    # edges not suppose to affect the model output
    for i in range(len(normalized_patches)):
        # if vertex v[i] contained in less or equal than 4 faces output append 0 value
        # if is_vertex_in_boundary(f_downsampled, i, 4):
        #     output.append(np.zeros((1, 2)))
        #     continue
        output.append(model(Data(x=torch.tensor(normalized_patches[i], dtype=torch.float32),
                                 pos=torch.tensor(normalized_patches[i], dtype=torch.float32),
                                edge_index=knn_graph(torch.tensor(normalized_patches[i]), k=20, batch=None, loop=False), global_pooling=True)).detach().numpy())

                                 # edge_index=compute_edges_from_faces(faces[i])), global_pooling=True).detach().numpy())

    output = np.concatenate(output, axis=0)
    # try to negate the output
    output = - output

    # output = model(Data(x=torch.tensor(v, dtype=torch.float32), pos=torch.tensor(v, dtype=torch.float32),
    #                     edge_index=compute_edges_from_faces(f)), global_pooling=False)
    # output = output.detach().numpy()

    # calculate_correlation(k1, k2)
    # calculate_correlation(output[:, 0], output[:, 1])

    # add_colored_mesh(scene, faces=sample_patch_downsampled.faces, vertices=sample_patch_downsampled.vertices, colors=output[:, 0],
    #                  position=(0, 0, 0), title='output0')
    # add_colored_mesh(scene, sample_patch_downsampled.faces, sample_patch_downsampled.vertices, colors=output[:, 1], position=(dis, 0, 0),
    #                  title='output1')
    # add_colored_mesh(scene, sample_patch_downsampled.faces, sample_patch_downsampled.vertices, colors=output[:, 0] * output[:, 1],
    #                  position=(2 * dis, 0, 0),
    #                  title='output0*output1')
    # c_uint8_gt = (_get_vertex_colors_from_k(k1*k2) * 255).astype(np.uint8)
    #
    # save_glb(vertices=v, faces=f, colors=c_uint8_gt, path=Path(mesh_name+"_"+str(ratio)+'_k1_k2_gt.glb'))
    # d1_downsampled, d2_downsampled, k1_downsampled, k2_downsampled = igl.principal_curvature(v_downsampled, f_downsampled, radius=GT_radius_size)
    #
    # c_uint8_gt_downsampled = (_get_vertex_colors_from_k(k1_downsampled*k2_downsampled) * 255).astype(np.uint8)
    # save_glb(vertices=sample_patch_downsampled.vertices, faces=sample_patch_downsampled.faces, colors=c_uint8_gt_downsampled, path=Path(mesh_name+"_"+str(ratio)+'_k1_k2_downsampled_gt.glb'))

    # c_uint8_output = (_get_vertex_colors_from_k(output[:, 0]*output[:, 1]) * 255).astype(np.uint8)
    # save_glb(vertices=sample_patch_downsampled.vertices, faces=sample_patch_downsampled.faces, colors=c_uint8_output, path=Path(mesh_name+"_"+str(ratio)+'_output0_output1.glb'))

    c_uint8_output = (_get_vertex_colors_from_2_vals(output[:, 0] , output[:, 1])*255).astype(np.uint8)
    # c_uint8_output = (_get_vertex_colors_from_1_val(output[:, 0]*output[:, 1])*255).astype(np.uint8)
    save_glb(vertices=sample_patch_downsampled.vertices, faces=sample_patch_downsampled.faces, colors=c_uint8_output,
             path=Path(mesh_name + "_" + str(ratio) + '_output0_output1.glb'))

    # c_uint8_gt = (_get_vertex_colors_from_k((k1+k2)/2) * 255).astype(np.uint8)
    #
    # save_glb(vertices=v, faces=f, colors=c_uint8_gt, path=Path(mesh_name + "_" + str(ratio) + '_mean_curvature_gt.glb'))
    #
    #
    # c_uint8_gt_downsampled = (_get_vertex_colors_from_k((k1_downsampled + k2_downsampled)/2) * 255).astype(np.uint8)
    # save_glb(vertices=sample_patch_downsampled.vertices, faces=sample_patch_downsampled.faces,
    #          colors=c_uint8_gt_downsampled, path=Path(mesh_name + "_" + str(ratio) + '_mean_curvature_downsampled_gt.glb'))
    #
    # c_uint8_gt_downsampled = (_get_vertex_colors_from_k((output[:, 0] + output[:, 1])/2) * 255).astype(np.uint8)
    # save_glb(vertices=sample_patch_downsampled.vertices, faces=sample_patch_downsampled.faces,
    #          colors=c_uint8_gt_downsampled, path=Path(mesh_name + "_" + str(ratio) + '_mean_curvature_output_gt.glb'))

    # add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1, position=(0, dis, 0), title='k1')
    # add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k2, position=(dis, dis, 0),
    #                  title='k2')
    # add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1 * k2, position=(2 * dis, dis, 0),
    #                  title='k1*k2')

    # print("k1 and k2 correlation:"+str(calculate_correlation(k1, k2)))
    # print("mean k1:"+str(np.mean(k1)))
    # print("mean k2:"+str(np.mean(k2)))
    # print("k1_downsampled and k2_downsampled correlation:"+str(calculate_correlation(k1_downsampled, k2_downsampled)))
    # print("mean k1_downsampled:" + str(np.mean(k1_downsampled)))
    # print("mean k2_downsampled:" + str(np.mean(k2_downsampled)))
    # print("output[:,0] and output[:,1] correlation:"+str(calculate_correlation(output[:, 0], output[:, 1])))


    # if dataset_reg_and_unreg:
    #     sample = data[i][index + 3]  # take the reg sample
    #     v = sample.v
    #     f = sample.f
    #     # second_moments = sample.v_second_moments
    #     # sample_patch = sample
    #     sample_patch = trimesh.Trimesh(vertices=v, faces=f)
    #     # d1, d2, k1, k2 = igl.principal_curvature(v, f)
    #
    #     add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1, position=(0, 2 * dis, 0),
    #                      title='k1_reg')
    #     add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k2, position=(dis, 2 * dis, 0),
    #                      title='k2_reg')

    # scene.add(gfx.AmbientLight(1, 0.2))

    # light = gfx.DirectionalLight(1, 3)
    # light.local.position = (0, 0, 1)
    # # add lighht exactly opposite to the first one
    # light2 = gfx.DirectionalLight(1, 3)
    # light2.local.position = (0, 0, -1)
    # light3 = gfx.DirectionalLight(1, 3)
    # light3.local.position = (0, 0, 0)
    # light4 = gfx.DirectionalLight(1, 3)
    # light4.local.position = (0, 1, 0)
    # light5 = gfx.DirectionalLight(1, 3)
    # light5.local.position = (0, -1, 0)
    # light6 = gfx.DirectionalLight(1, 3)
    # light6.local.position = (1, 0, 0)
    # light7 = gfx.DirectionalLight(1, 3)
    # light7.local.position = (-1, 0, 0)
    # scene.add(light)
    # scene.add(light2)
    # scene.add(light3)
    # scene.add(light4)
    # scene.add(light5)
    # scene.add(light6)
    # scene.add(light7)
    # dark_gray = np.array((169, 167, 168, 255)) / 255
    # light_gray = np.array((100, 100, 100, 255)) / 255
    # background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
    # scene.add(background)
    # canvas.request_draw(lambda: renderer.render(scene, camera))
    #
    # def on_key_down2(event):
    #     global state, image_num
    #     nonlocal dis
    #     controls = OrbitController(camera)
    #     controls.zoom_speed = 0.001
    #     controls.distance = 0.01
    #     controls.register_events(renderer)
    #     if event.key == "s":
    #         state = camera.get_state()
    #     elif event.key == "a":
    #         # move position of rendered meshes
    #         dis  += 0.1*dis
    #         positions = [(0, 0, 0), (dis , 0, 0), (2 * dis , 0, 0), (0, dis , 0), (dis , dis , 0), (2 * dis , dis , 0)]
    #         for child in scene.children:
    #             if isinstance(child, gfx.Mesh):
    #                 child.local.position = positions[scene.children.index(child)]
    #         renderer.render(scene, camera)
    #     elif event.key == "d":
    #         dis -= 0.1*dis
    #         positions = [(0, 0, 0), (dis, 0, 0), (2 * dis, 0, 0), (0, dis, 0), (dis, dis, 0), (2 * dis, dis, 0)]
    #         for child in scene.children:
    #             if isinstance(child, gfx.Mesh):
    #                 child.local.position = positions[scene.children.index(child)]
    #         renderer.render(scene, camera)
    #
    #     elif event.key == "l":
    #         camera.set_state(state)
    #     elif event.key == "r":
    #         camera.show_object(scene)
    #     elif event.key == "q":
    #         image = renderer.snapshot()
    #         # Save the image as a PNG file
    #         date_now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    #         imageio.imwrite(mesh_name + '_' + str(ratio)+"_k_"+str(patches_size) + "_" + str(date_now) + '.png', image)
    #         image_num += 1
    #
    # renderer.add_event_handler(on_key_down2, "key_down")
    #
    # # snapshot(camera=camera, scene=scene, renderer=renderer, canvas=canvas)
    # canvas.request_draw(animate(scene=scene, camera=camera, renderer=renderer, canvas=canvas))
    # run()

if __name__ == "__main__":
    # for interactive mode
    # canvas.request_draw(animate)
    # run()


    # animate()
    # animate()

    # snapshot()
    # snapshot_multiple_patches()
    # snapshot_or_animate_torus()
    # meshes_names = ["vase-lion100K", "chair100K", "chair", "botijo", "aircraft", "blade", "dancer_25K", "dancer2", "dancing_children100K"]
    # meshes_names = ["mesh_different_sampling/non_uniform/same_ratio/peak33007.obj"]
    meshes_names = ["3d_vis/vase-lion100K.obj"]

    for mesh_name in meshes_names:
        animate_mesh(mesh_name=mesh_name)




# import matplotlib.pyplot as plt
# colormap = plt.get_cmap('viridis')  # Choose a colormap that suits your data
#
# # Normalize curvature values to the range [0, 1]
# # values = output0.detach().numpy()
# values = k1
# normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
#
# # Map normalized curvature values to colors using the colormap
# colors = colormap(normalized_values).astype(np.float32)[:, :3]
#
# # colors = colors[:, :3]  # Colors can be Nx1, Nx2, Nx3, Nx4
# setattr(geometry, 'colors', gfx.Buffer(colors))
# # setattr(geometry, 'texcoords', gfx.Buffer(texcoords))
#
# mesh = gfx.Mesh(
#     geometry,
#     gfx.MeshPhongMaterial(vertex_colors=True, emissive=(0.1, 0.1, 0.1), shininess=0.0, specular=0.0),
# )
#
#
#
#
# # if __name__ == "__main__":
# #     disp = gfx.Display()
# #     disp.show(mesh, up=(0, 0, 1))
# #
# pyvista_faces = standard_faces_to_pyvista_faces(sample.f)
# pyvista_mesh = pv.PolyData(sample.v, pyvista_faces)
# # pyvista_mesh["color_values"] = normalized_values
# plotter = pv.Plotter()
#
# plotter.add_mesh(mesh=pyvista_mesh,scalars=normalized_values, show_edges=True)
# # pyvista_mesh.plot(plotter=plotter)
# plotter.add_axes()
# plotter.show_grid()
# plotter.view_xz()
# plotter.show()