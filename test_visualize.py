import pickle

import igl
import numpy as np
import imageio.v3 as iio
import torch
from pygfx import Mesh, MeshPhongMaterial, Geometry, Color, WorldObject, Text
from torch_geometric.data import Data
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


import trimesh
import pygfx as gfx
from pygfx import OrbitController
import imageio


import pyvista as pv

from data.triplet_dataset import CustomTripletDataset
from deep_signature.generation import QuadraticMonagePatchGenerator2
from models.point_transformer_conv.model import PointTransformerConvNet
from utils import compute_edges_from_faces


def _rescale_k(k: np.ndarray) -> np.ndarray:
    # rescale gaussian curvature so it will range between 0 and 1
    min_val = k.min()
    max_val = k.max()
    rescaled_k = (k - min_val) / (max_val - min_val)
    return rescaled_k


def _get_vertex_colors_from_k(k: np.ndarray) -> np.ndarray:
    k = _rescale_k(k=k)
    k_one_minus = 1 - k

    # convex combinations between red and blue colors, based on the predicted gaussian curvature
    c1 = np.column_stack((k_one_minus, np.zeros_like(k), np.zeros_like(k), np.ones_like(k)))
    c2 = np.column_stack((np.zeros_like(k), np.zeros_like(k), k, np.ones_like(k)))
    c = c1 + c2

    return c


def _create_world_object_for_mesh(faces, vertices, k: np.ndarray, color: Color = '#ffffff') -> WorldObject:
    c = _get_vertex_colors_from_k(k=k)
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



def add_colored_mesh(scene,  faces, vertices, colors,position=(0, 0, 0),title='title'):
    mesh = _create_world_object_for_mesh(faces, vertices, colors)
    mesh.local.position = position
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
    text.local.position = mesh.local.position + (1.0, 0.5, 0)
    mesh.add(text)


file_path = "./triplets_data_size_50_N_10_all_monge_patch_normalized_pos_and_rot.pkl"

# Load the triplets from the file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

sample = data[0][1]


# patch_generator = QuadraticMonagePatchGenerator2(limit=1.0, grid_size=100)
# sample, k1, k2, point_0_0_index = patch_generator.generate(k1=0.5, k2=-0.5)

model_path = "C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2/trained_models\model_point_transformer_3_layers_width_128-epoch=99.ckpt"

model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
model.eval()

sample_patch = sample
sample_patch = trimesh.Trimesh(vertices=sample_patch.v, faces=sample_patch.f)

d1, d2, k1, k2 = igl.principal_curvature(sample.v, sample.f)

canvas = WgpuCanvas(size=(900, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
camera = gfx.PerspectiveCamera(70, 16 / 9)
# camera.show_object(scene)

output = model(Data(x=sample.v_second_moments.to(torch.float32), pos=torch.tensor(sample.v, dtype=torch.float32),edge_index=compute_edges_from_faces(sample.f)))
output = output.detach().numpy()

add_colored_mesh(scene, faces=sample_patch.faces, vertices=sample_patch.vertices, colors=output[:,0],position=(0,0,0),title='output0')
add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=output[:,1],position=(2,0,0),title='output1')
add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors = k1, position=(0,2,0),title='k1')
add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors = k2, position=(2,2,0),title='k2')


scene.add(gfx.AmbientLight(1, 0.2))

light = gfx.DirectionalLight(1, 2)
light.local.position = (0, 0, 1)
scene.add(light)
dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
scene.add(background)




# Rotate the object a bit

# rot = la.quat_from_euler((0.71, 0.1), order="XY")
# for obj in scene.children:
#     obj.local.rotation = la.quat_mul(rot, obj.local.rotation)
controls = OrbitController(camera)
controls.zoom_speed = 0.001
controls.distance = 0.01
controls.register_events(renderer)
# def on_move(event):
#     controls.handle_event(event)
# canvas.connect("mouse-move", on_move)
# canvas.connect("mouse-wheel", on_move)

image_num = 0

def on_key_down(event):
    global state, image_num
    if event.key == "s":
        state = camera.get_state()
    elif event.key == "l":
        camera.set_state(state)
    elif event.key == "r":
        camera.show_object(scene)
    elif event.key == "q":
        image = renderer.snapshot()
        # Save the image as a PNG file
        imageio.imwrite('screenshot'+str(image_num)+'.png', image)
        image_num += 1

renderer.add_event_handler(on_key_down, "key_down")

canvas.request_draw(lambda: renderer.render(scene, camera))

def animate():

    renderer.render(scene, camera)
    canvas.request_draw()

if __name__ == "__main__":
    canvas.request_draw(animate)
    run()

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