import pickle
from datetime import datetime

import igl
import torch
from PIL import Image
from pygfx import Mesh, MeshPhongMaterial, Geometry, Color, WorldObject, Text, OrbitController
from scipy.spatial import Delaunay
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops

from wgpu.gui.offscreen import WgpuCanvas

import trimesh
import pygfx as gfx
import imageio
import numpy as np


from utils import compute_edges_from_faces



def _rescale_k(k: np.ndarray) -> np.ndarray:
    # rescale colors so it will range between 0 and 1
    min_val = k.min()
    max_val = k.max()
    rescaled_k = (k - min_val+1e-5) / (max_val - min_val+1e-5)
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



def snapshot(renderer, scene, camera, canvas):
    renderer.render(scene, camera)
    # canvas.request_draw()
    camera.show_object(scene)
    camera.rotation = (50, 5, 0)
    canvas.draw()
    # canvas._get_event_wait_time()
    # wait for render to complete
    image = renderer.snapshot()
    # Save the image as a PNG file
    path = "snapshot.png"
    imageio.imwrite(path, image)
    return path

f = None
k1 = None
k2 = None

def log_visualization(model, data):
    global f, k1, k2
    v = data.pos.cpu().numpy()
    # get faces from edge index
    if f is None:
        f = Delaunay(np.stack([v[:,0], v[:,1]], axis=1)).simplices
    # second_moments = sample.v_second_moments
    dis = 2  # distance between the rendered patches in the visualization



    model.eval()
    # output = model(Data(x=second_moments.to(torch.float32), pos=torch.tensor(v, dtype=torch.float32),
    #                     edge_index=compute_edges_from_faces(f)), global_pooling=False)
    output = model(data, global_pooling=False)
    model.train()

    sample_patch = trimesh.Trimesh(vertices=v, faces=f)
    v = sample_patch.vertices
    f = sample_patch.faces
    if k1 is None:
        d1, d2, k1, k2 = igl.principal_curvature(v, f)

    canvas = WgpuCanvas(size=(900, 400))
    # im getting assert adapter_id is not None
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, 16 / 9)
    # camera.show_object(scene)
    output = output.detach().numpy()

    add_colored_mesh(scene, faces=sample_patch.faces, vertices=sample_patch.vertices, colors=output[:, 0],
                     position=(0, 0, 0), title='output0')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=output[:, 1], position=(dis, 0, 0),
                     title='output1')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k1, position=(0, dis, 0), title='k1')
    add_colored_mesh(scene, sample_patch.faces, sample_patch.vertices, colors=k2, position=(dis, dis, 0), title='k2')

    scene.add(gfx.AmbientLight(1, 0.2))

    light = gfx.DirectionalLight(1, 2)
    light.local.position = (0, 0, 1)
    scene.add(light)
    dark_gray = np.array((169, 167, 168, 255)) / 255
    light_gray = np.array((100, 100, 100, 255)) / 255
    background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
    scene.add(background)

    canvas.request_draw(lambda: renderer.render(scene, camera))

    return snapshot(renderer, scene, camera, canvas)

