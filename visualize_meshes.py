
import torch
import pyvista as pv

# use different library for visualization
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas
import numpy as np

def visualize_meshes_func(mesh1, mesh2=None, mesh3=None, mesh4=None, labels=None, title=None, vector_fields_to_visualize=None, fps_indices=None, arrow_scale=0.001):


        mesh1_pv = pv.PolyData(mesh1[0], mesh1[1].astype(np.int32))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh1_pv, color="blue", show_edges=True, line_width=1, opacity=0.5)

        if vector_fields_to_visualize is not None:
            colors = ["red","blue", "green", "yellow"]
            for i in range(len(vector_fields_to_visualize)):
                plotter.add_arrows(cent=mesh1[0][fps_indices], direction=vector_fields_to_visualize[i], color=colors[i], mag=arrow_scale)


        if title:
            plotter.set_title(title)

        plotter.add_axes_at_origin()

        # Show the plot

        plotter.show()


def visualize_meshes_func2(mesh1, vector_fields_to_visualize=None, fps_indices=None, arrow_scale=0.001):


        # use pygfx without pyvista
        # create a scene
        scene = gfx.Scene()

        # Add meshes to the scene
        if mesh1 is not None:
            scene.add( gfx.Mesh(geometry=gfx.Geometry(position=mesh1[0], indices=mesh1[1])))

        if vector_fields_to_visualize is not None:
            colors = [(1, 0, 0, 0)]
            for i in range(len(vector_fields_to_visualize)):
                # create a line geometry from the vector field
                line_geometry = gfx.Geometry(positions=vector_fields_to_visualize[i][None,:])
                # create a line material with the corresponding color
                line_material = gfx.LineMaterial()
                # create a line object from the geometry and material
                line = gfx.Line(line_geometry, line_material)
                # add the line to the scene
                scene.add(line)


            # create a renderer
        # create a canvas to display the scene
        canvas = WgpuCanvas()
        renderer = gfx.WgpuRenderer(target=canvas)
        gfx.show(scene,canvas=canvas, renderer=renderer)



def visualize_meshes_func3(mesh1, mesh2=None, mesh3=None, mesh4=None, labels=None, title=None, vector_fields_to_visualize=None, fps_indices=None, arrow_scale=0.001):
    """
    Visualizes meshes using pygfx.

    Args:
        mesh1: A tuple (vertices, faces) representing the first mesh.
        mesh2: A tuple (vertices, faces) representing the second mesh (optional).
        mesh3: A tuple (vertices, faces) representing the third mesh (optional).
        mesh4: A tuple (vertices, faces) representing the fourth mesh (optional).
        labels: A list of labels for the meshes (optional).
        title: The title for the visualization (optional).
        vector_fields_to_visualize: A list of vector fields to visualize (optional).
        fps_indices: A list of indices corresponding to the vector field locations (optional).
        arrow_scale: The scale for the arrows representing vector fields (optional).
    """

    scene = gfx.Scene()

    # Create meshes
    if mesh1 is not None:
        vertices, faces = mesh1
        mesh_geometry = gfx.Geometry(positions=vertices, indices=faces)
        mesh = gfx.Mesh(mesh_geometry)
        scene.add(mesh)  # Use add instead of add_entity



    # Add vector fields (if provided)
    if vector_fields_to_visualize is not None and fps_indices is not None:
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
        for i, vector_field in enumerate(vector_fields_to_visualize):
            origin = gfx.vec3(mesh1[0][fps_indices[i]])
            direction = gfx.vec3(vector_field)
            arrow = gfx.LineArrowMaterial(origin=origin, direction=direction, color=colors[i], scale=arrow_scale)
            scene.add(arrow)


    # Add camera and light
    camera = gfx.PerspectiveCamera()
    scene.add(camera)

    light = gfx.DirectionalLight(direction=(1, 1, 1))
    scene.add(light)

    # Render and show the scene
    renderer = gfx.Renderer()
    renderer.render(scene)
    renderer.show()
