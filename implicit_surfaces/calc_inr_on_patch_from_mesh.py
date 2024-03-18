import torch

from data.generation import PeakSaddleGenerator, QuadraticMonageParabolicPlanarPatchGenerator
import numpy as np
import open3d as o3d

from implicit_surfaces.differential_geometry_on_inr import calculate_shape_operator_and_principal_directions, \
    calculate_laplace_beltrami
# from visualize_pointclouds import visualize_pointclouds2
from implicit_surfaces.model_relu import MLP
from model import SIREN
from loss import calculate_dirichlet_energy, rand_differences_loss, DirichletEnergyLoss, dirichlet_loss, \
    sort_of_laplacian_loss

from scipy.spatial import distance,cKDTree
import random

epochs = 1000


# calculate mesh mid point normal
mesh_name = "bunny_curvs"
mesh = o3d.io.read_triangle_mesh("meshes/"+mesh_name+".ply")
v = np.array(mesh.vertices)
subset_sample_size = int(0.01*len(v))
random_points_indices = random.sample(range(len(v)), subset_sample_size)
# just to calculate distances
v_subset = v[random_points_indices]


sampled_point_index = np.random.randint(0, len(v))
distances = distance.pdist(v_subset)
r = 0.05*np.mean(distances)
# sample neighborhood
kdtree = cKDTree(v)
indices = kdtree.query_ball_point(v[sampled_point_index],r)
v_subset = v[indices]
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(v_subset)
# calculate normals
point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
point_cloud.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

# calculate midpoint of point cloud by taking the point closest to the center of the point cloud
center = point_cloud.get_center()
distances = distance.cdist(np.array(center).reshape(1,3), v_subset)
mid_point_index = np.argmin(distances)
mid_point_normal = point_cloud.normals[mid_point_index]

# calculate the tangent plane projection of all the points on the mesh
v_subset =v_subset - v_subset[mid_point_index]
f_uv = np.dot(v_subset, mid_point_normal)
input_v = v_subset - f_uv[:, None] * mid_point_normal[None, :]
input_v = torch.stack([torch.tensor(input_v[:, 0], dtype=torch.float32), torch.tensor(input_v[:, 1], dtype=torch.float32)], dim=1)
f_uv = torch.tensor(f_uv, dtype=torch.float32).view(-1, 1)

# declare the model
hidden_layer_config =  [256, 256, 256]
model = SIREN(n_in_features=2, n_out_features=1, hidden_layer_config=hidden_layer_config)
# model = MLP(n_in_features=2, n_out_features=1, hidden_layer_config=hidden_layer_config)
model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_dirichlet = DirichletEnergyLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1000, gamma=0.5)
# train the model on input_v
x_unit_vector = torch.tensor([1.0, 0.0], dtype=torch.float32)
y_unit_vector = torch.tensor([0.0, 1.0], dtype=torch.float32)
center_point = torch.tensor([0.0,0.0])
# boundary_indices = torch.where((torch.abs(torch.abs(input_v[:, 0]) -1)<0.03) | ((torch.abs(torch.abs(input_v[:, 1]) - 1)<0.03)))

for epoch in range(epochs):
    # forward pass
    output = model(input_v)
    # 0,0 point is the center of the patch

    # output_center = {"model_out": output["model_out"][2].view(1, 1), "model_in": output["model_in"][2].view(1, 2)}

    output_center = model(center_point.view(1, 2))
    center_loss = torch.norm(output_center['model_out']) ** 2
    # output_boundaries = output["model_out"][boundary_indices]
    # boundary_loss =  1000*torch.mean(torch.norm(output_boundaries-f_uv[boundary_indices])**2)

    # backward pass
    # loss = torch.norm(output['model_out']-f_uv) ** 2 + 0.01 * rand_differences_loss(model)
    # dxdy = torch.autograd.grad(output["model_out"], output["model_in"],grad_outputs=torch.ones_like(output["model_out"]), create_graph=True)
    # dx = dxdy[0][:, 0]
    # dy = dxdy[0][:, 1]
    loss_reconstruction = torch.mean(torch.norm(output['model_out']-f_uv)**2) + 1000*center_loss

    reg_term = sum(torch.norm(param) ** 2 for param in model.parameters())

    # loss_reconstruction = torch.mean(torch.norm(output['model_out']-f_uv)**2) + 1000*(center_loss+boundary_loss)
    loss_smooth = dirichlet_loss(model)
    # loss_smooth = sort_of_laplacian_loss(model) + dirichlet_loss(model)
    # squared mean
    # if epoch>500:
    # v = torch.stack([input_v[:, 0], input_v[:, 1], output['model_out'][:,0]], dim=1)
    # loss_smooth = calculate_laplace_beltrami(model, num_eigenvalues=10)
    # loss_reconstruction = loss_reconstruction + 0.001 * loss_smoothness
    # loss_reconstruction = loss_reconstruction + 0.01 * loss_smoothness

    if epoch > 500:
        e1, e2, grad = calculate_shape_operator_and_principal_directions(output, mid_point=mid_point_index)
        e1 = e1.to(input_v.device)
        e2 = e2.to(input_v.device)
        # rotate input_v and f_uv to align e1 and e2 with x and y unit vectors
        # e1 vs x_unit_vector angle differentiable


        # with torch.no_grad():
        dot_product = torch.dot(e1, x_unit_vector)
        # if abs(dot_product)< 0.999:
        print("dot_product", dot_product)
        if epoch% 2 == 0:
            angle_to_rotate = torch.acos(dot_product)*0.01
        else:
            angle_to_rotate = torch.tensor(0.0, dtype=torch.float32).to(input_v.device)
        # angle_to_rotate = torch.atan2(e1[0] * x_unit_vector[1] - e1[1] * x_unit_vector[0], torch.dot(e1, x_unit_vector))

        # rotate input_v and f_uv to align e1 and e2 with x and y unit vectors rotate around z axis
        input_v = torch.stack([input_v[:, 0]*torch.cos(angle_to_rotate) - input_v[:, 1]*torch.sin(angle_to_rotate), input_v[:, 0]*torch.sin(angle_to_rotate) + input_v[:, 1]*torch.cos(angle_to_rotate)], dim=1)
        # loss_principal_directions = torch.norm(e1-x_unit_vector) ** 2 + torch.norm(e2-y_unit_vector) ** 2 + torch.dot(e1, e2) ** 2
        loss_principal_directions = torch.norm(torch.dot(e1,x_unit_vector)-1) ** 2 + torch.norm(torch.dot(e2,y_unit_vector)-1) ** 2 + torch.norm(torch.dot(e1, e2)) ** 2
        loss = loss_reconstruction + 10.0*reg_term + 0.01*loss_principal_directions + 0.001*loss_smooth
        # else:
        #     loss = loss_reconstruction + 10.0 * reg_term

    else:

        # loss = loss_reconstruction + 0.001 * loss_smooth
        loss = loss_reconstruction + 10.0*reg_term
        print("reg_term_loss:"+str(reg_term.item()))
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    # update the weights
    model.optimizer.step()
    model.optimizer.zero_grad()
    # scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    if epoch%1000 == 0:
        torch.save(model.state_dict(), "inr_model_" + mesh_name + "_patch_with_reg_dirchlet_principal_loss.pth")

# save model
torch.save(model.state_dict(), "inr_model_"+mesh_name+"_patch_with_reg_dirchlet_principal_loss.pth")







