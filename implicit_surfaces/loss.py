
import torch
from torch.nn.modules.loss import _Loss

class DirichletEnergyLoss(_Loss):
    def __init__(self, k_neighbors=10):
        super(DirichletEnergyLoss, self).__init__()
        self.k_neighbors = k_neighbors

    def forward(self, point_cloud):
        # Step 1: Find k-nearest neighbors for each point
        distances, indices = torch.topk(torch.cdist(point_cloud, point_cloud), self.k_neighbors + 1, dim=1, largest=False)

        # Step 2: Fit local planes using linear regression
        A = torch.vstack([point_cloud[:, 0], torch.ones_like(point_cloud[:, 0])]).T
        coefficients, _ = torch.lstsq(point_cloud[:, 1].unsqueeze(1), A)

        # Step 3: Compute second derivatives
        d2u_dx2 = 2 * coefficients[0, 0]
        d2u_dy2 = 0  # Assuming a 2D point cloud

        # Step 4: Dirichlet energy contribution for all points
        energy_contributions = (d2u_dx2 ** 2 + d2u_dy2 ** 2).index_select(0, indices[:, 1:]).squeeze()

        # Total Dirichlet energy
        total_energy = torch.sum(energy_contributions)

        return total_energy

def calculate_dirichlet_energy(vertices, faces):
    vertex_i = vertices[faces[:, 0]]
    vertex_j = vertices[faces[:, 1]]
    vertex_k = vertices[faces[:, 2]]

    # Compute face normals
    face_normals = torch.cross(vertex_j - vertex_i, vertex_k - vertex_i)

    # Calculate face areas (magnitude of face normals)
    face_areas = torch.norm(face_normals, dim=1) / 2.0

    # Calculate Dirichlet energy contributions for each face
    energy_contributions = face_areas * torch.sum((vertex_i - vertex_j)**2 + (vertex_j - vertex_k)**2 + (vertex_k - vertex_i)**2, dim=1) / 12.0

    total_energy = torch.sum(energy_contributions)
    return total_energy.item()  # Convert to Python scalar

def rand_differences_loss(model):
    # rand 50 points
    delta = (torch.rand(50, 2) * 2 - 1)*0.1
    rand_sampled_uv = torch.rand(50, 2) * 2 - 1
    rand_sampled_uv_plus_delta = rand_sampled_uv + delta
    rand_sampled_f_uv = model(rand_sampled_uv)['model_out']
    rand_sampled_f_uv_plus_delta = model(rand_sampled_uv_plus_delta)['model_out']
    rand_sampled_f_uv_derivative = (rand_sampled_f_uv_plus_delta - rand_sampled_f_uv)
    rand_sampled_f_uv_derivative_norm = torch.norm(rand_sampled_f_uv_derivative, dim=1)
    return torch.mean(rand_sampled_f_uv_derivative_norm).item()  # Convert to Python scalar

def dirichlet_loss(model):
    rand_sampled_uv = torch.rand(50, 2) * 2 - 1
    rand_sampled_output = model(rand_sampled_uv)
    dxdy = torch.autograd.grad(rand_sampled_output["model_out"], rand_sampled_output["model_in"],
                               grad_outputs=torch.ones_like(rand_sampled_output["model_out"]), create_graph=True)
    dx = dxdy[0][:, 0]
    dy = dxdy[0][:, 1]
    loss = torch.norm(dx)**2 + torch.norm(dy)**2
    return loss