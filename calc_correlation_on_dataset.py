import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from tqdm import tqdm
import igl
from scipy.spatial import Delaunay

from models.point_transformer_conv.model import PointTransformerConvNet


def calculate_outputs_with_model(model, data):
    outputs = []
    for i in tqdm(range(len(data))):
        # data[i] = data[i].cuda()
        outputs.append(model(Data(x=torch.tensor(data[i].v, dtype=torch.float32),
                                  pos=torch.tensor(data[i].v, dtype=torch.float32),
                                  edge_index=knn_graph(torch.tensor(data[i].v), k=6, batch=None,
                                                       loop=False), global_pooling=True)).detach().numpy())


    return outputs

def calculate_outputs_with_igl(data):
    outputs = []
    for i in tqdm(range(len(data))):
        f = Delaunay(data[i].v[:, :2])
        d1, d2, k1, k2 = igl.principal_curvature(data[i].v, f.simplices, radius=30)
        center_indice = np.argmin(data[i].v[:, 0] ** 2 + data[i].v[:, 1] ** 2)
        # create array of size 1x2
        output = np.array([k1[center_indice], k2[center_indice]])

        outputs.append(output)
        # f = igl.delaunay_triangulation(np.array([data[i].v[:, 0],data[i].v[:, 1]]).T)


    return outputs





if torch.cuda.is_available():
    server_dir = "/home/gal.yona/deep-signature-2/"
    file_path = server_dir+"data/spherical_monge_patches_100_N_500.pkl"
    file_path2 = server_dir+"data/hyperbolic_monge_patches_100_N_500.pkl"
    file_path3 = server_dir+"data/parabolic_monge_patches_100_N_500.pkl"

else:
    file_path = "data/spherical_monge_patches_100_N_10.pkl"
    file_path2 = "data/hyperbolic_monge_patches_100_N_10.pkl"
    file_path3 = "data/parabolic_monge_patches_100_N_10.pkl"




# Load the triplets from the file
with open(file_path, 'rb') as f:
    f.seek(0)  # Move the file pointer to the beginning of the file
    data_spherical = pickle.load(f)
with open(file_path2, 'rb') as f:
    f.seek(0)
    data_hyperbolic = pickle.load(f)
with open(file_path3, 'rb') as f:
    f.seek(0)
    data_parabolic = pickle.load(f)


# model_path = "C:/Users\galyo\Downloads\model_point_transformer_1_layers_width_128_train_non_uniform_samples_also_with_planar_patches-epoch=149.ckpt"
model_path = "/home/gal.yona/deep-signature-2/checkpoints/model_point_transformer_1_layers_width_128_train_non_uniform_samples_also_with_planar_patches-epoch=344.ckpt"
model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
model.eval()
# outputs1 = calculate_outputs_with_model(model, data_spherical)
# outputs2 = calculate_outputs_with_model(model, data_hyperbolic)
# outputs3 = calculate_outputs_with_model(model, data_parabolic)
outputs1 = calculate_outputs_with_igl(data_spherical)
outputs2 = calculate_outputs_with_igl(data_hyperbolic)
outputs3 = calculate_outputs_with_igl(data_parabolic)

output = np.concatenate([outputs1, outputs2, outputs3], axis=0).squeeze()

corr_mat = np.corrcoef(output.T)
print(corr_mat)




