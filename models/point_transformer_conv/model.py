import igl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.pool
import wandb
from torch.nn.utils.rnn import pad_packed_sequence
from torch_geometric.nn import PointTransformerConv, radius_graph, global_mean_pool, global_max_pool, global_add_pool, \
    GraphUNet
import pytorch_lightning as pl

from loss import loss_contrastive_plus_codazzi_and_pearson_correlation, \
    loss__pearson_correlation_k1_k2, \
    loss_contrastive_plus_pearson_correlation_k1_k2, loss_gaussian_curvature_supervised, contrastive_tuplet_loss, \
    loss_contrastive_plus_pearson_correlation_k1__greater_k2, loss_contrastive_plus_k1__greater_k2, \
    loss_contrastive_plus_pearson_correlation_k1__greater_k2_hinge_loss, calculate_pearson_k1_k2_loss_vectorized, \
    loss_chamfer_distance, loss_chamfer_distance_torch, loss_intra_set_distance
from utils import normalize_points_translation_and_rotation, \
    normalize_point_cloud
from vars import LR, WEIGHT_DECAY

# Taken from https://github.com/vsitzmann/siren
# from visualize_pointclouds import visualize_pointclouds
# from visualize_pointclouds import visualize_pointclouds, visualize_pointclouds2


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(input)

class MLPWithSkipConnections(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU()):
        super(MLPWithSkipConnections, self).__init__()
        self.num_layers = num_layers
        self.activation = activation

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()

        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input layer
        input_skip = x
        x = self.activation(self.input_bn(self.input_layer(x)))

        # Hidden layers with skip connections
        for i in range(self.num_layers - 2):
            # Skip connection
            residual = x
            x = self.activation(self.hidden_bns[i](self.hidden_layers[i](x)))
            x += residual  # Skip connection

        # Output layer
        x = self.output_layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()

        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.input_bn(self.input_layer(x)))

        for i in range(self.num_layers - 2):
            x = self.activation(self.hidden_bns[i](self.hidden_layers[i](x)))

        x = self.output_layer(x)
        return x

class PointTransformerConvNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels=8, hidden_channels=64, num_point_transformer_layers=3, num_encoder_decoder_layers=8):
        super(PointTransformerConvNet, self).__init__()

        self.save_hyperparameters()  # Saves all hyperparameters for logging
        self.activation = Sine()

        self.encoder = MLPWithSkipConnections(input_dim=in_channels, hidden_dim=hidden_channels, output_dim=hidden_channels, num_layers=num_encoder_decoder_layers, activation=self.activation)
        self.conv_layers = nn.ModuleList([
            PointTransformerConv(hidden_channels, hidden_channels) for _ in range(num_point_transformer_layers)
        ])

        self.hidden_bns = nn.ModuleList()
        for _ in range(num_point_transformer_layers):  # Last layer doesn't need BN and activation
            self.hidden_bns.append(nn.BatchNorm1d(hidden_channels))

        # self.pooling = global_mean_pool  # can use other pooling functions if needed
        self.pooling = global_max_pool
        # self.pooling = global_add_pool

        # self.decoder = MLPWithSkipConnections(input_dim=hidden_channels, hidden_dim=hidden_channels, output_dim=out_channels, num_layers=num_encoder_decoder_layers, activation=self.activation)
        self.decoder = MLPWithSkipConnections(input_dim=hidden_channels, hidden_dim=hidden_channels, output_dim=out_channels, num_layers=num_encoder_decoder_layers, activation=self.activation)

        # self.loss_func = loss_contrastive_plus_pearson_correlation_k1_k2
        self.loss_func_contrastive = contrastive_tuplet_loss
        self.loss_func_pearson_corelation = calculate_pearson_k1_k2_loss_vectorized
        # self.loss_func = contrastive_tuplet_loss
        self.outputs_list_train = []
        self.outputs_list_val = []


    def forward(self, data, global_pooling=False):

        # x = self.append_moments(data.x) # x,y,z,xx,xy,xz,yy,yz,zz coordinates of each point
        x = data.x # x,y,z coordinates of each point
        # edge_index = radius_graph(x, r=0.5, batch=None, loop=True, max_num_neighbors=32)

        # Apply initial embedding
        x = self.encoder(x)

        # Apply Point Transformer Convolutional layers
        for i in range(len(self.conv_layers)):
            x = self.activation(self.hidden_bns[i](self.conv_layers[i](x=x,pos=data.pos, edge_index=data.edge_index)))

        # Apply pooling to aggregate information from vertices
        if global_pooling:
            # x = self.pooling(x, batch=data.batch)
            global_features = self.pooling(x, batch=data.batch)
            # concatenate with global features
            x = torch.cat([x, global_features[data.batch]], dim=1)

        # Apply final linear layer
        x = self.decoder(x)

        return x




    def training_step(self, batch, batch_idx):
        # batch.x = self.normalize_patches(batch)
        # batch.x = self.apply_random_rotations_to_batch(batch)
        batch.x = self.append_moments(batch.x)

        output = self.forward(batch)
        device = output.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device)%3==0]
        positive_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device)%3==1]
        negative_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device)%3==2]

        anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)
        negative_output = torch.index_select(output, 0, negative_idx)

        # make the size of the anchor and positive the same as the negative
        anchor_output = anchor_output[:negative_output.size(0)]
        positive_output = positive_output[:negative_output.size(0)]
        negative_output = negative_output[:anchor_output.size(0)]
        # self.outputs_list_train.append(torch.cat([anchor_output.T, positive_output.T, negative_output.T], dim=1).T)
        # loss = self.loss_func(a=anchor_output.T, p=positive_output.T, n=negative_output.T)
        loss_tuplet = self.loss_func_contrastive(a=anchor_output.T, p=positive_output.T, n=negative_output.T)
        # loss_pearson = self.loss_func_pearson_corelation(torch.cat([anchor_output.T, positive_output.T, negative_output.T], dim=1).T, device=anchor_output.device)
        # loss = loss_tuplet + loss_pearson
        loss = loss_tuplet

        self.log('train_loss_tuplet', loss_tuplet.item(), on_step=False, on_epoch=True)
        # self.log('train_loss_pearson', loss_pearson.item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch.x = self.normalize_patches(batch)
        # batch.x = self.apply_random_rotations_to_batch(batch)
        batch.x = self.append_moments(batch.x)
        output = self.forward(batch)
        device = output.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 1]
        negative_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 2]

        anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)
        negative_output = torch.index_select(output, 0, negative_idx)

        # make the size of the anchor and positive the same as the negative and vice versa
        anchor_output = anchor_output[:negative_output.size(0)]
        positive_output = positive_output[:negative_output.size(0)]
        negative_output = negative_output[:anchor_output.size(0)]
        # self.outputs_list_val.append(torch.cat([anchor_output.T, positive_output.T, negative_output.T], dim=1).T)

        loss_tuplet = self.loss_func_contrastive(a=anchor_output.T, p=positive_output.T, n=negative_output.T)
        # loss_pearson = self.loss_func_pearson_corelation(
        #     torch.cat([anchor_output.T, positive_output.T, negative_output.T], dim=1).T, device=anchor_output.device)
        loss = loss_tuplet

        self.log('val_loss_tuplet', loss_tuplet.item(), on_step=False, on_epoch=True)
        # self.log('val_loss_pearson', loss_pearson.item(), on_step=False, on_epoch=True)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True)
        # if batch_idx == 0:
        #     self.logger.experiment.log({"visuals - output0 and 1 on patch": wandb.Image(log_visualization(self, batch[0]))})
        return loss


    def unpack_batch(self, batch):
        output_anc = []
        output_pos = []
        output_neg = []
        for i in range(len(batch)):
            # Unpack the packed_patches to get the padded representation
            padded_patches, original_lengths = pad_packed_sequence(batch[i], batch_first=True)

            # Now you can access the individual patches
            for j in range(len(original_lengths)):
                patch_j = padded_patches[j][:original_lengths[j]]
                if j==0:
                    output_anc.append(patch_j)
                elif j==1:
                    output_pos.append(patch_j)
                else:
                    output_neg.append(patch_j)

        # Convert the lists to tensors
        # output_anc = torch.cat(output_anc, dim=0)
        # output_pos = torch.cat(output_pos, dim=0)
        # output_neg = torch.cat(output_neg, dim=0)
        return output_anc, output_pos, output_neg

    def configure_optimizers(self, lr=LR,weight_decay=WEIGHT_DECAY):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)
        return [optimizer], [scheduler]


    def on_train_epoch_end(self):
        # Get the current learning rate from the optimizer
        current_lr = self.optimizers().param_groups[0]['lr']
        # corr_mat = torch.corrcoef(torch.cat(self.outputs_list_train, dim=0).T)

        # Log the learning rate
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, sync_dist=True)
        # self.log('train_pearson_corr', corr_mat[0,1], on_step=False, on_epoch=True, sync_dist=True)

        # Reset the outputs list
        # self.outputs_list_train = []

    # def on_validation_epoch_end(self):
        # corr_mat = torch.corrcoef(torch.cat(self.outputs_list_val, dim=0).T)
        # self.log('val_pearson_corr', corr_mat[0,1], on_step=False, on_epoch=True, sync_dist=True)
        # self.outputs_list_val = []


    def normalize_features(self, features):
        mean_values = torch.mean(features, axis=1)
        std_values = torch.std(features, axis=1, unbiased=False)
        # Mask out elements with zero variance
        zero_variance_mask = (std_values == 0.0)
        std_values[zero_variance_mask] = 1.0  # Set to 1.0 to avoid division by zero

        # Step 2: Normalize the features to have zero mean and unit variance
        normalized_features = (features - torch.unsqueeze(mean_values,dim=1)) / torch.unsqueeze(std_values,dim=1)
        return normalized_features

    def append_moments(self, x: torch.Tensor) -> torch.Tensor:
        second_order_moments = torch.einsum('bi,bj->bij', x, x)

        # Get the upper triangular indices
        rows, cols = torch.triu_indices(second_order_moments.shape[1], second_order_moments.shape[2])

        # Extract the upper triangular part for each MxM matrix
        upper_triangular_values = second_order_moments[:, rows, cols]

        appended_x = torch.cat((x, upper_triangular_values.view(x.shape[0], -1)), dim=1)
        return appended_x


    def random_rotation(self, x: torch.Tensor) -> torch.Tensor:
        # Generate a random rotation matrix
        rotation_matrix = torch.randn((3, 3), device=x.device)
        # Make sure the rotation matrix is orthogonal
        q, r = torch.linalg.qr(rotation_matrix)
        d = torch.diag(r).sign()
        d = torch.diag_embed(d)
        rotation_matrix = torch.mm(q, d)

        # Apply the rotation to the input features
        rotated_x = torch.mm(x, rotation_matrix)
        # visualize_pointclouds(x, rotated_x)
        return rotated_x

    def apply_random_rotations_to_batch(self, batch):
        rotated_x = []
        for i in range(len(batch)):
            rotated_x.append(self.random_rotation(batch[i].x))
        rotated_x = torch.cat(rotated_x, dim=0)
        return rotated_x

    def normalize_patches(self, batch):
        rotated_x = []
        for i in range(len(batch)):
            # take the point with the smallest distance to the origin as the center
            mid_point = torch.argmin(torch.sqrt((batch[i].x[:,0])**2+(batch[i].x[:,1])**2), dim=0)
            # rotated_x_curr = torch_normalize_points_translation_and_rotation(batch[i].x, batch[i].x[mid_point])
            rotated_x_curr = normalize_point_cloud(batch[i].x, batch[i].x[mid_point])
            rotated_x.append(rotated_x_curr)

        # visualize_pointclouds(rotated_x[0], rotated_x[1])

        rotated_x = torch.cat(rotated_x, dim=0)
        return rotated_x

class PointTransformerConvNetReconstruct(PointTransformerConvNet):
    def __init__(self, in_channels, out_channels=8, hidden_channels=64, num_point_transformer_layers=3, num_encoder_decoder_layers=8):
        super(PointTransformerConvNetReconstruct, self).__init__(in_channels, out_channels, hidden_channels, num_point_transformer_layers, num_encoder_decoder_layers)
        self.loss_func = loss_chamfer_distance_torch

    def training_step(self, batch, batch_idx):
        x_input = batch.x
        # Unpack the batch
        batch.x = self.append_moments(batch.x)

        output = self.forward(batch)
        device = output.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 1]

        # anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)




        # Compute the loss
        anchor_input = torch.index_select(x_input, 0, anchor_idx)
        loss = self.loss_func(anchor_input, positive_output)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # visualize the point clouds
        if batch_idx % 1 == 0:
            visualize_pointclouds2(anchor_input, positive_output)
        return loss

    def validation_step(self, batch, batch_idx):
        x_input = batch.x
        batch.x = self.append_moments(batch.x)

        output = self.forward(batch)
        device = output.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, output.size(0), device=device)[batch.batch.to(device) % 3 == 1]

        # anchor_output = torch.index_select(output, 0, anchor_idx)
        positive_output = torch.index_select(output, 0, positive_idx)


        # Compute the loss
        anchor_input = torch.index_select(x_input, 0, anchor_idx)

        loss = self.loss_func(anchor_input, positive_output)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        if batch_idx % 1 == 0:
            visualize_pointclouds2(anchor_input, positive_output)
        return loss


class PointCloudReconstruction(pl.LightningModule):
    def __init__(self, num_blocks, in_channels, latent_dim, num_points_to_reconstruct=64, lamda=0.01):
        super(PointCloudReconstruction, self).__init__()
        self.first_point_transformer_conv = PointTransformerConv(in_channels, latent_dim,
            attn_nn=nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            ),
            pos_nn=nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            ))
        self.blocks = nn.ModuleList([
                PointTransformerConv(latent_dim, latent_dim,
                    attn_nn=nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim)
                ))
            for _ in range(num_blocks-1)
        ])

        self.graph_unet = GraphUNet(in_channels=latent_dim, hidden_channels=latent_dim, out_channels=latent_dim, depth=4, pool_ratios=[0.5, 0.5, 0.5, 0.5],sum_res=False)
        # self.centroid_layer = nn.Linear(latent_dim, 3)  # 3D centroid
        # self.normal_layer = nn.Linear(latent_dim, 3)    # 3D normal

        # self.mlp_project_x_axis = nn.Sequential(
        #     nn.Linear(latent_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_points_to_reconstruct)  # Output 3D point cloud
        # )
        # self.mlp_project_y_axis = nn.Sequential(
        #     nn.Linear(latent_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_points_to_reconstruct)  # Output 3D point cloud
        # )
        self.mlp_project_z_axis = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_points_to_reconstruct)  # Output 3D point cloud
        )
        self.mlp_eps_x = nn.Sequential(
            nn.Linear(num_points_to_reconstruct, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output 3D point cloud
        )
        self.mlp_eps_y = nn.Sequential(
            nn.Linear(num_points_to_reconstruct, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output 3D point cloud
        )

        self.loss_func_chamfer = loss_chamfer_distance_torch
        self.loss_func_intra = loss_intra_set_distance

        # want to pool the output from Mxnum_points_to_reconstructx3 to num_points_to_reconstructx3

        self.num_points_to_reconstruct = num_points_to_reconstruct
        self.lamda = lamda


    def forward(self, data):
        x= data.x
        x = self.first_point_transformer_conv(x=x,pos=data.pos, edge_index=data.edge_index)
        for block in self.blocks:
            x = block(x=x,pos=data.pos, edge_index=data.edge_index)
        x = self.graph_unet(x, edge_index=data.edge_index)
        # extract fps points from x
        x= x[data.fps_indices]
        # centroid = self.centroid_layer(x)
        # normal = self.normal_layer(x)
        # proj_x = self.mlp_project_x_axis(x)
        # proj_y = self.mlp_project_y_axis(x)
        x = self.mlp_project_z_axis(x)
        # x = torch.stack([proj_x, proj_y, proj_z], dim=-1)
        # predict epsilon
        eps_x = self.mlp_eps_x(x)
        eps_y = self.mlp_eps_y(x)

        # x = torch.cat([proj_z, eps], dim=-1)
        # x = torch.mean(x, dim=0)
        return x, eps_x, eps_y

    def training_step(self, batch, batch_idx):
        device = batch.x.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, batch.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, batch.size(0), device=device)[batch.batch.to(device) % 3 == 1]

        anchor_input_x = batch.x[anchor_idx]
        anchor_input_normals = batch.fps_normals[0]
        anchor_input_fps_indices = batch[0].fps_indices

        positive_batch_idx = torch.unique(batch.batch[positive_idx])
        # work only on 1 sized batch
        positive_input_batch = batch[positive_batch_idx.item()]


        positive_input_batch.x = self.append_moments(positive_input_batch.x)

        pos_z_output, pos_eps_x, pos_eps_y = self.forward(positive_input_batch)
        # create meshgrid from pos_eps_x and pos_eps_y
        grid_dim = int(torch.sqrt(torch.Tensor([self.num_points_to_reconstruct])))

        grids_array = [torch.meshgrid([torch.linspace(int(-(grid_dim - 1) / 2),
                                                      int((grid_dim - 1) / 2),
                                                      grid_dim, device=device) * eps_x,
                                       torch.linspace(int(- (grid_dim - 1) / 2),
                                                      int((grid_dim - 1) / 2, ),
                                                      grid_dim, device=device) * eps_y]) for eps_x, eps_y in
                       zip(pos_eps_x, pos_eps_y)]
        # create 3d point clouds as patches and translate them to the sampled points positions
        # depended on self.num_points_to_reconstruct to be a perfect square like 64
        pos_output = torch.stack([self.rotate_point_cloud(torch.stack([x[0], x[1], pos_z_output[0].view(int(np.sqrt(self.num_points_to_reconstruct)),int(np.sqrt(self.num_points_to_reconstruct)))], dim=-1),torch.tensor(anchor_input_normals[i],dtype=torch.float))+anchor_input_x[anchor_input_fps_indices[i]] for  i, x in enumerate(grids_array)], dim=0)
        pos_output = pos_output.view(-1, self.num_points_to_reconstruct, 3)
        pos_output = pos_output.view(-1,3)
        # anchor_output = torch.index_select(output, 0, anchor_idx)
        # positive_output = torch.index_select(output, 0, positive_idx)
        # reshape positive output to be of shape (batch_size, num_points, 3)
        # positive_output = positive_output.view(-1, self.num_points_to_reconstruct, 3)


        # Compute the loss
        # anchor_input = torch.index_select(anchor_input, 0, anchor_idx)
        loss_chamfer = self.loss_func_chamfer(anchor_input_x, pos_output)
        loss_intra = self.loss_func_intra(pos_output)
        loss = loss_chamfer + self.lamda*loss_intra
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # visualize the point clouds
        # if batch_idx % 8 == 0:
        #     visualize_pointclouds2(anchor_input_x, pos_output)
        #     visualize_pointclouds2(anchor_input_x, positive_input_batch.x[:, :3])
        return loss

    def validation_step(self, batch, batch_idx):
        device = batch.x.device

        # fix all tensors to device
        anchor_idx = torch.arange(0, batch.size(0), device=device)[batch.batch.to(device) % 3 == 0]
        positive_idx = torch.arange(0, batch.size(0), device=device)[batch.batch.to(device) % 3 == 1]

        anchor_input_x = batch.x[anchor_idx]
        anchor_input_normals = batch.fps_normals[0]
        anchor_input_fps_indices = batch[0].fps_indices

        positive_batch_idx = torch.unique(batch.batch[positive_idx])
        # work only on 1 sized batch
        positive_input_batch = batch[positive_batch_idx.item()]

        positive_input_batch.x = self.append_moments(positive_input_batch.x)

        pos_z_output, pos_eps_x, pos_eps_y = self.forward(positive_input_batch)
        # create meshgrid from pos_eps_x and pos_eps_y
        grid_dim = int(torch.sqrt(torch.Tensor([self.num_points_to_reconstruct])))
        # make sure all in the same device

        grids_array = [torch.meshgrid([torch.linspace(int(-(grid_dim - 1) / 2),
                                                      int((grid_dim - 1) / 2),
                                                      grid_dim, device=device) * eps_x,
                                       torch.linspace(int(- (grid_dim - 1) / 2),
                                                      int((grid_dim - 1) / 2,),
                                                      grid_dim,  device=device) * eps_y]) for eps_x, eps_y in
                       zip(pos_eps_x, pos_eps_y)]
        # create 3d point clouds as patches and translate them to the sampled points positions
        # depended on self.num_points_to_reconstruct to be a perfect square like 64
        pos_output = torch.stack([self.rotate_point_cloud(torch.stack([x[0], x[1], pos_z_output[0].view(
            int(np.sqrt(self.num_points_to_reconstruct)), int(np.sqrt(self.num_points_to_reconstruct)))], dim=-1).to(device),
                                                          torch.tensor(anchor_input_normals[i], dtype=torch.float, device=device)) +
                                  anchor_input_x[anchor_input_fps_indices[i]].to(device) for i, x in enumerate(grids_array)],
                                 dim=0)
        pos_output = pos_output.view(-1, self.num_points_to_reconstruct, 3)
        pos_output = pos_output.view(-1, 3)
        # anchor_output = torch.index_select(output, 0, anchor_idx)
        # positive_output = torch.index_select(output, 0, positive_idx)
        # reshape positive output to be of shape (batch_size, num_points, 3)
        # positive_output = positive_output.view(-1, self.num_points_to_reconstruct, 3)

        # Compute the loss
        # anchor_input = torch.index_select(anchor_input, 0, anchor_idx)
        loss_chamfer = self.loss_func_chamfer(anchor_input_x, pos_output)
        loss_intra = self.loss_func_intra(pos_output)
        loss = loss_chamfer + self.lamda * loss_intra
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        # if batch_idx % 1 == 0:
        #     visualize_pointclouds2(anchor_input, pos_output)
        return loss

    def append_moments(self, x: torch.Tensor) -> torch.Tensor:
        second_order_moments = torch.einsum('bi,bj->bij', x, x)

        # Get the upper triangular indices
        rows, cols = torch.triu_indices(second_order_moments.shape[1], second_order_moments.shape[2])

        # Extract the upper triangular part for each MxM matrix
        upper_triangular_values = second_order_moments[:, rows, cols]

        appended_x = torch.cat((x, upper_triangular_values.view(x.shape[0], -1)), dim=1)

        return appended_x

    def configure_optimizers(self, lr=LR, weight_decay=WEIGHT_DECAY):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, sync_dist=True)



    def calculate_rotation_matrix(self, normals):
        # Create a reference vector pointing in the z-direction
        reference_vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        # Calculate the rotation axis as the cross product between normals and reference_vector
        rotation_axis = torch.cross(normals, reference_vector)

        # Calculate the dot product between normals and reference_vector for the angle
        dot_product = torch.sum(normals * reference_vector, dim=-1)
        angle = torch.acos(dot_product / (torch.norm(normals, dim=-1) * torch.norm(reference_vector)))

        # Normalize the rotation axis
        rotation_axis = rotation_axis / torch.norm(rotation_axis, dim=-1, keepdim=True)

        # Convert axis-angle to quaternion
        half_angle = angle / 2
        quaternion = torch.cat(
            [torch.cos(half_angle).unsqueeze(-1), rotation_axis * torch.sin(half_angle).unsqueeze(-1)], dim=-1)

        # Convert quaternion to rotation matrix
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)

        return rotation_matrix

    def quaternion_to_rotation_matrix(self, quaternion):
        q0, q1, q2, q3 = quaternion.unbind(dim=-1)

        rotation_matrix = torch.stack([
            1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
            2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1),
            2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)
        ]).view(-1, 3, 3)

        return rotation_matrix

    def rotate_point_cloud(self, point_cloud, normal_desired_direction):
        # assume each patch has normal of [0,0,1] and need to be rotated to normal_desired_direction
        # calculate the rotation matrix
        point_cloud = point_cloud.view(-1, 3)

        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).to(normal_desired_direction.device)
        rotation_axis = torch.cross(z_axis, normal_desired_direction)
        angle = torch.acos(torch.dot(z_axis, normal_desired_direction) / (torch.norm(z_axis) * torch.norm(normal_desired_direction)))
        rotation_matrix = self.rotation_matrix_from_axis_angle(rotation_axis, angle)

        # rotate the point cloud
        rotated_point_cloud = torch.matmul(point_cloud.to(normal_desired_direction.device), (rotation_matrix.to(normal_desired_direction.device)).T)
        return rotated_point_cloud

    def rotation_matrix_from_axis_angle(self, axis, angle):
        """
        Create a 3x3 rotation matrix from an axis and an angle using the Rodrigues' rotation formula.
        """
        axis = axis / torch.norm(axis)
        c = torch.cos(angle)
        s = torch.sin(angle)
        t = 1 - c

        x, y, z = axis

        rotation_matrix = torch.tensor([
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]
        ], dtype=torch.float32)

        return rotation_matrix

