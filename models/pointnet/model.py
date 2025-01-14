from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_packed_sequence

from loss import loss_contrastive_plus_codazzi_and_pearson_correlation


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNet_FC(pl.LightningModule):
    def __init__(self, k=64, dropout_prob=0.8):
        super(PointNet_FC, self).__init__()

        self.fc1 = nn.Linear(k, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 128)  # Additional fc layer after the Max Pooling
        self.fc5 = nn.Linear(128, 8)  # Additional fc layer after the Max Pooling

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(128)

        self.k = k
        self.loss_func = loss_contrastive_plus_codazzi_and_pearson_correlation

    def forward(self, x):
        normalized_features = self.normalize_features(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Apply Max Pooling
        x = torch.max(x, 0, keepdim=True)[0]

        # Additional fully connected layers
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        return x


    def forward_with_bn(self, x):
        normalized_features = self.normalize_features(x)
        x = F.relu(self.bn1(self.fc1(normalized_features)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        # Apply Max Pooling
        x = torch.max(x, 0, keepdim=True)[0]

        # Additional fully connected layers
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))

        return x


    def training_step(self, batch, batch_idx):
        anc_patches, pos_patches, neg_patches = self.unpack_batch(batch)
        loss = 0.0
        for i in range(len(anc_patches)):
            output_anc = self.forward(anc_patches[i].float())

            output_pos = self.forward(pos_patches[i].float())

            output_neg = self.forward(neg_patches[i].float())

            loss += self.loss_func(a=output_anc.T, p=output_pos.T,
                                   n=output_neg.T)

        print("Training loss: ", loss.item())
        self.log('train_loss', loss,on_step=False, on_epoch=True)  # Logging the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        # anc_patches, pos_patches, neg_patches = batch[:,0,:,:], batch[:,1,:,:], batch[:,2,:,:]
        anc_patches, pos_patches, neg_patches = self.unpack_batch(batch)
        loss = 0.0
        for i in range(len(anc_patches)):
            output_anc = self.forward(anc_patches[i].float())

            output_pos = self.forward(pos_patches[i].float())

            output_neg = self.forward(neg_patches[i].float())

            loss += self.loss_func(a=output_anc.T, p=output_pos.T,
                                   n=output_neg.T)

        self.log('val_loss', loss,on_step=False, on_epoch=True)  # Logging the validation loss
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

    def configure_optimizers(self, lr=0.001,weight_decay=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]


    def on_train_epoch_end(self):
        # Get the current learning rate from the optimizer
        current_lr = self.optimizers().param_groups[0]['lr']

        # Log the learning rate
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True)


    def normalize_features(self, features):
        mean_values = torch.mean(features, axis=1)
        std_values = torch.std(features, axis=1, unbiased=False)
        # Mask out elements with zero variance
        zero_variance_mask = (std_values == 0.0)
        std_values[zero_variance_mask] = 1.0  # Set to 1.0 to avoid division by zero

        # Step 2: Normalize the features to have zero mean and unit variance
        normalized_features = (features - torch.unsqueeze(mean_values,dim=1)) / torch.unsqueeze(std_values,dim=1)
        return normalized_features


class STNkd(pl.LightningModule):
    def __init__(self, k=64, dropout_prob=0.8):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer with given dropout probability

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        self.loss_func = loss_contrastive_plus_codazzi_and_pearson_correlation

    def forward(self, x):

        normalized_features = self.normalize_features(x)
        x = F.relu(self.bn1(self.conv1(normalized_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.avg_pool1d(x, x.size(-1))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, self.k, self.k)
        return x

    def training_step(self, batch, batch_idx):
        anc_patches, pos_patches, neg_patches = batch[:,0,:,:], batch[:,1,:,:], batch[:,2,:,:]
        # ... (your training logic, forward pass, loss computation, etc.)

        output_anc = self.forward(torch.transpose(anc_patches, 1, 2).float())

        output_pos = self.forward(torch.transpose(pos_patches, 1, 2).float())

        output_neg = self.forward(torch.transpose(neg_patches, 1, 2).float())

        loss = self.loss_func(a=output_anc.T, p=output_pos.T,n=output_neg.T)

        self.log('train_loss', loss,on_step=False, on_epoch=True)  # Logging the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        anc_patches, pos_patches, neg_patches = batch[:,0,:,:], batch[:,1,:,:], batch[:,2,:,:]

        output_anc = self.forward(torch.transpose(anc_patches,1,2).float())

        output_pos = self.forward(torch.transpose(pos_patches,1,2).float())

        output_neg = self.forward(torch.transpose(neg_patches,1,2).float())

        loss = self.loss_func(a=output_anc.T, p=output_pos.T, n=output_neg.T)


        self.log('val_loss', loss,on_step=False, on_epoch=True)  # Logging the validation loss
        return loss

    def configure_optimizers(self, lr=0.001,weight_decay=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        # Get the current learning rate from the optimizer
        current_lr = self.optimizers().param_groups[0]['lr']

        # Log the learning rate
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True)

    def normalize_features(self, features):
        mean_values = torch.mean(features, axis=2)
        std_values = torch.std(features, axis=2)
        # Mask out elements with zero variance
        zero_variance_mask = (std_values == 0.0)
        std_values[zero_variance_mask] = 1.0  # Set to 1.0 to avoid division by zero

        # Step 2: Normalize the features to have zero mean and unit variance
        normalized_features = (features - torch.unsqueeze(mean_values,dim=2)) / torch.unsqueeze(std_values,dim=2)
        return normalized_features

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

# if __name__ == '__main__':
#     sim_data = Variable(torch.rand(32,3,2500))
#     trans = STN3d()
#     out = trans(sim_data)
#     print('stn', out.size())
#     print('loss', feature_transform_regularizer(out))
#
#     sim_data_64d = Variable(torch.rand(32, 64, 2500))
#     trans = STNkd(k=64)
#     out = trans(sim_data_64d)
#     print('stn64d', out.size())
#     print('loss', feature_transform_regularizer(out))
#
#     pointfeat = PointNetfeat(global_feat=True)
#     out, _, _ = pointfeat(sim_data)
#     print('global feat', out.size())
#
#     pointfeat = PointNetfeat(global_feat=False)
#     out, _, _ = pointfeat(sim_data)
#     print('point feat', out.size())
#
#     cls = PointNetCls(k = 5)
#     out, _, _ = cls(sim_data)
#     print('class', out.size())
#
#     seg = PointNetDenseCls(k = 3)
#     out, _, _ = seg(sim_data)
#     print('seg', out.size())
