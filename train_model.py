import os
import pickle

import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.point_transformer_conv.model import PointTransformerConvNet
from utils import init_wandb, custom_euclidean_transform

from data.triplet_dataset import CustomTripletDataset
from models.pointnet.model import STNkd, PointNet_FC
from visualize.visualize__results_on_mesh import VisualizerCallback


def main_loop():
    max_epochs = 100
    lr = 0.0001
    weight_decay = 0.01
    num_workers = 1
    combine_reg_and_non_reg_patches = False
    if torch.cuda.is_available():
        file_path = "/home/gal.yona/deep-signature-2/triplets_data_size_50_N_100000_all_monge_patch_normalized_pos_and_rot.pkl"
        num_workers = 1
        combine_reg_and_non_reg_patches = True

    else:
        file_path = "triplets_data_size_50_N_10_all_monge_patch_normalized_pos_and_rot.pkl"
        os.environ["WANDB_MODE"] = "offline"




    # Load the triplets from the file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if combine_reg_and_non_reg_patches:
        file_path2 = "/home/gal.yona/deep-signature-2/triplets_data_size_50_N_10000_all_monge_patch_normalized_pos_and_rot.pkl"
        with open(file_path2, 'rb') as f:
            data2 = pickle.load(f)

        data = data + data2

    # Create custom dataset
    custom_dataset = CustomTripletDataset(data[int(len(data)*0.2):])
    # release memory
    del data
    # Define the ratio for train and validation split (e.g., 80% for training, 20% for validation)
    train_ratio = 0.8

    # Calculate the number of samples for train and validation sets
    num_samples = len(custom_dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_val_samples = num_samples - num_train_samples

    # Use random_split to divide the dataset into train and validation sets
    train_dataset, val_dataset = random_split(custom_dataset, [num_train_samples, num_val_samples])

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers, collate_fn=custom_dataset.batch_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers, collate_fn=custom_dataset.batch_collate_fn)  # No need to shuffle validation data


    # model - initiallize to recieve input length as 9 for x,y,z,xy,yz,zx,xx,yy,zz
    # model = PointNet_FC(k=9)
    # model = STNkd(k=9)
    num_layers = 3
    hidden_channels = 128
    in_channels = 3
    out_channels = 2
    model = PointTransformerConvNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
    # model_path = "C:/Users\galyo\Documents\Computer science\M.Sc\Projects\DeepSignatureProject\deep-signature-2/trained_models\model_point_transformer_3_layers_width_128-epoch=99.ckpt"

    # model = PointTransformerConvNet.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
    # model.eval()

    # training
    logger = init_wandb(lr=lr,max_epochs=max_epochs, weight_decay=weight_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # Directory to save the checkpoints
        filename='model_point_transformer_'+str(num_layers)+'_layers_width_'+str(hidden_channels)+'-{epoch:02d}',
        save_top_k=1,  # Save all checkpoints
        save_on_train_epoch_end=True,
        every_n_epochs=10
    )
    # visualizer_callback = VisualizerCallback(radius=0.5, sample=data[0][0])
    trainer = Trainer(num_nodes=1,
                      gradient_clip_val=1.0,
                      # log_every_n_steps=1,
                      accelerator='auto',
                      # overfit_batches=1.0,
                      max_epochs=max_epochs,
                      logger=logger,
                      # callbacks=[visualizer_callback, checkpoint_callback])
                      callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == "__main__":
    main_loop()
