import pickle
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import init_wandb

from data.triplet_dataset import CustomTripletDataset
from models.pointnet.model import STNkd, PointNet_FC
from visualize.visualize__results_on_mesh import VisualizerCallback


def main_loop():
    max_epochs = 100
    lr = 0.001
    weight_decay = 0.1
    file_path = "/home/gal.yona/diffusion-net/src/diffusion_net/triplets_data_size_30_N_100000_all_monge_patch.pkl"
    # file_path = "./triplets_data_size_30_N_10_all_monge_patch.pkl"




    # Load the triplets from the file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Create custom dataset
    custom_dataset = CustomTripletDataset(data)

    # Define the ratio for train and validation split (e.g., 80% for training, 20% for validation)
    train_ratio = 0.8
    val_ratio = 1 - train_ratio

    # Calculate the number of samples for train and validation sets
    num_samples = len(custom_dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_val_samples = num_samples - num_train_samples

    # Use random_split to divide the dataset into train and validation sets
    train_dataset, val_dataset = random_split(custom_dataset, [num_train_samples, num_val_samples])

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=8, collate_fn=custom_dataset.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=8, collate_fn=custom_dataset.custom_collate_fn)  # No need to shuffle validation data


    # model - initiallize to recieve input length as 9 for x,y,z,xy,yz,zx,xx,yy,zz
    model = PointNet_FC(k=9)

    # training
    logger = init_wandb(lr=lr,max_epochs=max_epochs, weight_decay=weight_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # Directory to save the checkpoints
        filename='model-{epoch:02d}',
        save_top_k=1,  # Save all checkpoints
        save_on_train_epoch_end=True,
        every_n_epochs=20
    )
    visualizer_callback = VisualizerCallback(radius=0.5, sample=data[0][1])
    trainer = Trainer(num_nodes=1,
                      gradient_clip_val=1.0,
                      # log_every_n_steps=1,
                      accelerator='auto',
                      # overfit_batches=1.0,
                      max_epochs=max_epochs,
                      logger=logger,
                      callbacks=[visualizer_callback, checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == "__main__":
    main_loop()
