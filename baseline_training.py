# AI usage statement: AI was used to assist in debugging

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import SelfTraining, UNet
from torch.utils.data import DataLoader, TensorDataset
from data import create_dataloaders
from evaluation import  get_binary_from_normalization,remap_mask

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_dir = os.path.join("checkpoints", "EVA")
os.makedirs(model_dir, exist_ok=True)

classification_mode = "breed"
batch_size = 64
dataloader_train, dataloader_val, dataloader_test = create_dataloaders(
    batch_size,
    resize_size=64,
    target_type=[classification_mode, "segmentation"],
    lazy_loading=True,
    shuffle=False)

dataset = dataloader_train.dataset  # get the dataset used inside the original DataLoader

tripled_data = [
    (
        img,
        remap_mask(mask["segmentation"]).unsqueeze(0),
        remap_mask(mask["segmentation"]).unsqueeze(0),
    )
    for img, mask in dataset
]

# Wrap it in a new DataLoader
dataloader_train_triple = DataLoader(
    tripled_data,
    batch_size=dataloader_train.batch_size,
    shuffle=False,  # or True if you want
    num_workers=0,
)
model_new = UNet(3, 1).to(device)

loss_function = nn.BCEWithLogitsLoss()

model_path = os.path.join(model_dir, "baseline_model.pt")



epochs_previous=0
# model_path_previous = os.path.join(model_dir, f"baseline_model_epoch{epochs_previous}.pt")
model_new = UNet(3, 1).to(device)
# model_new.load_state_dict(torch.load(f"{model_path_previous}"))
epochs=1
model_path= os.path.join(model_dir, f"baseline_model_epoch{epochs}.pt")
SelfTraining.fit_sgd_pixel(
    model_new, dataloader_train_triple, epochs-epochs_previous, 0.05, loss_function, model_path, device=device
)

