import os

import torch
import torch.nn as nn
from PIL import ImageFile
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from CAM.cam_model import CNN, ResNetBackbone, fit_sgd
from models import CAMManager, SelfTraining, UNet

from data import OxfordPetDataset,create_dataloaders,create_sample_loader_from_existing_loader
from evaluation import evaluate_dataset, evaluate_model, get_binary_from_normalization
from utils import save_tensor_dataset, unnormalize

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
gt_sample_loader = create_sample_loader_from_existing_loader(dataloader_val,100,64,False)
dataset = dataloader_train.dataset  # get the dataset used inside the original DataLoader
# for i in range(5):  # check first 5 samples
#     img, mask = dataset[i]
#     seg = mask["segmentation"]
#
#     print(f"Sample {i}:")
#     print(f"  shape: {seg.shape}")
#     print(f"  dtype: {seg.dtype}")
#     print(f"  min: {seg.min().item():.4f}, max: {seg.max().item():.4f}")
#     print(f"  unique values: {torch.unique(seg)}\n")
# Create a new dataset that returns (img, segmentation_mask, segmentation_mask)
tripled_data = [
    (img, get_binary_from_normalization(mask["segmentation"]), get_binary_from_normalization(mask["segmentation"]),)
    for img, mask in dataset
]

# Wrap it in a new DataLoader
dataloader_train_triple = DataLoader(
    tripled_data,
    batch_size=dataloader_train.batch_size,
    shuffle=False,  # or True if you want
    num_workers=0)

#
# images, masks, masks_gt = next(iter(dataloader_train_triple))

# # Plot the first N samples
# images_vis = images[:8].detach().cpu()
# masks_vis = masks[:8].detach().cpu()
# preds_vis = masks_gt[:8].float().detach().cpu()
#
#
# for i in range(images_vis.size(0)):
#     fig, axs = plt.subplots(1, 3, figsize=(10, 3))
#
#     img = images_vis[i].permute(1, 2, 0).numpy()
#     img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for visualization
#
#     axs[0].imshow(img)
#     axs[0].set_title("Input Image")
#     axs[0].axis("off")
#
#     axs[1].imshow(masks_vis[i].squeeze(), cmap='gray')
#     axs[1].set_title("Ground Truth Mask")
#     axs[1].axis("off")
#
#     axs[2].imshow(preds_vis[i].squeeze(), cmap='gray')
#     axs[2].set_title("Predicted Mask")
#     axs[2].axis("off")
#
#     plt.tight_layout()
#     plt.show()

model_new = UNet(3, 1).to(device)

# print(f"Dataset: {len(dataloader_bootstrap.dataset)} samples")
loss_function = nn.BCEWithLogitsLoss()
# loss_function = nn.MSELoss()

epochs_previous=0
# model_path_previous = os.path.join(model_dir, f"baseline_model_epoch{epochs_previous}.pt")
# model_path = os.path.join(model_dir, f"best_model_selftrain.pt")
model_new = UNet(3, 1).to(device)
# model_new.load_state_dict(torch.load(f"{model_path_previous}"))
epochs=40
model_path= os.path.join(model_dir, f"baseline_model_epoch{epochs}.pt")
SelfTraining.fit_sgd_pixel(
    model_new, dataloader_train_triple, epochs-epochs_previous, 0.05, loss_function, model_path, device=device
)

