from data import OxfordPetDataset
import data
import json
import os

from models import ResNetBackbone, CNNBackbone, BboxHead, ClassifierHead
import torch
import torch.nn as nn
import sys
import io
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import compute_accuracy, computeBBoxIoU, convertVOCBBoxFormatToAnchorFormat

# data = torch.load("res_species_breed_bbox_50_ClassifierHead(2)_GradCAM_idx46_cams.pt")

import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the saved data


data = torch.load("resized_64_species_breed_cam_mask.pt")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# ImageNet mean and std (commonly used for pretrained models like ResNet)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Pick an index

image, _, _ = data[idx]

# Convert to numpy and un-normalize
img_np = image.permute(1, 2, 0).numpy()  # [H, W, C]
img_np = (img_np * std) + mean  # unnormalize
img_np = np.clip(img_np, 0, 1)  # ensure valid range

# Save
plt.imsave("sample_image_unnormalized.png", img_np)
print("Saved unnormalized image as sample_image_unnormalized.png")

#
# first_entry = data[0]
# print(f"Type of first entry: {type(first_entry)}")
# print(f"Length of tuple: {len(first_entry)}")
#
# for i, item in enumerate(first_entry):
#     print(f"Item {i}: Type: {type(item)}, Shape: {getattr(item, 'shape', 'N/A')}")
#
# all_cams = torch.stack([entry[1] for entry in data])    # shape: [5173, 1, 256, 256]
# all_images = torch.stack([entry[0] for entry in data])  # shape: [5173, 3, 256, 256]
#
# # CAM value range
# print(f"CAM value range: {all_cams.min().item()} to {all_cams.max().item()}")
# print(f"CAM mean ± std: {all_cams.mean().item():.4f} ± {all_cams.std().item():.4f}")
#
# # Image value range
# print(f"Image value range: {all_images.min().item()} to {all_images.max().item()}")
# print(f"Image mean ± std (overall): {all_images.mean().item():.4f} ± {all_images.std().item():.4f}")
#
# # Optional: Per-channel image mean/std (RGB)
# means = all_images.mean(dim=(0, 2, 3))  # [C]
# stds = all_images.std(dim=(0, 2, 3))
# print(f"Image mean per channel: {means}")
# print(f"Image std per channel: {stds}")
#
# # Check type and length
# # Basic stats
# print(f"Total samples: {len(data)}")
#
# # Extract all masks (flattened) for stats
# all_masks = torch.stack([entry[2] for entry in data])  # shape: [5173, 1, 256, 256]
# print(f"Mask value range: {all_masks.min().item()} to {all_masks.max().item()}")
# print(f"Mask unique values: {torch.unique(all_masks)}")
#
# # Optional: Proportion of foreground pixels across dataset
# mask_mean = all_masks.float().mean().item()
# print(f"Average foreground pixel ratio: {mask_mean:.4f}")
# # Visualize a batch
# def show_image_and_cam_mask(img, cam=None, mask=None, title=""):
#     img = img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
#     plt.imshow(img)
#     if cam is not None:
#         plt.imshow(cam.squeeze().numpy(), cmap='jet', alpha=0.4)
#     if mask is not None:
#         plt.imshow(mask.squeeze().numpy(), cmap='gray', alpha=0.3)
#     plt.title(title)
#     plt.axis('off')
#
# plt.figure(figsize=(12, 8))
# for i in range(4):
#     image, cam, mask = data[i]
#     plt.subplot(2, 2, i + 1)
#     show_image_and_cam_mask(image, cam=cam, mask=mask, title=f"Sample {i}")
# plt.tight_layout()
# plt.show()