from custom_data import OxfordPetDataset
from simple_dataloader import train_loader, val_loader, test_loader
import custom_loader
import os
from CAM.cam_model import ResNetBackbone, CNN, fit_sgd, visualize_cam, CAMWrapper
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# dataset = OxfordPetDataset().prepare_dataset()
# print(f"Classes found: {len(dataset.class_names)}")  # Should be 37
# print(f"Samples: {len(dataset.get_all_data())}")
# classification_mode = "breed"
# dataset = OxfordPetDataset().prepare_dataset()

sample_count = 0
for images, labels in train_loader:
    sample_count += images.size(0)
print(f"Total number of training samples: {sample_count}")

import matplotlib.pyplot as plt

# Optional: label map for better readability
# label_map = dataset.breed_label_map if classification_mode == "breed" else {0: "cat", 1: "dog"}

# Get the first batch
# batch = next(iter(train_loader))
# images, labels = batch
#
# print(labels)
#
#
# from collections import Counter
#
# # Initialize a counter
# label_counter = Counter()
#
# # Full pass over the entire DataLoader
# for images, labels in train_loader:
#     label_counter.update(labels.tolist())  # Convert tensor to list
# label_map = {idx: name for idx, name in enumerate(dataset.class_names)}
#
# print(f"\nLabel counts across entire training set:\n")
# for label_idx in sorted(label_counter):
#     label_name = label_map.get(label_idx, f"Label {label_idx}")
#     count = label_counter[label_idx]
#     print(f"{label_idx:2d} ({label_name:25}) → {count} samples")
#
# total_samples = sum(label_counter.values())
# print(f"\nTotal training samples: {total_samples}")
# # Unnormalize function (if needed)
# # def unnormalize(img_tensor):
# #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# #     return (img_tensor * std + mean).clamp(0, 1)
# #
# # # Show batch
# # def show_batch(images, labels, label_map, n=8):
# #     images = images[:n]
# #     labels = labels[:n]
# #     fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
# #     for i in range(n):
# #         img = unnormalize(images[i].cpu())
# #         label_idx = labels[i].item()
# #         label_name = label_map[label_idx] if label_map else str(label_idx)
# #
# #         axes[i].imshow(img.permute(1, 2, 0))
# #         axes[i].set_title(label_name, fontsize=10)
# #         axes[i].axis('off')
# #     plt.tight_layout()
# #     plt.show()
# #
# # # Call the function
# # show_batch(images, labels)
