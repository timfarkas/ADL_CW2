import matplotlib.pyplot as plt
import torch
import numpy as np
from experiment import get_MCG  # Assuming this returns a [H, W] float32 map in [0, 1]

# Load the dataset
data = torch.load("resized_64_species_breed_cam_mask.pt")

# Unnormalization constants
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for idx, (image, cam, mask) in enumerate(data):
    print(f"Sample {idx}")
    print(f"Image shape: {image.shape}, CAM shape: {cam.shape}, Mask shape: {mask.shape}")

    # Unnormalize and convert image to uint8 for MCG
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    img_bgr_uint8 = (img_np * 255).astype(np.uint8)

    # Get MCG segmentation proposal (should be [H, W], float32 in [0, 1])
    mcg_mask = get_MCG(img_bgr_uint8, threshold=0)  # Assume it's already resized to 64x64

    # Resize and normalize cam if needed
    cam_np = cam.squeeze().numpy()  # [H, W], should be in [0, 1]

    # Mix CAM and MCG equally
    mixed_mask = 0.5 * cam_np + 0.5 * mcg_mask
    mixed_mask = np.clip(mixed_mask, 0.0, 1.0)  # Ensure valid range

    # Visualize
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img_np)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(cam_np, cmap='jet')
    axs[1].set_title("CAM")
    axs[1].axis("off")

    axs[2].imshow(mcg_mask, cmap='gray')
    axs[2].set_title("MCG")
    axs[2].axis("off")

    axs[3].imshow(mixed_mask, cmap='gray')
    axs[3].set_title("Mixed Mask (0.5 CAM + 0.5 MCG)")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

    if idx == 10:  # Limit to first 10 samples
        break