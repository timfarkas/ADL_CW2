import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import get_binary_from_normalization
import numpy as np
import cv2
data = torch.load("res_species_breed_bbox_50_ClassifierHead(2)_GradCAM_idx46_cams.pt")

# Resize helper
def resize_tensor(tensor, size=(64, 64), mode='nearest'):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode).squeeze(0)

# Binarize mask (everything > 0 becomes 1, rest 0)
resized_data = []

for i, (image, cam, mask) in enumerate(tqdm(data, desc="Resizing + Binarizing")):
    image_resized = resize_tensor(image, size=(64, 64), mode='nearest')     # [3, H, W]
    cam_resized   = resize_tensor(cam, size=(64, 64), mode='nearest')       # [1, H, W]
    mask_resized  = resize_tensor(mask, size=(64, 64), mode='nearest')      # [1, H, W]
    # print(f"Sample {i} - Unique values in mask_resized:", torch.unique(mask_resized))

                            # threshold to 0/

    # Fill inner black holes using OpenCV flood fill
    def patch_fill(mask_sized, fg_value=0.0118, boundary_value=0.0078):
        """
        Fill in background pixels that are surrounded by foreground and boundary.
        Works with masks using soft labels like 0.0039 (bg), 0.0078 (boundary), 0.0118 (fg).
        """
        mask_np = mask_sized.clone()

        # Create padding for edge handling
        padded = F.pad(mask_np, (1, 1, 1, 1), mode='constant', value=0)

        filled = mask_np.clone()
        h, w = mask_np.shape[-2:]

        for i in range(1, h + 1):
            for j in range(1, w + 1):
                center = padded[0, i, j]

                if center < boundary_value:  # background
                    # Extract 3x3 neighborhood
                    neighborhood = padded[0, i - 1:i + 2, j - 1:j + 2].flatten()

                    has_fg = (neighborhood == fg_value).any()
                    has_boundary = (neighborhood == boundary_value).any()

                    if has_fg and has_boundary:
                        filled[0, i - 1, j - 1] = fg_value  # Fill as foreground

        return filled
    mask_cleaned=patch_fill(mask_resized)

    resized_data.append((image_resized, cam_resized, mask_cleaned))

# Save the dataset
torch.save(resized_data, "resized_64_species_breed_cam_mask.pt")
print("✅ Saved resized dataset with binarized masks.")