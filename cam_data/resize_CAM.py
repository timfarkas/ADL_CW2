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
    resized_data.append((image_resized, cam_resized, mask_resized))

# Save the dataset
torch.save(resized_data, "resized_64_species_breed_cam_mask_raw.pt")
print("✅ Saved resized dataset with binarized masks.")