import torch.nn.functional as F
from tqdm import tqdm

data = torch.load("res_species_breed_bbox_50_ClassifierHead(2)_GradCAM_idx46_cams.pt")

def resize_tensor(tensor, size=(64, 64), mode='bilinear'):
    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)
    else:
        return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode).squeeze(0)

# Resize and store all samples
resized_data = []
cam_threshold = 0.5  # ← Set your binarization threshold

for image, cam, mask in tqdm(data, desc="Resizing and Binarizing CAMs"):
    image_resized = resize_tensor(image, size=(64, 64), mode='bilinear')      # [3, H, W]
    cam_resized   = resize_tensor(cam, size=(64, 64), mode='bilinear')        # [1, H, W]
    mask_resized  = resize_tensor(mask, size=(64, 64), mode='nearest')        # [1, H, W]

    cam_binary = (cam_resized > cam_threshold).float()  # Binarize CAM

    resized_data.append((image_resized, cam_binary, mask_resized))

# Optionally convert to TensorDataset
resized_dataset = torch.utils.data.TensorDataset(
    torch.stack([x[0] for x in resized_data]),  # images
    torch.stack([x[1] for x in resized_data]),  # binarized CAMs
    torch.stack([x[2] for x in resized_data]),  # masks
)

# Save to local file
save_path = "resized_64_species_breed_cam_mask_binary.pt"
torch.save(resized_dataset, save_path)
print(f"\n✅ Saved resized and binarized dataset to: {save_path}")