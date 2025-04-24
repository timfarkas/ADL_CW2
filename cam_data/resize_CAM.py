import torch
import torch.nn.functional as F
data = torch.load("res_species_breed_bbox_50_ClassifierHead(2)_GradCAM_idx46_cams.pt")

def resize_tensor(tensor, size=(64, 64), mode='nearest'):
    return F.interpolate(tensor.unsqueeze(0), size=size, mode=mode).squeeze(0)

resized_data = []

for image, cam, mask in data:
    image_resized = resize_tensor(image, size=(64, 64), mode="nearest")
    cam_resized   = resize_tensor(cam,   size=(64, 64), mode="nearest")
    mask_resized  = resize_tensor(mask,  size=(64, 64), mode="nearest")
    resized_data.append((image_resized, cam_resized, mask_resized))

torch.save(resized_data, "resized_64_species_breed_cam_mask_raw.pt")
print("Saved resized dataset with binarized masks.")