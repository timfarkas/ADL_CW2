# AI usage statement: AI was used to assist in visualisation

import torch
import matplotlib.pyplot as plt

dataset = torch.load("resized_64_species_breed_cam_mask.pt")

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

n_samples = 8
rows = 3
cols = n_samples

fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

for i in range(n_samples):
    image_tensor, cam_tensor, label_tensor = dataset[i]

    image = image_tensor
    print(f"Sample {i} - Unique values in label:", torch.unique(label_tensor))

    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = image * std + mean
    image_np = image.permute(1, 2, 0).clamp(0, 1).numpy()

    cam_np = cam_tensor[0].numpy()

    label_np = label_tensor.squeeze().numpy()

    axs[0, i].imshow(image_np)
    axs[0, i].set_title(f"Image {i}")
    axs[0, i].axis("off")

    axs[1, i].imshow(cam_np, cmap="jet")
    axs[1, i].set_title("CAM")
    axs[1, i].axis("off")

    axs[2, i].imshow(label_np, cmap="gray")
    axs[2, i].set_title("GT")
    axs[2, i].axis("off")

plt.tight_layout()
plt.show()
