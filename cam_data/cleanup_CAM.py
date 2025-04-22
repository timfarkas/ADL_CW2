# AI usage statement: AI was used to assist in building the filtering algorithm (Although not being very helpful)

import torch
import numpy as np

def promote_background_to_boundary(mask_tensor,
                                    fg_value=0.0039,
                                    bg_value=0.0078,
                                    boundary_value=0.0118,
                                    tolerance=0.001):

    mask_np = mask_tensor.squeeze().numpy()
    padded = np.pad(mask_np, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    updated = mask_np.copy()
    h, w = mask_np.shape

    fg_min, fg_max = fg_value - tolerance, fg_value + tolerance
    bnd_min, bnd_max = boundary_value - tolerance, boundary_value + tolerance
    bg_min, bg_max = bg_value - tolerance, bg_value + tolerance

    changed = 0
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            center = padded[i, j]
            if bg_min <= center <= bg_max:
                neighborhood = padded[i - 1:i + 2, j - 1:j + 2]
                has_fg = ((fg_min <= neighborhood) & (neighborhood <= fg_max)).any()
                has_bnd = ((bnd_min <= neighborhood) & (neighborhood <= bnd_max)).any()
                if has_fg and has_bnd:
                    updated[i - 1, j - 1] = boundary_value
                    changed += 1

    return torch.tensor(updated, dtype=mask_tensor.dtype).unsqueeze(0)

data = torch.load("resized_64_species_breed_cam_mask_raw.pt")


cleaned_data = []
for image, cam, mask in data:
    patched_mask = promote_background_to_boundary(mask)
    cleaned_data.append((image, cam, patched_mask))
    # if i == 0:  # visualize only the first sample
    #     plt.imshow(patched_mask.squeeze().numpy(), cmap="gray")
    #     plt.title("Patched Mask - Sample 0")
    #     plt.axis("off")
    #     plt.show()

torch.save(cleaned_data, "resized_64_species_breed_cam_mask.pt")
print("Saved cleaned dataset with patch-filled masks.")